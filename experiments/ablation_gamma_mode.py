#!/usr/bin/env python3
"""Quick γ-mode ablation: CIFAR-100 MobileNetV2.

Five variants compared:
  * nelu_pl           : per-layer scalar γ, learnable from start
  * nelu_pc           : per-channel γ, learnable from start
  * nelu_sched        : γ on cosine warmup then HOLD, never learnable
  * nelu_schedlearn_pl: cosine warmup to 1.0 (frozen), then per-layer learnable
  * nelu_schedlearn_pc: cosine warmup to 1.0 (frozen), then per-channel learnable

The schedlearn variants fix a failure mode of direct learnable γ with
deep init (1e-4): gradient through softplus vanishes, and starting from
scratch at γ=1.0 diverges on ImageNet. schedlearn sidesteps both: γ is
ramped up on a schedule while frozen (no gradient), then unfrozen at
γ≈1.0 where the gradient landscape is healthy.

Seeds: 3 / 3 / 2 / 2 / 2 by default (12 runs total).

Math (NCHW CIFAR — rms over all non-batch dims, as in the original
ResAct CIFAR recipe that produced NELU wins 7/7):
  y = z * Phi(γ * z / rms(z)),   rms over dim=(1,2,3) for 4D

Usage:
  python experiments/ablation_gamma_mode.py --wandb
  python experiments/ablation_gamma_mode.py --modes nelu_pl nelu_pc --seeds 42 123 456
  python experiments/ablation_gamma_mode.py --epochs 100 --modes nelu_sched --seeds 42 123
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Reuse the existing MobileNetV2 builder + CIFAR-100 loader from the main
# training script so we stay consistent with the prior CIFAR results.
from experiments.main_cifar_tinyimagenet import (
    set_seed, get_cifar100, build_mobilenetv2,
    CIFAR100_MEAN, CIFAR100_STD,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ═══════════════════════════════════════════════════════════════════
#  NELU variants
# ═══════════════════════════════════════════════════════════════════

class _BaseNELU(nn.Module):
    """Shared rms (dim=(1,2,3) for 4D NCHW) + Phi gate."""
    eps: float = 1e-6

    def _rms(self, z: torch.Tensor) -> torch.Tensor:
        dim = (1, 2, 3) if z.dim() == 4 else -1
        return z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()

    def _gate(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(t * INV_SQRT2))


class NELU_PerLayer(_BaseNELU):
    """γ is a single scalar nn.Parameter per module. Learnable."""
    def __init__(self, eps: float = 1e-6, gamma_init: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        t = self.gamma * z / self._rms(z)
        return z * self._gate(t)

    def extra_repr(self):
        return f"gamma={self.gamma.item():.4f}"


class NELU_PerChannel(_BaseNELU):
    """γ shape (C,), lazy-materialized on first forward.

    Caller must do ONE dummy forward before creating the optimizer so
    the uninitialized parameter gets registered with a real shape.
    """
    def __init__(self, eps: float = 1e-6, gamma_init: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.gamma_init = float(gamma_init)
        self.gamma = nn.UninitializedParameter()

    def _maybe_init(self, z: torch.Tensor):
        if isinstance(self.gamma, nn.UninitializedParameter):
            C = z.size(1) if z.dim() == 4 else z.size(-1)
            self.gamma.materialize((C,))
            with torch.no_grad():
                self.gamma.fill_(self.gamma_init)
            # Ensure correct dtype/device (materialize preserves device)
            self.gamma.data = self.gamma.data.to(dtype=torch.float32,
                                                 device=z.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self._maybe_init(z)
        if z.dim() == 4:
            g = self.gamma.view(1, -1, 1, 1)
        else:
            shape = [1] * z.dim()
            shape[-1] = self.gamma.numel()
            g = self.gamma.view(*shape)
        t = g * z / self._rms(z)
        return z * self._gate(t)

    def extra_repr(self):
        if isinstance(self.gamma, nn.UninitializedParameter):
            return "gamma=uninit"
        return f"C={self.gamma.numel()}, gamma_mean={self.gamma.mean().item():.4f}"


class NELU_Scheduled(_BaseNELU):
    """γ is a non-persistent buffer, set externally by the training loop."""
    def __init__(self, eps: float = 1e-6, gamma_init: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "gamma",
            torch.tensor(float(gamma_init)),
            persistent=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        t = self.gamma * z / self._rms(z)
        return z * self._gate(t)

    def extra_repr(self):
        return f"gamma={self.gamma.item():.4f} [scheduled]"


class NELU_SchedLearn_PL(_BaseNELU):
    """γ_effective = schedule(t) * γ_learnable,  γ_learnable is a scalar init 1.0.

    The schedule is a buffer (set externally by the training loop each
    epoch). γ_learnable is always a Parameter, so it receives gradient
    throughout training. During warmup the effective gradient on
    γ_learnable is attenuated by `schedule(t)` (small early, full by
    warmup end), which mimics a curriculum without any freeze/unfreeze
    phase transition. Post-warmup schedule = 1, effective γ = γ_learnable.

    Starting γ_learnable at 1.0 (not 1e-4) means the learnable scale
    is at a healthy gradient magnitude from the start — no softplus
    dead-zone issue.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "schedule", torch.tensor(1e-4, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        g_eff = self.schedule * self.gamma
        t = g_eff * z / self._rms(z)
        return z * self._gate(t)

    def extra_repr(self):
        return (f"gamma(learn)={self.gamma.item():.4f}  "
                f"schedule={self.schedule.item():.4f}  "
                f"eff={self.schedule.item() * self.gamma.item():.4f}")


class NELU_SchedLearn_PC(_BaseNELU):
    """Per-channel version: γ_effective = schedule(t) * γ_learnable_{c}.

    γ_learnable is a length-C vector initialized to 1.0, materialized
    lazily on first forward. Caller must do one dummy forward before
    constructing the optimizer.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.UninitializedParameter()
        self.register_buffer(
            "schedule", torch.tensor(1e-4, dtype=torch.float32),
            persistent=False,
        )

    def _maybe_init(self, z: torch.Tensor):
        if isinstance(self.gamma, nn.UninitializedParameter):
            C = z.size(1) if z.dim() == 4 else z.size(-1)
            self.gamma.materialize((C,))
            with torch.no_grad():
                self.gamma.fill_(1.0)
            self.gamma.data = self.gamma.data.to(dtype=torch.float32,
                                                 device=z.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self._maybe_init(z)
        if z.dim() == 4:
            g_view = self.gamma.view(1, -1, 1, 1)
        else:
            shape = [1] * z.dim()
            shape[-1] = self.gamma.numel()
            g_view = self.gamma.view(*shape)
        g_eff = self.schedule * g_view
        t = g_eff * z / self._rms(z)
        return z * self._gate(t)

    def extra_repr(self):
        if isinstance(self.gamma, nn.UninitializedParameter):
            return "gamma=uninit"
        return (f"C={self.gamma.numel()}, "
                f"gamma_mean={self.gamma.mean().item():.4f}  "
                f"schedule={self.schedule.item():.4f}")


_MODE_CLS = {
    "nelu_pl":            NELU_PerLayer,
    "nelu_pc":            NELU_PerChannel,
    "nelu_sched":         NELU_Scheduled,
    "nelu_schedlearn_pl": NELU_SchedLearn_PL,
    "nelu_schedlearn_pc": NELU_SchedLearn_PC,
}


# Module tuples for isinstance checks
_SCHEDLEARN_CLASSES = (NELU_SchedLearn_PL, NELU_SchedLearn_PC)


def set_scheduled_gamma(model: nn.Module, value: float) -> int:
    """Pure-schedule variant: fill NELU_Scheduled.gamma buffer directly."""
    n = 0
    for m in model.modules():
        if isinstance(m, NELU_Scheduled):
            with torch.no_grad():
                m.gamma.fill_(float(value))
            n += 1
    return n


def set_schedule_multiplier(model: nn.Module, value: float) -> int:
    """schedlearn variants: update the `schedule` buffer (multiplier on γ_l)."""
    n = 0
    for m in model.modules():
        if isinstance(m, _SCHEDLEARN_CLASSES):
            with torch.no_grad():
                m.schedule.fill_(float(value))
            n += 1
    return n


def gamma_schedule(epoch: int,
                   warmup_epochs: int,
                   g_start: float = 1e-4,
                   g_end: float = 1.0,
                   curve: str = "cosine") -> float:
    """γ warmup schedule: rises from g_start to g_end over `warmup_epochs`,
    then HOLDS at g_end for the rest of training.

    The philosophy mirrors LR warmup: γ=1 is the target, but starting
    from γ=1 on epoch 0 is unstable, so we ramp up over a short early
    window and then use full γ for the bulk of training.

    Default warmup_epochs=40 (20% of a 200-epoch CIFAR run) gives 160
    epochs at full γ — enough time for the model to actually learn the
    gated activation, instead of spending most of training near-linear.
    """
    if epoch >= warmup_epochs:
        return g_end
    t = epoch / max(1, warmup_epochs)
    if curve == "cosine":
        return g_start + (g_end - g_start) * 0.5 * (1 - math.cos(math.pi * t))
    elif curve == "linear":
        return g_start + (g_end - g_start) * t
    else:
        raise ValueError(f"unknown curve: {curve!r}")


# ═══════════════════════════════════════════════════════════════════
#  Activation replacement
# ═══════════════════════════════════════════════════════════════════

_REPLACE_TYPES = (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU)


def replace_activations(model: nn.Module, mode: str) -> nn.Module:
    target_cls = _MODE_CLS[mode]
    for name, child in model.named_children():
        if isinstance(child, _REPLACE_TYPES):
            setattr(model, name, target_cls())
        else:
            replace_activations(child, mode)
    if isinstance(model, nn.Sequential):
        for i, child in enumerate(model):
            if isinstance(child, _REPLACE_TYPES):
                model[i] = target_cls()
    return model


def count_gammas(model: nn.Module):
    """Report effective-γ stats on all NELU modules in `model`.

    For schedlearn variants the effective value is schedule * γ_learn.
    For pure-schedule the effective value is the gamma buffer.
    For always-learnable variants the effective value is just γ.
    """
    gammas = []       # effective γ (what the activation actually uses)
    learn_mag = []    # raw learnable γ magnitude (for schedlearn variants)
    sched_val = None
    for m in model.modules():
        if isinstance(m, NELU_PerLayer):
            gammas.append(m.gamma.detach().float().cpu().item())
        elif isinstance(m, NELU_PerChannel):
            if not isinstance(m.gamma, nn.UninitializedParameter):
                gammas.append(m.gamma.detach().float().cpu().mean().item())
        elif isinstance(m, NELU_Scheduled):
            gammas.append(m.gamma.detach().float().cpu().item())
        elif isinstance(m, NELU_SchedLearn_PL):
            s = float(m.schedule.item())
            g = float(m.gamma.detach().cpu().item())
            gammas.append(s * g)
            learn_mag.append(g)
            sched_val = s
        elif isinstance(m, NELU_SchedLearn_PC):
            if isinstance(m.gamma, nn.UninitializedParameter):
                continue
            s = float(m.schedule.item())
            g_mean = float(m.gamma.detach().float().cpu().mean().item())
            gammas.append(s * g_mean)
            learn_mag.append(g_mean)
            sched_val = s
    if not gammas:
        return {}
    arr = np.array(gammas)
    out = {
        "gamma_eff_mean": float(arr.mean()),
        "gamma_eff_min":  float(arr.min()),
        "gamma_eff_max":  float(arr.max()),
        "gamma_eff_std":  float(arr.std()),
        # Backward-compat keys (used by wandb/history)
        "gamma_mean": float(arr.mean()),
        "gamma_min":  float(arr.min()),
        "gamma_max":  float(arr.max()),
        "gamma_std":  float(arr.std()),
        "n_modules":  int(len(gammas)),
    }
    if learn_mag:
        lm = np.array(learn_mag)
        out["gamma_learnable_mean"] = float(lm.mean())
        out["gamma_learnable_min"]  = float(lm.min())
        out["gamma_learnable_max"]  = float(lm.max())
        out["schedule_value"] = float(sched_val) if sched_val is not None else 1.0
    return out


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_one_run(mode: str, seed: int, args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # Data
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    train_ds, test_ds, num_classes = get_cifar100(data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=256,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    # Model (MobileNetV2 has ReLU6 activations)
    model = build_mobilenetv2(num_classes=num_classes, img_size=32)
    replace_activations(model, mode)
    model.to(device)

    # Dummy forward to trigger lazy-init for ANY per-channel variant
    # (nelu_pc uses UninitializedParameter directly; nelu_schedlearn_pc
    # also uses it for its learnable γ). Without this, `model.parameters()`
    # would crash on .numel() of the uninitialized tensor.
    if mode in ("nelu_pc", "nelu_schedlearn_pc"):
        model.eval()
        with torch.no_grad():
            x_dummy = torch.zeros(2, 3, 32, 32, device=device)
            model(x_dummy)
        model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[mode={mode}, seed={seed}]  params={param_count/1e6:.2f}M  "
          f"γ-init stats: {count_gammas(model)}")

    # Optimizer + schedule: SGD recipe from main_cifar_tinyimagenet.py CNN default
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    milestones = [int(args.epochs * 0.3),
                  int(args.epochs * 0.6),
                  int(args.epochs * 0.8)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                               gamma=0.2)

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    wandb_run = None
    if args.wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project="nelu",
            name=f"cifar100_mbv2_{mode}_s{seed}",
            config={
                "arch": "mobilenetv2",
                "dataset": "cifar100",
                "mode": mode,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "params": param_count,
            },
            tags=["ablation", "cifar100", "mobilenetv2", "gamma-mode", mode],
            reinit=True,
        )

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_acc": [], "gamma_mean": [], "gamma_min": [],
               "gamma_max": [], "gamma_std": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        # γ scheduling — unified for all scheduled variants
        if mode in ("nelu_sched", "nelu_schedlearn_pl", "nelu_schedlearn_pc"):
            g_t = gamma_schedule(
                epoch, warmup_epochs=args.gamma_warmup_epochs,
                g_start=args.gamma_start,
                g_end=args.gamma_end,
                curve=args.gamma_curve,
            )
            if mode == "nelu_sched":
                # Pure: the γ buffer IS the effective value
                set_scheduled_gamma(model, g_t)
            else:
                # schedlearn: update the schedule multiplier; γ_learnable
                # is always a Parameter and always receives gradient
                # (attenuated by schedule during warmup).
                set_schedule_multiplier(model, g_t)
        # LR warmup: first epoch linearly from 0 to lr
        if epoch == 0:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1
        elif epoch == 1:
            for g in optimizer.param_groups:
                g["lr"] = args.lr

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc  = 100.0 * correct / total

        # Eval
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if args.amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = model(x)
                else:
                    out = model(x)
                _, pred = out.max(1)
                t_correct += pred.eq(y).sum().item()
                t_total += x.size(0)
        test_acc = 100.0 * t_correct / t_total
        best_acc = max(best_acc, test_acc)

        if epoch >= 1:  # only step after warmup
            scheduler.step()

        # Record
        gstats = count_gammas(model)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["gamma_mean"].append(gstats.get("gamma_mean", float("nan")))
        history["gamma_min"].append(gstats.get("gamma_min", float("nan")))
        history["gamma_max"].append(gstats.get("gamma_max", float("nan")))
        history["gamma_std"].append(gstats.get("gamma_std", float("nan")))

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  ep {epoch+1:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.2f}  "
                  f"test_acc={test_acc:.2f}  "
                  f"best={best_acc:.2f}  "
                  f"γ_mean={gstats.get('gamma_mean', 0):.4f}")

        if wandb_run is not None:
            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc":  train_acc,
                "test/acc":   test_acc,
                "test/best_acc": best_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log.update({f"gamma/{k}": v for k, v in gstats.items()})
            wandb_run.log(log)

    elapsed = time.time() - t0
    print(f"  [mode={mode}, seed={seed}]  done in {elapsed:.0f}s  best={best_acc:.2f}%")

    final_gstats = count_gammas(model)
    result = {
        "mode": mode,
        "seed": seed,
        "epochs": args.epochs,
        "best_test_acc": best_acc,
        "final_test_acc": history["test_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "params": param_count,
        "time_s": elapsed,
        "final_gamma_stats": final_gstats,
        "history": history,
    }
    if wandb_run is not None:
        wandb.summary["best_test_acc"] = best_acc
        wandb.summary["final_gamma"] = final_gstats
        wandb_run.finish()
    return result


# ═══════════════════════════════════════════════════════════════════
#  Main — sweep runner
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200,
                    help="Total epochs per run (default 200 matches CIFAR CNN recipe).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["nelu_pl", "nelu_pc", "nelu_sched"],
                    choices=list(_MODE_CLS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Override seed list. Default: 42/123/456 for pl/pc, 42/123 for sched.")
    # γ schedule controls (only used for --modes nelu_sched)
    ap.add_argument("--gamma-warmup-epochs", type=int, default=None,
                    help="Epochs to ramp γ from γ_start to γ_end, then HOLD. "
                         "Default: 5%% of --epochs (e.g. 10 for 200ep).")
    ap.add_argument("--gamma-warmup-frac", type=float, default=0.05,
                    help="Fraction of total epochs to use for γ warmup if "
                         "--gamma-warmup-epochs is not given. Default 0.05 (5%%). "
                         "Matches ConvNeXt/Swin LR-warmup convention (6.7%%) roughly. "
                         "For ImageNet later, sync this to each architecture's LR warmup.")
    ap.add_argument("--gamma-start", type=float, default=1e-4,
                    help="Starting γ for scheduled mode.")
    ap.add_argument("--gamma-end", type=float, default=1.0,
                    help="Target γ after warmup; held constant until end of training.")
    ap.add_argument("--gamma-curve", type=str, default="cosine",
                    choices=["cosine", "linear"],
                    help="Shape of the γ warmup segment.")
    ap.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "ablation_gamma_mode"))
    args = ap.parse_args()

    # Resolve γ warmup epoch count: absolute arg overrides, else fraction of epochs
    if args.gamma_warmup_epochs is None:
        args.gamma_warmup_epochs = max(1, int(round(args.epochs * args.gamma_warmup_frac)))
    print(f"[config] γ warmup epochs = {args.gamma_warmup_epochs} "
          f"({100 * args.gamma_warmup_epochs / args.epochs:.1f}% of {args.epochs}ep)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default seed schedule: 2/2/2/1/1 = 8 runs total
    default_seeds = {
        "nelu_pl":            [42, 123],
        "nelu_pc":            [42, 123],
        "nelu_sched":         [42, 123],
        "nelu_schedlearn_pl": [42],
        "nelu_schedlearn_pc": [42],
    }

    all_results = []
    t_start = time.time()
    total_runs = 0
    for mode in args.modes:
        seeds = args.seeds if args.seeds is not None else default_seeds[mode]
        for seed in seeds:
            run_tag = f"{mode}_s{seed}_e{args.epochs}"
            result_path = out_dir / f"{run_tag}.json"
            if result_path.exists():
                print(f"[SKIP] {run_tag} already exists")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue
            total_runs += 1
            print(f"\n{'='*60}\nRun {total_runs}: mode={mode}  seed={seed}\n{'='*60}")
            result = train_one_run(mode, seed, args)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

    total_elapsed = time.time() - t_start
    print(f"\n\n{'='*70}\nSUMMARY  (total {total_elapsed/60:.1f} min)\n{'='*70}")
    print(f"{'mode':>12s} {'seed':>6s} {'best':>8s} {'final':>8s} {'γ_mean':>10s}")
    print("-" * 60)
    by_mode = {}
    for r in all_results:
        print(f"{r['mode']:>12s} {r['seed']:>6d} "
              f"{r['best_test_acc']:>7.2f}% {r['final_test_acc']:>7.2f}% "
              f"{r.get('final_gamma_stats', {}).get('gamma_mean', float('nan')):>10.4f}")
        by_mode.setdefault(r["mode"], []).append(r["best_test_acc"])
    print("-" * 60)
    print(f"\n{'mode':>12s} {'n':>4s} {'mean±std':>18s}  {'min..max':>14s}")
    for mode, accs in by_mode.items():
        a = np.array(accs)
        print(f"{mode:>12s} {len(a):>4d} {a.mean():>8.2f} ± {a.std(ddof=0):.2f}"
              f"    {a.min():.2f}..{a.max():.2f}")

    summary = {
        "modes": list(by_mode.keys()),
        "per_mode_mean": {m: float(np.mean(by_mode[m])) for m in by_mode},
        "per_mode_std":  {m: float(np.std(by_mode[m], ddof=0)) for m in by_mode},
        "all_runs": [
            {"mode": r["mode"], "seed": r["seed"],
             "best": r["best_test_acc"], "final": r["final_test_acc"]}
            for r in all_results
        ],
        "total_minutes": total_elapsed / 60,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
