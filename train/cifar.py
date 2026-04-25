"""CIFAR-100 training with activation comparison.

Self-contained script for training CIFAR-scale models with NELU/NiLU
activation swaps. Supports standard CIFAR-100 benchmarking models and
logs gamma dynamics for gated activations.

Usage:
    python train/train_cifar.py --model resnet20 --activation nelu --seed 42
    python train/train_cifar.py --model wrn28_10 --activation gelu --seed 123
    python train/train_cifar.py --model densenet100 --activation nilu --epochs 300
"""

import argparse
import json
import math
import os
import random
import sys
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gate_norm import NELU, NiLU, collect_gamma_stats
from train.swap import (  # noqa: F401
    replace_activation,
    replace_activation_auto_axes,
)


# ---------------------------------------------------------------------------
#  Model factory
# ---------------------------------------------------------------------------
#
# Seven CIFAR-100 architectures loaded from public reference implementations:
#
#   * ResNet-20 / 56 / 110        — chenyaofo/pytorch-cifar-models hub entry
#   * VGG-16-BN                   — same hub
#   * MobileNetV2 x1.0            — same hub
#   * ShuffleNetV2 x1.0           — same hub
#   * DenseNet-BC-100-12          — bearpaw/pytorch-classification vendored
#                                   in ``third_party/bearpaw_cifar/``
#
# All architectures enter the activation swap through
# :func:`train.swap.replace_activation_auto_axes`, which picks per-site
# ``norm_axes`` from the adjacent Conv2d's kernel shape + groups. Pre- vs.
# post-activation ordering is a per-architecture property listed in
# :data:`_ACTIVATION_ORDER`.

_SUPPORTED_MODELS = (
    "resnet20", "resnet56", "resnet110",
    "vgg16_bn",
    "densenet_bc_100_12",
    "mobilenetv2", "shufflenetv2",
)


# Pre-activation architectures feed each ReLU into the *next* conv, not out of
# the previous one. DenseNet's Bottleneck is the only pre-activation model in
# our lineup; everything else uses post-activation (Conv → BN → ReLU).
_ACTIVATION_ORDER = {
    "resnet20":           "post",
    "resnet56":           "post",
    "resnet110":          "post",
    "vgg16_bn":           "post",
    "densenet_bc_100_12": "pre",
    "mobilenetv2":        "post",
    "shufflenetv2":       "post",
}


def _from_chenyaofo_hub(hub_name: str, num_classes: int) -> nn.Module:
    """Load a model from the chenyaofo/pytorch-cifar-models torch.hub repo."""
    import torch
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        hub_name,
        pretrained=False,
        num_classes=num_classes,
        trust_repo=True,
    )
    return model


def _build_backbone(name: str, num_classes: int) -> nn.Module:
    """Instantiate the raw ReLU-based backbone for ``name``."""
    if name == "resnet20":
        return _from_chenyaofo_hub("cifar100_resnet20", num_classes)
    if name == "resnet56":
        return _from_chenyaofo_hub("cifar100_resnet56", num_classes)
    if name == "resnet110":
        # chenyaofo stops at depth 56; depth 110 comes from the bearpaw
        # vendored copy, which is the classical He-2015 CIFAR ResNet at
        # arbitrary depth via ``(depth-2)/6`` basic blocks per stage.
        from third_party.bearpaw_cifar import resnet as _bp_resnet
        return _bp_resnet.resnet(
            depth=110, num_classes=num_classes, block_name="BasicBlock",
        )
    if name == "vgg16_bn":
        return _from_chenyaofo_hub("cifar100_vgg16_bn", num_classes)
    if name == "mobilenetv2":
        return _from_chenyaofo_hub("cifar100_mobilenetv2_x1_0", num_classes)
    if name == "shufflenetv2":
        return _from_chenyaofo_hub("cifar100_shufflenetv2_x1_0", num_classes)
    if name == "densenet_bc_100_12":
        from third_party.bearpaw_cifar import densenet as _bp_densenet
        return _bp_densenet.densenet(
            depth=100, growthRate=12, num_classes=num_classes, dropRate=0.0,
        )
    raise ValueError(
        f"Unknown model: {name!r}. Supported: {_SUPPORTED_MODELS}"
    )
def build_model(
    name: str,
    activation: str = "relu",
    num_classes: int = 100,
    *,
    gamma_init: float = 0.0,
    beta_init: float = 0.0,
):
    """Create a CIFAR-100 model and apply the requested activation swap.

    ``name`` must be one of :data:`_SUPPORTED_MODELS`. Five of the seven come
    from the chenyaofo/pytorch-cifar-models hub; ResNet-110 and
    DenseNet-BC-100-12 come from the bearpaw copy vendored in
    ``third_party/``.

    Activation handling:
      * ``relu``                 — leaves the model untouched.
      * ``gelu`` / ``silu``      — unconditional drop-in replacement.
      * ``nelu`` / ``nilu``      — swap via ``replace_activation_auto_axes``,
                                   which picks per-site ``norm_axes`` from
                                   the adjacent Conv2d's kernel shape.
    """
    if name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {name!r}. Supported: {_SUPPORTED_MODELS}"
        )

    model = _build_backbone(name, num_classes)

    if activation == "relu":
        return model

    # torchvision MobileNetV2 uses nn.ReLU6; custom CIFAR blocks use
    # plain nn.ReLU. Match both so every family swaps uniformly.
    relu_types: tuple[type, ...] = (nn.ReLU, nn.ReLU6)
    order = _ACTIVATION_ORDER[name]

    if activation in {"gelu", "silu"}:
        factory = (lambda: nn.GELU()) if activation == "gelu" else (lambda: nn.SiLU())
        n = replace_activation(model, relu_types, factory)
        print(f"Swapped {n} ReLU -> {activation}")
    elif activation == "nelu":
        n = replace_activation_auto_axes(
            model, relu_types, NELU, activation_order=order,
            gamma_init=gamma_init, beta_init=beta_init,
        )
        print(f"Swapped {n} ReLU -> NELU ({order}-activation axes)")
    elif activation == "nilu":
        n = replace_activation_auto_axes(
            model, relu_types, NiLU, activation_order=order,
            gamma_init=gamma_init, beta_init=beta_init,
        )
        print(f"Swapped {n} ReLU -> NiLU ({order}-activation axes)")
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return model



# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_dataloaders(data_dir, train_batch_size, val_batch_size=None, num_workers=4):
    if val_batch_size is None:
        val_batch_size = train_batch_size * 2

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    # Expect the CIFAR-100 archive already extracted to
    # ``<data_dir>/cifar-100-python/`` by scripts/prepare_data.sh and baked
    # into the worker's data snapshot. The snapshot mount is read-only, so
    # ``download=True`` would crash on the first worker trying to populate
    # it. Fail loud and early instead.
    train_ds = datasets.CIFAR100(data_dir, train=True, download=False,
                                  transform=train_transform)
    test_ds = datasets.CIFAR100(data_dir, train=False, download=False,
                                 transform=test_transform)
    # persistent_workers keeps the dataloader subprocesses alive across
    # epoch boundaries. CIFAR-100 has ~200 epoch boundaries per run and
    # spinning workers up takes ~0.5-1s each, so this saves ~3 minutes
    # of wall-clock per run with no behavioral change.
    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

class TrainingInterrupted(Exception):
    """Raised when SIGTERM/SIGINT requests a graceful stop."""
    pass


_INTERRUPTION_REQUESTED = False


def _request_interruption(signum, _frame):
    global _INTERRUPTION_REQUESTED
    _INTERRUPTION_REQUESTED = True
    print(f"\n[signal] Received signal {signum}; will stop after the current step.", flush=True)


def interruption_requested():
    return _INTERRUPTION_REQUESTED


def save_checkpoint(state, output_dir, interrupted=False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    if interrupted:
        (output_path / "INTERRUPTED").touch()


def _orig_module(model):
    """Return the underlying module, stripping torch.compile's wrapper."""
    return getattr(model, "_orig_mod", model)


def build_checkpoint_state(model, optimizer, scheduler, best_acc, args, wandb_id,
                           training_log, epoch):
    return {
        "epoch": epoch,
        "model": _orig_module(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc": best_acc,
        "args": vars(args),
        "training_log": training_log,
        "wandb_id": wandb_id,
    }

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    scaler=None, use_amp=False, amp_dtype=torch.float16):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested at epoch {epoch}, step {batch_idx}."
            )
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested at epoch {epoch}, step {batch_idx}."
            )
    scheduler.step()
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if interruption_requested():
            raise TrainingInterrupted(
                f"Interruption requested during evaluation at step {batch_idx}."
            )
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def init_wandb_run(args, saved_wandb_id=None):
    if not args.wandb:
        return None, saved_wandb_id

    if args.resume and os.path.isfile(args.resume) and not saved_wandb_id:
        raise RuntimeError(
            f"Resume checkpoint {args.resume} is missing wandb_id; refusing to create a new wandb run."
        )

    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed, disabling wandb logging")
        return None, saved_wandb_id

    init_kwargs = {
        "project": args.wandb_project,
        "name": f"{args.model}_{args.activation}_s{args.seed}",
        "config": vars(args),
    }
    if saved_wandb_id:
        # resume="allow" so a deleted/expired remote run falls back to creating
        # a fresh run with a new id, instead of crashing the trainer (which
        # then loses the in-progress checkpoint to spot-preempt corruption).
        init_kwargs["id"] = saved_wandb_id
        init_kwargs["resume"] = "allow"
    else:
        init_kwargs["id"] = wandb.util.generate_id()
        init_kwargs["resume"] = "never"

    wandb_run = wandb.init(**init_kwargs)
    if saved_wandb_id and wandb_run.id != saved_wandb_id:
        # Server returned a different id (e.g. the saved one was deleted).
        # Continue with the new id so the next checkpoint write picks it up.
        print(
            f"WARNING: wandb returned new id {wandb_run.id} (saved {saved_wandb_id} "
            f"likely deleted); continuing with new run."
        )

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    return wandb_run, wandb_run.id


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def _load_config_with_includes(path: str) -> dict:
    """Load a YAML config, resolving a single ``include: <path>`` directive.

    ``include`` is interpreted relative to the current config's directory.
    Fields declared in the outer file override fields inherited through
    ``include`` — exactly the mental model of YAML-with-defaults configs.
    No recursion limit beyond a single base file is needed: our layout is
    ``_base.yaml`` + model-specific leaf, never deeper.
    """
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    inc = cfg.pop("include", None)
    if inc is None:
        return cfg
    base_path = os.path.join(os.path.dirname(os.path.abspath(path)), inc)
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    base.update(cfg)
    return base


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-100 training with activation comparison")
    p.add_argument("--model", type=str, default="resnet20",
                    choices=list(_SUPPORTED_MODELS))
    p.add_argument("--activation", type=str, default="relu",
                    choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=None,
                   help="Validation batch size; defaults to 2x train batch size")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--nesterov", action="store_true", default=True,
                   help="Use Nesterov momentum (chenyaofo default; True)")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd"])
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["multistep", "cosine"])
    p.add_argument("--milestones", type=int, nargs="+", default=[60, 120, 160],
                   help="MultiStepLR milestones (epoch indices)")
    p.add_argument("--lr_gamma", type=float, default=0.2,
                   help="MultiStepLR decay factor")
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Linear warmup epochs from start_factor*lr to lr (0 disables)")
    p.add_argument("--warmup_start_factor", type=float, default=1e-3,
                   help="Initial multiplier on lr when warmup is enabled")
    p.add_argument("--min_lr", type=float, default=0.0,
                   help="Cosine schedule floor (ignored for multistep)")
    p.add_argument("--gamma_init", type=float, default=0.0,
                   help="Initial value of the learnable γ scalar in NELU/NiLU.")
    p.add_argument("--beta_init", type=float, default=0.0,
                   help="Initial value of the learnable β scalar in NELU/NiLU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="results/cifar")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--data_dir", type=str, default="/data")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Use AMP (mixed precision)")
    p.add_argument("--amp_dtype", type=str, default="float16",
                   choices=["float16", "bfloat16"],
                   help="AMP dtype when --amp is set")
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--compile_mode", type=str, default=None,
                   choices=[None, "default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"],
                   help="torch.compile mode (see PyTorch docs)")
    p.add_argument("--compile_backend", type=str, default="inductor",
                   help="torch.compile backend (default: inductor)")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="nelu-cifar")
    p.add_argument("--config", type=str, default=None, help="YAML config (overrides defaults)")
    return p.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _request_interruption)
    signal.signal(signal.SIGINT, _request_interruption)

    # Load YAML config if provided. Configs declare ``include: <path>`` to
    # inherit from a base config; model-specific stubs use this to pull
    # the unified recipe from ``configs/cifar100/_base.yaml`` without
    # repeating every field.
    #
    # Precedence: CLI > YAML > argparse defaults. We detect which flags
    # the user passed explicitly by scanning ``sys.argv`` so YAML entries
    # only fill in defaults, never overwrite explicit CLI values.
    if args.config and os.path.exists(args.config):
        cfg = _load_config_with_includes(args.config)
        cli_tokens = set(sys.argv[1:])
        for k, v in cfg.items():
            if not hasattr(args, k) or k == "config":
                continue
            # Match on the argparse flag form for this attribute.
            cli_flags = {f"--{k}", f"--{k.replace('_', '-')}"}
            if cli_flags & cli_tokens:
                continue
            setattr(args, k, v)

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb init (deferred until after resume, so we can reuse the saved run ID)
    wandb_run = None
    saved_wandb_id = None

    # Data
    train_loader, test_loader = get_dataloaders(
        args.data_dir,
        args.batch_size,
        args.val_batch_size,
        args.num_workers,
    )
    # Model
    model = build_model(
        args.model, args.activation, num_classes=100,
        gamma_init=args.gamma_init, beta_init=args.beta_init,
    )
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, activation: {args.activation}, "
          f"params: {param_count:,}, device: {device}")

    # torch.compile — wrap the model in-place. State dicts are saved from
    # the underlying ``._orig_mod`` when compiled so checkpoints remain
    # portable between --compile and eager runs.
    if args.compile:
        assert hasattr(torch, "compile"), "torch.compile requires PyTorch >= 2.0"
        model = torch.compile(model, backend=args.compile_backend,
                              mode=args.compile_mode)
        print(f"Model compiled with torch.compile "
              f"(backend={args.compile_backend}, mode={args.compile_mode})")

    # AMP scaler — only needed for fp16; bf16 has the fp32 dynamic range.
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = (torch.amp.GradScaler("cuda")
              if args.amp and amp_dtype == torch.float16 else None)

    # Optimizer + scheduler. Warmup is optional — the unified CIFAR-100
    # recipe skips it (warmup_epochs=0), but the plumbing still supports
    # LinearLR → main-schedule chains for experimentation.
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=args.nesterov,
    )

    warmup_epochs = max(0, int(args.warmup_epochs))
    if args.scheduler == "multistep":
        main_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(args.milestones), gamma=args.lr_gamma)
    elif args.scheduler == "cosine":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_epochs),
            eta_min=args.min_lr)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=args.warmup_start_factor,
            total_iters=warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    # Resume
    start_epoch = 0
    best_acc = 0.0
    training_log = []

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _orig_module(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        training_log = ckpt.get("training_log", [])
        saved_wandb_id = ckpt.get("wandb_id", None)
        print(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # wandb init — after resume so we can reuse the saved run ID
    wandb_run, wandb_id = init_wandb_run(args, saved_wandb_id)

    # Probe batch for gate entropy (fixed across training)
    probe_batch = next(iter(test_loader))[0][:32].to(device)

    # Training
    interrupted = False
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                     scheduler, device, epoch,
                                                     scaler=scaler, use_amp=args.amp,
                                                     amp_dtype=amp_dtype)
            test_loss, test_acc = evaluate(model, test_loader, device,
                                           use_amp=args.amp, amp_dtype=amp_dtype)
        except TrainingInterrupted as e:
            interrupted = True
            print(f"\n{'='*60}")
            print(f"  TRAINING INTERRUPTED: {e}")
            print("  Saving checkpoint for automatic spot resume.")
            print(f"{'='*60}")
            interrupted_state = build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_acc=best_acc,
                args=args,
                wandb_id=wandb_id,
                training_log=training_log,
                epoch=epoch - 1,
            )
            interrupted_state["interrupted_at_epoch"] = epoch
            save_checkpoint(interrupted_state, args.output_dir, interrupted=True)
            if wandb_run:
                import wandb
                wandb.log({"epoch": epoch, "interrupted": True, "interrupted_epoch": epoch})
                wandb.finish()
            break
        elapsed = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        from train.diagnostics import gate_stats as measure_gate_stats, weight_norms as log_weight_norms

        # Gamma stats (nelu/nilu only)
        gamma_stats = {}
        if args.activation in ("nelu", "nilu"):
            gamma_stats = collect_gamma_stats(model)

        # Gate entropy + variance (all activations)
        gate_stats = measure_gate_stats(model, probe_batch, device)

        # Weight norms (all activations)
        weight_norm_stats = log_weight_norms(model)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        }
        log_entry.update(gamma_stats)
        log_entry.update(gate_stats)
        log_entry.update(weight_norm_stats)
        training_log.append(log_entry)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {train_acc:.2f}% | test {test_acc:.2f}% | "
              f"best {best_acc:.2f}% | lr {optimizer.param_groups[0]['lr']:.6f} | "
              f"{elapsed:.1f}s", end="")
        if gamma_stats:
            print(f" | gamma_mean {gamma_stats.get('nelu/gamma/mean', 0):.4f}", end="")
        if gate_stats:
            print(f" | H {gate_stats.get('gate_entropy/mean', 0):.3f}"
                  f" S {gate_stats.get('gate_var/mean', 0):.4f}", end="")
        print()

        if wandb_run:
            import wandb
            wandb_metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/top1": train_acc,
                "time/epoch_seconds": elapsed,
                "test/loss": test_loss,
                "test/top1": test_acc,
                "best/test_top1": best_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            wandb_metrics.update(gamma_stats)
            wandb_metrics.update(gate_stats)
            wandb_metrics.update(weight_norm_stats)
            wandb.log(wandb_metrics)

        # Save checkpoints (includes wandb_id for run continuity across spot interruptions)
        state = build_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            args=args,
            wandb_id=wandb_id,
            training_log=training_log,
            epoch=epoch,
        )
        save_checkpoint(state, args.output_dir)
        if is_best:
            torch.save(state, os.path.join(args.output_dir, "checkpoint-best.pt"))

    if interrupted:
        print("\nTraining interrupted cleanly. Resume checkpoint saved.")
        return

    # Save final result.json
    result = {
        "model": args.model,
        "activation": args.activation,
        "seed": args.seed,
        "final_test_acc": training_log[-1]["test_acc"] if training_log else 0,
        "best_test_acc": best_acc,
        "epochs": args.epochs,
        "params": param_count,
        "training_log": training_log,
    }
    result_path = os.path.join(args.output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_path}")
    print(f"Best test accuracy: {best_acc:.2f}%")

    # Sentinel for the S3-backed job queue: presence of ``complete`` tells
    # the orchestrator this experiment is done and should not be re-run.
    # Must be written only after every other artifact is on disk so a worker
    # crashing mid-finalize does not falsely mark the job finished.
    Path(args.output_dir, "complete").write_text(
        f"{args.model}-{args.activation}-s{args.seed} @ {best_acc:.4f}\n"
    )

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
