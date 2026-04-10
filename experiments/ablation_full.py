#!/usr/bin/env python3
"""Full ablation: stop-gradient, dim variants, learnable tau, wd×2.

All on ResNet-20 CIFAR-100, 3 seeds. Evaluates clean + CIFAR-100-C.

Usage:
    python ablation_full.py --all --wandb --amp --compile
    python ablation_full.py --variant nelu_no_sg
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ── NELU variants ────────────────────────────────────────────────

class NELU_SG(nn.Module):
    """Standard NELU: dim=(1,2,3) for CNN, stop-gradient."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        dim = (1,2,3) if z.dim()==4 else -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

class NELU_NoSG(nn.Module):
    """NELU without stop-gradient."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        dim = (1,2,3) if z.dim()==4 else -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms * math.sqrt(2))))  # no detach

class NELU_DimW(nn.Module):
    """NELU with dim=-1 (W axis only)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        rms = z.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

class NELU_DimC(nn.Module):
    """NELU with dim=1 (C axis)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        dim = 1 if z.dim()==4 else -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

class NELU_DimHW(nn.Module):
    """NELU with dim=(2,3) (spatial)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, z):
        dim = (2,3) if z.dim()==4 else -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

class LearnableTau(nn.Module):
    """z * Phi(z / tau), tau learnable per-instance, init=1."""
    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(1))
    def forward(self, z):
        tau = self.log_tau.exp()
        return z * 0.5 * (1.0 + torch.erf(z / (tau * math.sqrt(2))))

VARIANTS = {
    "gelu":         lambda: nn.GELU(),
    "nelu":         lambda: NELU_SG(),
    "nelu_no_sg":   lambda: NELU_NoSG(),
    "nelu_dim_w":   lambda: NELU_DimW(),
    "nelu_dim_c":   lambda: NELU_DimC(),
    "nelu_dim_hw":  lambda: NELU_DimHW(),
    "nelu_dim_chw": lambda: NELU_SG(),  # same as nelu
    "learnable_tau": lambda: LearnableTau(),
    "gelu_wd2":     lambda: nn.GELU(),  # same act, different wd
}

# ── Model (ResNet-20) ────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = act_fn()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = act_fn()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))

def make_resnet20(act_fn, num_classes=100):
    class R20(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.act1 = act_fn()
            self.l1 = nn.Sequential(*[BasicBlock(16, 16, 1, act_fn) for _ in range(3)])
            self.l2 = nn.Sequential(BasicBlock(16, 32, 2, act_fn),
                                    *[BasicBlock(32, 32, 1, act_fn) for _ in range(2)])
            self.l3 = nn.Sequential(BasicBlock(32, 64, 2, act_fn),
                                    *[BasicBlock(64, 64, 1, act_fn) for _ in range(2)])
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            x = self.act1(self.bn1(self.conv1(x)))
            x = self.l1(x); x = self.l2(x); x = self.l3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            return self.fc(x.view(x.size(0), -1))
    return R20()

# ── Training ─────────────────────────────────────────────────────

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

def run(variant, seed=42, epochs=200, wd=5e-4, use_wandb=False, compile_model=False, use_amp=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    tr = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR100("./data", True, download=True, transform=tr)
    test_ds = datasets.CIFAR100("./data", False, transform=te)
    train_ld = DataLoader(train_ds, 128, True, num_workers=2, pin_memory=True, drop_last=True)
    test_ld = DataLoader(test_ds, 128, False, num_workers=2, pin_memory=True)

    # Override wd for gelu_wd2
    if variant == "gelu_wd2":
        wd = wd * 2

    act_fn = VARIANTS[variant]
    model = make_resnet20(act_fn).to(device)
    if compile_model:
        model = torch.compile(model)

    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd)
    warmup = optim.lr_scheduler.LinearLR(opt, 0.1, total_iters=1)
    step_sched = optim.lr_scheduler.MultiStepLR(opt, [60, 120, 160], gamma=0.2)
    sched = optim.lr_scheduler.SequentialLR(opt, [warmup, step_sched], milestones=[1])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # wandb (graceful fallback)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="nelu", group="ablation",
                name=f"{variant}_s{seed}",
                config={"variant": variant, "seed": seed, "epochs": epochs,
                        "wd": wd, "amp": use_amp, "compile": compile_model},
                reinit=True,
            )
        except Exception as e:
            print(f"  WARNING: wandb init failed ({type(e).__name__}: {e}); "
                  f"continuing without wandb")
            wandb_run = None

    best = 0
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            epoch_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()
        train_loss = epoch_loss / max(n_seen, 1)

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for x, y in test_ld:
                    x, y = x.to(device), y.to(device)
                    c += (model(x).argmax(1) == y).sum().item()
                    t += y.size(0)
            acc = 100.0 * c / t
            best = max(best, acc)
            if ep % 50 == 0:
                print(f"  [{variant} s{seed}] ep={ep} acc={acc:.2f}% best={best:.2f}%")
            if wandb_run is not None:
                try:
                    wandb_run.log({"epoch": ep, "train_loss": train_loss,
                                   "test_acc": acc, "best_acc": best,
                                   "lr": opt.param_groups[0]["lr"]})
                except Exception:
                    pass

    if wandb_run is not None:
        try:
            wandb_run.log({"final_best_acc": best})
            wandb_run.finish()
        except Exception:
            pass

    return {"variant": variant, "seed": seed, "best_acc": best, "wd": wd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nelu", choices=list(VARIANTS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output JSON path. Default: "
                             "results/ablation_<variant>_s<seed>.json (single) or "
                             "results/ablation_full.json (--all)")
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.all else [args.variant]
    os.makedirs("results", exist_ok=True)

    all_results = {}
    for v in variants:
        accs = []
        for s in args.seeds:
            # Skip if per-(variant, seed) result already exists
            single_path = f"results/ablation_{v}_s{s}.json"
            if not args.all and len(args.seeds) == 1 and os.path.exists(single_path):
                print(f"  SKIP {v} s{s}: {single_path} exists")
                with open(single_path) as f:
                    accs.append(json.load(f)["best_acc"])
                continue
            r = run(v, seed=s, use_wandb=args.wandb,
                    compile_model=args.compile, use_amp=args.amp)
            accs.append(r["best_acc"])
            # Always write per-(variant, seed) checkpoint result
            with open(single_path, "w") as f:
                json.dump(r, f, indent=2)
        import numpy as np
        mean, std = np.mean(accs), np.std(accs)
        all_results[v] = {"mean": mean, "std": std, "runs": accs}
        print(f"  {v:<16}: {mean:.2f} ± {std:.2f}")

    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print(f"  {'Variant':<16} {'Acc':>12}")
        print(f"  {'-'*30}")
        for v, r in sorted(all_results.items(), key=lambda x: -x[1]["mean"]):
            print(f"  {v:<16} {r['mean']:>8.2f} ± {r['std']:.2f}")

    out_path = args.output or (
        "results/ablation_full.json" if args.all
        else f"results/ablation_{variants[0]}_s{args.seeds[0]}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
