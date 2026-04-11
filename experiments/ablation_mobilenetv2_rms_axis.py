#!/usr/bin/env python3
"""RMS reduction-axis ablation on MobileNetV2 (CIFAR-100, seed 42).

Why MobileNetV2 specifically: it's the only arch in the main grid
that uses depthwise convolutions, where each output channel is
computed by an independent 3×3 filter over a single input channel.
Channels are the most decoupled processing units in the network,
so the RMS reduction axis matters most here:

    NELU_CHW : rms over (C,H,W) — couples every position+channel
    NELU_HW  : rms over (H,W)   — channel-wise normalization
    NELU_C   : rms over (C,)    — position-wise normalization

NELU_CHW (the default `nelu`) is already covered by Phase 1a:
    results/main_mobilenetv2_cifar100_nelu_s42.json
This script only re-runs the HW and C variants. Single seed 42 —
the goal is a "does the principle still hold under DW conv?" check,
not a 3-seed mean.

Outputs go to results/rms_axis/ so they don't clobber Phase 1a:
    results/rms_axis/main_mobilenetv2_cifar100_nelu_hw_s42.json
    results/rms_axis/main_mobilenetv2_cifar100_nelu_c_s42.json

Direct usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ablation_mobilenetv2_rms_axis.py \
        --variant nelu_hw --seed 42
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── NELU axis variants ────────────────────────────────────────────

class NELU_HW(nn.Module):
    """RMS reduction over spatial axes (H, W) only — channel-wise."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        if z.dim() == 4:
            dim = (2, 3)
        else:
            dim = -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms * math.sqrt(2))))


class NELU_C(nn.Module):
    """RMS reduction over channel axis only — per-position."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        if z.dim() == 4:
            dim = (1,)
        else:
            dim = -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms * math.sqrt(2))))


# ── Monkey-patch main_cifar_tinyimagenet ──────────────────────────

import main_cifar_tinyimagenet as mc

mc._ACT_MAP = dict(mc._ACT_MAP)
mc._ACT_MAP["nelu_hw"] = NELU_HW
mc._ACT_MAP["nelu_c"] = NELU_C

# Redirect outputs to results/rms_axis/ so they live alongside the
# Phase 1a CHW result without overwriting it.
axis_dir = mc.RESULTS_DIR / "rms_axis"
axis_dir.mkdir(parents=True, exist_ok=True)
mc.RESULTS_DIR = axis_dir

# Suppress checkpoint dumps — we only need best_acc in result.json.
_orig_save = torch.save
def _noop_save(obj, path, *args, **kwargs):
    p = str(path)
    if "/checkpoints/" in p or p.endswith((".pt", ".pt.tmp")):
        return
    return _orig_save(obj, path, *args, **kwargs)
torch.save = _noop_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["nelu_hw", "nelu_c"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()

    mc_args = argparse.Namespace(
        arch="mobilenetv2",
        dataset="cifar100",
        act=args.variant,
        seed=args.seed,
        lr=None,
        label_noise=0.0,
        wandb=args.wandb,
        compile=args.compile,
        amp=args.amp,
        all=False,
        no_resume=False,
        epochs=args.epochs,
    )

    print("=" * 60)
    print(f"  RMS-axis ablation on MobileNetV2 (DW conv)")
    print(f"  variant={args.variant}  seed={args.seed}  epochs={args.epochs}")
    print(f"  output: {axis_dir}")
    print("=" * 60)

    mc.run_experiment("mobilenetv2", "cifar100", args.variant, mc_args)


if __name__ == "__main__":
    main()
