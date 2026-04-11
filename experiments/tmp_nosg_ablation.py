#!/usr/bin/env python3
"""Temporary ablation: NELU without stop-gradient on WRN-28-10 + ResNet-110.

Monkey-patches main_cifar_tinyimagenet._ACT_MAP so --act nelu uses
NELU_NoSG, and redirects RESULTS_DIR to results/nosg/ so the outputs
don't clobber the Phase 1a NELU (with sg) results.

Single GPU per run. Use the companion launcher scripts/run_tmp_nosg.sh
to dispatch 6 runs (2 archs × 3 seeds) across 6 GPUs in parallel.

Direct usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/tmp_nosg_ablation.py \
        --arch wrn28_10 --seed 42
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── NELU without stop-gradient (target of this ablation) ──────
class NELU_NoSG(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        dim = (1, 2, 3) if z.dim() == 4 else -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        # Note: NO .detach() on rms — this is the ablation point.
        return z * 0.5 * (1.0 + torch.erf(z / (rms * math.sqrt(2))))


# ── Monkey-patch main_cifar_tinyimagenet ──────────────────────
import main_cifar_tinyimagenet as mc

mc._ACT_MAP = dict(mc._ACT_MAP)
mc._ACT_MAP["nelu"] = NELU_NoSG

# Redirect outputs to results/nosg/ so we don't overwrite Phase 1a.
nosg_dir = mc.RESULTS_DIR / "nosg"
nosg_dir.mkdir(parents=True, exist_ok=True)
mc.RESULTS_DIR = nosg_dir

# Disable all checkpoint saves (we only need best_acc in result.json).
# WRN-28-10 checkpoints are ~140MB each × 18 files = 1.3 GB — not worth.
_orig_save = torch.save
def _noop_save(obj, path, *args, **kwargs):
    p = str(path)
    if "/checkpoints/" in p or p.endswith((".pt", ".pt.tmp")):
        return  # skip all model/optimizer dumps
    return _orig_save(obj, path, *args, **kwargs)
torch.save = _noop_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True,
                        choices=["wrn28_10", "resnet110", "resnet56",
                                 "densenet100", "mobilenetv2", "shufflenetv1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()

    mc_args = argparse.Namespace(
        arch=args.arch,
        dataset="cifar100",
        act="nelu",           # → NELU_NoSG via monkey-patch
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
    print(f"  NELU_NoSG ablation")
    print(f"  arch={args.arch}  seed={args.seed}  epochs={args.epochs}")
    print(f"  output: {nosg_dir}")
    print("=" * 60)

    mc.run_experiment(args.arch, "cifar100", "nelu", mc_args)


if __name__ == "__main__":
    main()
