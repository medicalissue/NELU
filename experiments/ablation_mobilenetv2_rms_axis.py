#!/usr/bin/env python3
"""RMS reduction-axis ablation on MobileNetV2 (CIFAR-100).

Why MobileNetV2 specifically: it's the only arch in the main grid that
uses depthwise convolutions, where each output channel is computed by
an independent 3×3 filter over a single input channel. Channels are
the most decoupled processing units, so the RMS reduction axis matters
most here.

Variants
--------
    nelu_hw          rms over (H,W)        — pure channel-wise
    nelu_c           rms over (C,)         — pure position-wise
    nelu_hybrid      DW→HW, PW→C           — channel-wise after DW conv,
                                              position-wise after PW conv
    nelu_hybrid_hwc  DW→HW, PW→HWC         — channel-wise after DW conv,
                                              full per-sample after PW conv

NELU_CHW (the default `nelu`, rms over the full (C,H,W)) is already
covered by Phase 1a's main grid as
    results/main_mobilenetv2_cifar100_nelu_s{42,123,456}.json
so we don't re-run it here.

Hybrid rationale:
  • depthwise conv processes each channel independently → per-channel
    rms (HW) keeps each channel's normalization local;
  • pointwise (1×1) conv mixes channels → either per-position (C) or
    per-sample (HWC) makes sense — both versions provided so reviewers
    can see the full design space.

Outputs go to results/rms_axis/ so they don't clobber Phase 1a:
    results/rms_axis/main_mobilenetv2_cifar100_<variant>_s<seed>.json

Direct usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/ablation_mobilenetv2_rms_axis.py \
        --variant nelu_hybrid --seed 42 --amp --compile --wandb
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


class NELU_HWC(nn.Module):
    """RMS reduction over all non-batch axes (H, W, C) — per-sample.
    Identical to the default NELU; provided as a named class for the
    hybrid variant so the swap-time print is unambiguous."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        if z.dim() == 4:
            dim = (1, 2, 3)
        else:
            dim = -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        return z * 0.5 * (1.0 + torch.erf(z / (rms * math.sqrt(2))))


class _NELUMarker(nn.Module):
    """Placeholder used for the hybrid variant — replaced post-hoc."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):  # never actually called
        raise RuntimeError("hybrid marker should be replaced before forward")


# ── Monkey-patch main_cifar_tinyimagenet ──────────────────────────

import main_cifar_tinyimagenet as mc

mc._ACT_MAP = dict(mc._ACT_MAP)
mc._ACT_MAP["nelu_hw"] = NELU_HW
mc._ACT_MAP["nelu_c"] = NELU_C
mc._ACT_MAP["nelu_hybrid"] = _NELUMarker      # DW→HW, PW→C
mc._ACT_MAP["nelu_hybrid_hwc"] = _NELUMarker  # DW→HW, PW→HWC (= default)

# Redirect outputs to results/rms_axis/ so they live alongside the
# Phase 1a CHW result without overwriting it.
axis_dir = mc.RESULTS_DIR / "rms_axis"
axis_dir.mkdir(parents=True, exist_ok=True)
mc.RESULTS_DIR = axis_dir

# Don't bother suppressing checkpoint dumps for this ablation —
# mobilenetv2 checkpoints are ~10 MB each so 8 jobs × 2 (last+best)
# = ~160 MB total. Trivial. Earlier no-op-save approach broke
# main_cifar's atomic-write pattern (torch.save tmp → pathlib.replace).


# ── Hybrid swap (context-aware) ───────────────────────────────────

def _hybrid_replace_after_build(model: nn.Module, pw_class, eps: float = 1e-6) -> nn.Module:
    """Walk every nn.Sequential in the model. For each NELUMarker
    placeholder, look backward through the sequence for the most recent
    Conv2d. If that conv is depthwise (groups > 1), replace the marker
    with NELU_HW; otherwise (pointwise / 1x1) replace with `pw_class`.
    Returns the same model in-place.
    """
    n_hw = 0
    n_pw = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        n = len(module)
        for i in range(n):
            child = module[i]
            if not isinstance(child, _NELUMarker):
                continue
            # Find most recent Conv2d preceding index i
            last_conv = None
            for j in range(i - 1, -1, -1):
                if isinstance(module[j], nn.Conv2d):
                    last_conv = module[j]
                    break
            if last_conv is None:
                # Standalone activation — fall back to per-channel HW
                module[i] = NELU_HW(eps=eps)
                n_hw += 1
                continue
            if last_conv.groups > 1:
                module[i] = NELU_HW(eps=eps)
                n_hw += 1
            else:
                module[i] = pw_class(eps=eps)
                n_pw += 1
    pw_name = pw_class.__name__
    print(f"  [hybrid] swapped: {n_hw} × NELU_HW (after DW), "
          f"{n_pw} × {pw_name} (after PW)")
    return model


# Wrap mc.replace_activations: when act_name is one of the hybrid
# variants, let the default replace put a _NELUMarker on every ReLU6
# (recursively), then walk the FULL model once to resolve markers based
# on context. A flag tracks the top-level call so the resolver runs
# only at the end of recursion, not after every nested Sequential.
_HYBRID_PW_CLASS = {
    "nelu_hybrid":     NELU_C,    # PW → per-position
    "nelu_hybrid_hwc": NELU_HWC,  # PW → per-sample (full)
}

_orig_replace = mc.replace_activations
_hybrid_in_progress = False

def _patched_replace_activations(model: nn.Module, act_name: str) -> nn.Module:
    global _hybrid_in_progress
    if act_name not in _HYBRID_PW_CLASS or _hybrid_in_progress:
        return _orig_replace(model, act_name)
    _hybrid_in_progress = True
    try:
        model = _orig_replace(model, act_name)
        _hybrid_replace_after_build(model, pw_class=_HYBRID_PW_CLASS[act_name])
    finally:
        _hybrid_in_progress = False
    return model

mc.replace_activations = _patched_replace_activations


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["nelu_hw", "nelu_c", "nelu_hybrid",
                                 "nelu_hybrid_hwc"])
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
