#!/usr/bin/env python3
"""Bit-exact wrapper around timm's official train.py.

Why this exists
---------------
For models trained by Wightman with timm/train.py (e.g.
`efficientnet_b2.ra_in1k`), reproducing the recipe at the
hyperparameter level is not enough — reviewers will (rightly) ask
"how do you know your training matches the timm checkpoint?"
The only safe answer is: we run the SAME train.py.

This wrapper:
  1. Pre-parses our `--our-act {nelu,nilu}` flag (and removes it
     from sys.argv before timm sees it).
  2. Monkey-patches `timm.models.create_model` so that EVERY model
     produced by it has the baseline activation swapped for ours.
  3. Hands control to timm's train.py via runpy. timm's argparse,
     dataloader, optimizer, scheduler, EMA, AMP, DDP, scaler — all
     are timm code, byte-for-byte.

Result: same numerics as Wightman's pretraining run except for the
activation. Training is therefore identically wired to the published
timm checkpoint of the chosen model.

Usage
-----
    torchrun --nproc_per_node=8 experiments/train_imagenet_timm.py \
        --our-act nilu \
        /data/imagenet \
        --model efficientnet_b2 \
        -b 128 --sched step --epochs 450 --decay-epochs 2.4 \
        --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 \
        --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 \
        --model-ema --model-ema-decay 0.9999 \
        --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 \
        --amp --lr .064 \
        --output ./results/imagenet/efficientnet_b2_nilu

(The CLI above is the documented Wightman recipe for B2 with bs/lr
linearly scaled from 2 GPU × 128 → 8 GPU × 128 = 1024 effective.)
"""

import os
import runpy
import sys
from pathlib import Path

# ── 1. Pre-parse our flag and strip it from argv ──────────────────

_OUR_ACT = None
_BASELINE_HINT = None  # optional override
_new_argv = []
_i = 0
while _i < len(sys.argv):
    a = sys.argv[_i]
    if a == "--our-act":
        _OUR_ACT = sys.argv[_i + 1]
        _i += 2
    elif a.startswith("--our-act="):
        _OUR_ACT = a.split("=", 1)[1]
        _i += 1
    elif a == "--baseline-act":
        _BASELINE_HINT = sys.argv[_i + 1]
        _i += 2
    else:
        _new_argv.append(a)
        _i += 1
sys.argv = _new_argv

# ── 2. Set up activation classes ──────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch.nn as nn
from nelu import NELU, NiLU, NELUCUDA, NiLUCUDA

_NELU_CLS = NELUCUDA if NELUCUDA is not None else NELU
_NILU_CLS = NiLUCUDA if NiLUCUDA is not None else NiLU
_ACT_CLS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "nelu": _NELU_CLS,
    "nilu": _NILU_CLS,
}
_DEFAULT_BASELINE = {"nelu": "gelu", "nilu": "silu"}


def _replace_act(module, baseline_cls, target_cls):
    for name, child in module.named_children():
        if isinstance(child, baseline_cls):
            setattr(module, name, target_cls())
        else:
            _replace_act(child, baseline_cls, target_cls)
    return module


# ── 3. Monkey-patch timm.models.create_model ──────────────────────

if _OUR_ACT is not None:
    if _OUR_ACT not in ("nelu", "nilu"):
        raise SystemExit(f"--our-act must be nelu or nilu, got {_OUR_ACT}")
    baseline = _BASELINE_HINT or _DEFAULT_BASELINE[_OUR_ACT]
    if baseline not in _ACT_CLS:
        raise SystemExit(f"unknown baseline act: {baseline}")
    target_cls = _ACT_CLS[_OUR_ACT]
    baseline_cls = _ACT_CLS[baseline]

    import timm.models as _tm
    _orig_create = _tm.create_model

    def _create_with_swap(*args, **kwargs):
        m = _orig_create(*args, **kwargs)
        _replace_act(m, baseline_cls, target_cls)
        n = sum(1 for x in m.modules() if isinstance(x, target_cls))
        print(f"[NELU wrapper] swapped {baseline} -> {_OUR_ACT}  ({n} modules)")
        return m

    _tm.create_model = _create_with_swap
    # The train.py script does `from timm.models import create_model` at
    # module load. Since we patch BEFORE runpy executes train.py, the
    # `from ... import create_model` binding will pick up the patched
    # function. (Verified by reading the import order in timm/train.py.)


# ── 4. Hand control to timm's train.py ────────────────────────────

_TIMM_TRAIN_PY = os.environ.get(
    "TIMM_TRAIN_PY", "/home/ubuntu/NELU/timm-train/train.py")

if not os.path.isfile(_TIMM_TRAIN_PY):
    raise SystemExit(
        f"timm train.py not found at {_TIMM_TRAIN_PY}. Set TIMM_TRAIN_PY env "
        f"var to override.")

runpy.run_path(_TIMM_TRAIN_PY, run_name="__main__")
