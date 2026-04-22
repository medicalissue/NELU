#!/usr/bin/env bash
# Setup script for SkyPilot workers. Idempotent; safe to re-run.
set -euxo pipefail

# SkyPilot AMIs ship PyTorch and CUDA drivers already.
python -m pip install --upgrade pip
python -m pip install -e '.[train]'

# Pre-build the fused CUDA extensions so the first training iteration is not
# held up by a JIT compile. Failures here are not fatal — the Python fallback
# path is always available.
python - <<'PY'
import torch
if torch.cuda.is_available():
    try:
        from gate_norm.cuda import _get_ext
        _get_ext("nelu")
        _get_ext("nilu")
        print("[setup] CUDA extensions compiled")
    except Exception as e:
        print(f"[setup] CUDA extension compile skipped: {e}")
else:
    print("[setup] No CUDA device detected; skipping extension prebuild.")
PY
