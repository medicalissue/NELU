"""Runtime backend selection for Gate Normalization forward.

:meth:`gate_norm.GateNorm.forward` consults :func:`should_use_cuda` on
every call. The rule is intentionally narrow: the fused CUDA kernel
takes the call iff

* the input lives on CUDA,
* its dtype is one of (fp32, fp16, bf16),
* CUDA is available to PyTorch in this process, and
* the user has not set ``GATE_NORM_FORCE_PYTHON=1`` (debug escape hatch).

If any condition fails, the pure PyTorch path runs — guaranteeing
identical numerical semantics on CPU, MPS, and CUDA-less builds.

The CUDA extension itself is built lazily by :mod:`gate_norm.cuda` on
first call; we don't probe it here so that environments without ``nvcc``
(macOS, CPU-only Linux) never trigger a build attempt.
"""

from __future__ import annotations

import os

import torch


_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _force_python() -> bool:
    """Read the escape-hatch env var on every call so tests can flip it
    without reloading or re-importing the gate_norm package."""
    return os.environ.get("GATE_NORM_FORCE_PYTHON", "0") == "1"


def should_use_cuda(z: torch.Tensor) -> bool:
    """True iff the fused CUDA kernel should handle ``z``."""
    if _force_python():
        return False
    if not z.is_cuda:
        return False
    if not torch.cuda.is_available():
        return False
    if z.dtype not in _SUPPORTED_DTYPES:
        return False
    return True
