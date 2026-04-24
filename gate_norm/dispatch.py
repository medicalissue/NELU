"""Where to run the forward pass: PyTorch or the fused CUDA kernel.

`GateNorm.forward` consults :func:`should_use_cuda` on every call. The dispatch
rule is dead simple: if the tensor lives on CUDA **and** the kernel is
available for this dtype, take the fused path; otherwise stay in PyTorch. No
environment variables, no global toggles — the kernel is an implementation
detail, not a feature flag.

The one escape hatch is ``GATE_NORM_FORCE_PYTHON=1`` for debugging. It forces
the PyTorch path everywhere so numerical mismatches between the two backends
can be bisected without rebuilding.
"""

from __future__ import annotations

import os

import torch


_FORCE_PYTHON = os.environ.get("GATE_NORM_FORCE_PYTHON", "0") == "1"

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def should_use_cuda(z: torch.Tensor) -> bool:
    """True iff the fused CUDA kernel should handle ``z``."""
    if _FORCE_PYTHON:
        return False
    if not z.is_cuda:
        return False
    if z.dtype not in _SUPPORTED_DTYPES:
        return False
    return True
