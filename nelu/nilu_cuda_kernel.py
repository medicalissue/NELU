"""Fused CUDA kernel for NiLU.

Mirrors nelu/cuda_kernel.py — same structure, different math (sigmoid).

Two execution paths:
  1. C++ autograd (nilu_autograd): backward stays in C++
  2. Legacy autograd.Function: fallback if C++ registration fails

4D CNN inputs are reshaped to (B, C*H*W) — contiguous, no copy —
so the kernel reduces over all non-batch dims, matching nelu.NiLU.
"""

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_nilu_cuda = load(
    name="nilu_cuda",
    sources=[os.path.join(_dir, "csrc", "nilu_cuda.cu")],
    verbose=False,
)

_HAS_CPP_AUTOGRAD = hasattr(_nilu_cuda, "nilu_autograd")


def _to_2d(z):
    if z.dim() == 4:
        return z.reshape(z.size(0), -1), z.shape
    return z.reshape(-1, z.size(-1)), z.shape


# ── Path 1: C++ autograd (preferred) ───────────────────────────

def _nilu_cpp_autograd(z, eps=1e-6):
    z_2d, orig = _to_2d(z.contiguous())
    y = _nilu_cuda.nilu_autograd(z_2d, float(eps))
    return y.reshape(orig)


# ── Path 2: Python fallback ────────────────────────────────────

class _LegacyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, eps):
        z_2d, orig = _to_2d(z.contiguous())
        y, rms = _nilu_cuda.forward(z_2d, eps)
        ctx.save_for_backward(z_2d, rms)
        ctx.orig_shape = orig
        return y.reshape(orig)

    @staticmethod
    def backward(ctx, dy):
        z_2d, rms = ctx.saved_tensors
        dy_2d, _ = _to_2d(dy.contiguous())
        dz = _nilu_cuda.backward(z_2d, rms, dy_2d)
        return dz.reshape(ctx.orig_shape), None


# ── Public API ──────────────────────────────────────────────────

def nilu_cuda(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if _HAS_CPP_AUTOGRAD:
        return _nilu_cpp_autograd(z, eps)
    return _LegacyFn.apply(z, eps)


# Tell torch.compile / Dynamo to treat the pybind11 entrypoint as opaque.
# Without this, Dynamo emits a graph break around every NiLU call.
try:
    torch.compiler.allow_in_graph(_nilu_cuda.nilu_autograd)
except Exception:
    pass


class NiLUCUDA(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nilu_cuda(z, self.eps)

    def extra_repr(self) -> str:
        backend = "cpp_autograd" if _HAS_CPP_AUTOGRAD else "legacy"
        return f"eps={self.eps}, backend={backend}"
