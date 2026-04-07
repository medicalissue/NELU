"""Fused CUDA kernel for NELU.

Two execution paths:
  1. C++ autograd (nelu_autograd): backward stays in C++, no Python dispatch
  2. Legacy autograd.Function: fallback if C++ registration fails

4D CNN inputs reshaped to (B, C*H*W) — contiguous, no copy.
"""

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_nelu_cuda = load(
    name="nelu_cuda",
    sources=[os.path.join(_dir, "csrc", "nelu_cuda.cu")],
    verbose=False,
)

# Check if C++ autograd path is available
_HAS_CPP_AUTOGRAD = hasattr(_nelu_cuda, "nelu_autograd")


def _to_2d(z):
    if z.dim() == 4:
        return z.reshape(z.size(0), -1), z.shape
    return z.reshape(-1, z.size(-1)), z.shape


# ── Path 1: C++ autograd (preferred — no Python backward dispatch) ──

def _nelu_cpp_autograd(z, eps=1e-6):
    z_2d, orig = _to_2d(z.contiguous())
    y = _nelu_cuda.nelu_autograd(z_2d, float(eps))
    return y.reshape(orig)


# ── Path 2: Legacy Python autograd.Function (fallback) ──

class _LegacyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, eps):
        z_2d, orig = _to_2d(z.contiguous())
        y, rms = _nelu_cuda.forward(z_2d, eps)
        ctx.save_for_backward(z_2d, rms)
        ctx.orig_shape = orig
        return y.reshape(orig)

    @staticmethod
    def backward(ctx, dy):
        z_2d, rms = ctx.saved_tensors
        dy_2d, _ = _to_2d(dy.contiguous())
        dz = _nelu_cuda.backward(z_2d, rms, dy_2d)
        return dz.reshape(ctx.orig_shape), None


# ── Public API ───────────────────────────────────────────────────

def nelu_cuda(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if _HAS_CPP_AUTOGRAD:
        return _nelu_cpp_autograd(z, eps)
    return _LegacyFn.apply(z, eps)


class NELUCUDA(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nelu_cuda(z, self.eps)

    def extra_repr(self) -> str:
        backend = "cpp_autograd" if _HAS_CPP_AUTOGRAD else "legacy"
        return f"eps={self.eps}, backend={backend}"
