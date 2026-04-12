"""Fused CUDA kernel for NELU — Surrogate backward.

Forward: identical NELU y = z · Φ(z/ρ)  (scale-invariant).
Backward: h(z) instead of h(z/ρ) — GELU-like damping via saturation.

Uses the same C++ extension as cuda_kernel.py:
  forward  → _nelu_cuda.forward      (shared, fast CUDA)
  backward → _nelu_cuda.backward_surr (new, element-wise, no reduction)
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_nelu_cuda = load(
    name="nelu_cuda",
    sources=[os.path.join(_dir, "csrc", "nelu_cuda.cu")],
    verbose=False,
)


def _to_2d(z):
    if z.dim() == 4:
        return z.reshape(z.size(0), -1)
    return z.reshape(-1, z.size(-1))


def _rho_size(z):
    if z.dim() == 4:
        return z.size(0)
    M = 1
    for d in z.shape[:-1]:
        M *= d
    return M


@torch.library.custom_op("nelu_surr::fwd", mutates_args=(), device_types="cuda")
def _fwd_op(z: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    z_2d = _to_2d(z)
    y, rho = _nelu_cuda.forward(z_2d, eps)
    return y.reshape(z.shape), rho  # rho returned for interface compat, unused in bwd


@_fwd_op.register_fake
def _fwd_fake(z, eps):
    return (
        torch.empty_like(z),
        torch.empty(_rho_size(z), dtype=torch.float32, device=z.device),
    )


@torch.library.custom_op("nelu_surr::bwd", mutates_args=(), device_types="cuda")
def _bwd_op(z: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    z = z.contiguous()
    dy = dy.contiguous()
    z_2d = _to_2d(z)
    dy_2d = _to_2d(dy)
    dz = _nelu_cuda.backward_surr(z_2d, dy_2d)
    return dz.reshape(z.shape)


@_bwd_op.register_fake
def _bwd_fake(z, dy):
    return torch.empty_like(z)


def _setup_ctx(ctx, inputs, output):
    z, _eps = inputs
    ctx.save_for_backward(z)  # only z — surrogate bwd doesn't need ρ


def _backward(ctx, grad_y, grad_rho):
    z, = ctx.saved_tensors
    return _bwd_op(z, grad_y), None


torch.library.register_autograd(
    "nelu_surr::fwd", _backward, setup_context=_setup_ctx)


def nelu_cuda_surr(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y, _ = _fwd_op(z, float(eps))
    return y


class NELUCUDA_Surr(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nelu_cuda_surr(z, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, backend=custom_op_surr"
