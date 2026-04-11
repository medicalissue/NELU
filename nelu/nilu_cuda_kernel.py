"""Fused CUDA kernel for NiLU.

Mirrors nelu/cuda_kernel.py — same custom_op + fake function pattern,
different math (sigmoid). Public API:  `nilu_cuda(z, eps)` and
`class NiLUCUDA(nn.Module)`.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_nilu_cuda = load(
    name="nilu_cuda",
    sources=[os.path.join(_dir, "csrc", "nilu_cuda.cu")],
    verbose=False,
)


# ── Shape helpers ──────────────────────────────────────────────

def _to_2d(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 4:
        return z.reshape(z.size(0), -1)
    return z.reshape(-1, z.size(-1))


def _rho_size(z: torch.Tensor) -> int:
    if z.dim() == 4:
        return z.size(0)
    M = 1
    for d in z.shape[:-1]:
        M *= d
    return M


# ── Custom op: forward ─────────────────────────────────────────

@torch.library.custom_op("nilu::fwd", mutates_args=(), device_types="cuda")
def _nilu_fwd_op(z: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    z_2d = _to_2d(z)
    y, rho = _nilu_cuda.forward(z_2d, eps)
    return y.reshape(z.shape), rho


@_nilu_fwd_op.register_fake
def _nilu_fwd_fake(z: torch.Tensor, eps: float):
    return (
        torch.empty_like(z),
        torch.empty(_rho_size(z), dtype=torch.float32, device=z.device),
    )


# ── Custom op: backward ─────────────────────────────────────────

@torch.library.custom_op("nilu::bwd", mutates_args=(), device_types="cuda")
def _nilu_bwd_op(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    z = z.contiguous()
    dy = dy.contiguous()
    z_2d = _to_2d(z)
    dy_2d = _to_2d(dy)
    dz = _nilu_cuda.backward(z_2d, rho, dy_2d)
    return dz.reshape(z.shape)


@_nilu_bwd_op.register_fake
def _nilu_bwd_fake(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor):
    return torch.empty_like(z)


# ── Autograd wiring ─────────────────────────────────────────────

def _nilu_setup_context(ctx, inputs, output):
    z, _eps = inputs
    _y, rho = output
    ctx.save_for_backward(z, rho)


def _nilu_backward(ctx, grad_y, grad_rho):
    z, rho = ctx.saved_tensors
    grad_z = _nilu_bwd_op(z, rho, grad_y)
    return grad_z, None


torch.library.register_autograd(
    "nilu::fwd",
    _nilu_backward,
    setup_context=_nilu_setup_context,
)


# ── Public API ──────────────────────────────────────────────────

def nilu_cuda(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y, _rho = _nilu_fwd_op(z, float(eps))
    return y


class NiLUCUDA(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nilu_cuda(z, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, backend=custom_op"
