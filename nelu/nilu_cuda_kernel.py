"""Fused CUDA kernel for NiLU with a learnable per-channel gamma vector.

Mirrors nelu/cuda_kernel.py. The underlying CUDA kernel still reduces over the
last dimension; the public wrapper exposes a general `dims=...` interface by
reordering / flattening the requested reduction dims before dispatch.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from .reduction_layout import DimsLike, canonicalize_reduce_dims, flatten_reduction_dims, restore_reduction_dims

_dir = os.path.dirname(os.path.abspath(__file__))
_nilu_cuda = load(
    name="nilu_cuda",
    sources=[os.path.join(_dir, "csrc", "nilu_cuda.cu")],
    verbose=False,
)


# ── Custom op: forward ─────────────────────────────────────────

@torch.library.custom_op("nilu::fwd", mutates_args=(), device_types="cuda")
def _nilu_fwd_op(z: torch.Tensor, gamma: torch.Tensor, eps: float
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    g = gamma.contiguous().to(torch.float32)
    y, rho = _nilu_cuda.forward(z, g, eps)
    return y, rho


@_nilu_fwd_op.register_fake
def _nilu_fwd_fake(z: torch.Tensor, gamma: torch.Tensor, eps: float):
    M = z.numel() // z.size(-1)
    return (
        torch.empty_like(z),
        torch.empty(M, dtype=torch.float32, device=z.device),
    )


# ── Custom op: backward ─────────────────────────────────────────

@torch.library.custom_op("nilu::bwd", mutates_args=(), device_types="cuda")
def _nilu_bwd_op(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor,
                 gamma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    dy = dy.contiguous()
    g = gamma.contiguous().to(torch.float32)
    dz, dgamma = _nilu_cuda.backward(z, rho, dy, g)
    return dz, dgamma


@_nilu_bwd_op.register_fake
def _nilu_bwd_fake(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor,
                   gamma: torch.Tensor):
    return (
        torch.empty_like(z),
        torch.empty(gamma.numel(), dtype=torch.float32, device=z.device),
    )


# ── Autograd wiring ─────────────────────────────────────────────

def _nilu_setup_context(ctx, inputs, output):
    z, gamma, _eps = inputs
    _y, rho = output
    ctx.save_for_backward(z, rho, gamma)


def _nilu_backward(ctx, grad_y, grad_rho):
    z, rho, gamma = ctx.saved_tensors
    grad_z, dgamma = _nilu_bwd_op(z, rho, grad_y, gamma)
    dg = dgamma.to(gamma.dtype).reshape(gamma.shape)
    return grad_z, dg, None


torch.library.register_autograd(
    "nilu::fwd",
    _nilu_backward,
    setup_context=_nilu_setup_context,
)


# ── Public API ──────────────────────────────────────────────────

def _prepare_gamma_vector(gamma: torch.Tensor, reduced_size: int, device: torch.device) -> torch.Tensor:
    if gamma.device != device:
        gamma = gamma.to(device)
    if gamma.numel() == 1:
        return gamma.reshape(1).expand(reduced_size)
    if gamma.numel() == reduced_size:
        return gamma.reshape(reduced_size)
    raise ValueError(
        f"gamma must be scalar or have {reduced_size} elements for the flattened reduction axis, "
        f"got shape {tuple(gamma.shape)}"
    )


def nilu_cuda(z: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6,
              dims: DimsLike = -1) -> torch.Tensor:
    dims = canonicalize_reduce_dims(z.ndim, dims)
    z_flat, layout = flatten_reduction_dims(z, dims)
    gamma_vec = _prepare_gamma_vector(gamma, z_flat.size(-1), z.device)
    y_flat, _rho = _nilu_fwd_op(z_flat, gamma_vec, float(eps))
    return restore_reduction_dims(y_flat, layout)


class NiLUCUDA(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6,
                 gamma_init: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.full((num_channels,),
                                             float(gamma_init),
                                             dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nilu_cuda(z, self.gamma, self.eps)

    def extra_repr(self) -> str:
        return (f"eps={self.eps}, C={self.gamma.numel()}, "
                f"gamma_mean={self.gamma.mean().item():.6f}, backend=cuda")
