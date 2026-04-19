"""Fused CUDA kernel for NELU with a learnable per-channel gamma vector.

Forward:  t[m,n] = gamma[n] * z[m,n] / rho[m],   y[m,n] = z[m,n] * Phi(t[m,n])
Backward: (dz, dgamma)   — dgamma is length-N, summed across all rows.

Wired through `torch.library.custom_op` so `torch.compile` can
FakeTensor-trace it without graph-breaking. The underlying CUDA kernel
normalizes over the last dimension only; the public wrapper exposes a
more general `dims=...` interface by reordering / flattening the chosen
reduction dims into a trailing axis before dispatch.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from .reduction_layout import DimsLike, canonicalize_reduce_dims, flatten_reduction_dims, restore_reduction_dims

_dir = os.path.dirname(os.path.abspath(__file__))
_nelu_cuda = load(
    name="nelu_cuda",
    sources=[os.path.join(_dir, "csrc", "nelu_cuda.cu")],
    verbose=False,
)


# ── Custom op: forward ─────────────────────────────────────────

@torch.library.custom_op("nelu::fwd", mutates_args=(), device_types="cuda")
def _nelu_fwd_op(z: torch.Tensor, gamma: torch.Tensor, eps: float
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    g = gamma.contiguous().to(torch.float32)
    y, rho = _nelu_cuda.forward(z, g, eps)
    return y, rho


@_nelu_fwd_op.register_fake
def _nelu_fwd_fake(z: torch.Tensor, gamma: torch.Tensor, eps: float):
    # rho has one scalar per row in the reshape (M = numel // last_dim).
    M = z.numel() // z.size(-1)
    return (
        torch.empty_like(z),
        torch.empty(M, dtype=torch.float32, device=z.device),
    )


# ── Custom op: backward ─────────────────────────────────────────

@torch.library.custom_op("nelu::bwd", mutates_args=(), device_types="cuda")
def _nelu_bwd_op(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor,
                 gamma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    dy = dy.contiguous()
    g = gamma.contiguous().to(torch.float32)
    dz, dgamma = _nelu_cuda.backward(z, rho, dy, g)
    return dz, dgamma


@_nelu_bwd_op.register_fake
def _nelu_bwd_fake(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor,
                   gamma: torch.Tensor):
    return (
        torch.empty_like(z),
        torch.empty(gamma.numel(), dtype=torch.float32, device=z.device),
    )


# ── Autograd wiring ─────────────────────────────────────────────

def _nelu_setup_context(ctx, inputs, output):
    z, gamma, _eps = inputs
    _y, rho = output
    ctx.save_for_backward(z, rho, gamma)


def _nelu_backward(ctx, grad_y, grad_rho):
    z, rho, gamma = ctx.saved_tensors
    grad_z, dgamma = _nelu_bwd_op(z, rho, grad_y, gamma)
    dg = dgamma.to(gamma.dtype).reshape(gamma.shape)
    return grad_z, dg, None  # eps is not differentiable


torch.library.register_autograd(
    "nelu::fwd",
    _nelu_backward,
    setup_context=_nelu_setup_context,
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


def nelu_cuda(z: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6,
              dims: DimsLike = -1) -> torch.Tensor:
    """Forward NELU with a scalar gamma or length-K gamma vector.

    The fused CUDA kernel itself reduces over the last dimension. This wrapper
    accepts arbitrary `dims`, flattens them into one trailing axis, and restores
    the original layout afterwards.
    """
    dims = canonicalize_reduce_dims(z.ndim, dims)
    z_flat, layout = flatten_reduction_dims(z, dims)
    gamma_vec = _prepare_gamma_vector(gamma, z_flat.size(-1), z.device)
    y_flat, _rho = _nelu_fwd_op(z_flat, gamma_vec, float(eps))
    return restore_reduction_dims(y_flat, layout)


class NELUCUDA(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6,
                 gamma_init: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.full((num_channels,),
                                             float(gamma_init),
                                             dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nelu_cuda(z, self.gamma, self.eps)

    def extra_repr(self) -> str:
        return (f"eps={self.eps}, C={self.gamma.numel()}, "
                f"gamma_mean={self.gamma.mean().item():.6f}, backend=cuda")
