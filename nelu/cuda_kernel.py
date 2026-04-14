"""Fused CUDA kernel for NELU with a learnable per-channel gamma vector.

Forward:  t[m,n] = gamma[n] * z[m,n] / rho[m],   y[m,n] = z[m,n] * Phi(t[m,n])
Backward: (dz, dgamma)   — dgamma is length-N, summed across all rows.

Wired through `torch.library.custom_op` so `torch.compile` can
FakeTensor-trace it without graph-breaking. The CUDA kernel only
handles the last-dim-as-channel layout (2D / 3D). 4D (NCHW) inputs
are handled in the Python reference in `activations.py`.
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

def nelu_cuda(z: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6
              ) -> torch.Tensor:
    """Forward NELU with a length-C gamma vector (C = z.size(-1)).

    `gamma` must be a 1-D tensor of length z.size(-1). It can live in
    any float dtype — it's cast to fp32 inside the kernel.
    """
    if gamma.device != z.device:
        gamma = gamma.to(z.device)
    y, _rho = _nelu_fwd_op(z, gamma, float(eps))
    return y


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
