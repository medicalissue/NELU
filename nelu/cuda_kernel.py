"""Fused CUDA kernel for NELU.

The C++ side (nelu_cuda.cu) exposes raw forward/backward primitives.
Autograd is wired in Python via `torch.library.custom_op` + a registered
fake function so `torch.compile` can FakeTensor-trace through the op
without crashing or graph-breaking.

API stays the same:  `nelu_cuda(z, eps)` and `class NELUCUDA(nn.Module)`.
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


# ── Shape helpers ──────────────────────────────────────────────

def _to_2d(z: torch.Tensor) -> torch.Tensor:
    """Reshape to (M, N). 4D inputs collapse channels+spatial; otherwise
    treat last dim as feature dim."""
    if z.dim() == 4:
        return z.reshape(z.size(0), -1)
    return z.reshape(-1, z.size(-1))


def _rho_size(z: torch.Tensor) -> int:
    """Number of rows in the (M, N) reshape — needed by the fake fn."""
    if z.dim() == 4:
        return z.size(0)
    M = 1
    for d in z.shape[:-1]:
        M *= d
    return M


# ── Custom op: forward ─────────────────────────────────────────
#
# Returns (y, rho) so backward has access to rho without recomputing it.
# The user-facing wrapper drops rho.

@torch.library.custom_op("nelu::fwd", mutates_args=(), device_types="cuda")
def _nelu_fwd_op(z: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    z_2d = _to_2d(z)
    y, rho = _nelu_cuda.forward(z_2d, eps)
    return y.reshape(z.shape), rho


@_nelu_fwd_op.register_fake
def _nelu_fwd_fake(z: torch.Tensor, eps: float):
    return (
        torch.empty_like(z),
        torch.empty(_rho_size(z), dtype=torch.float32, device=z.device),
    )


# ── Custom op: backward ─────────────────────────────────────────

@torch.library.custom_op("nelu::bwd", mutates_args=(), device_types="cuda")
def _nelu_bwd_op(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    z = z.contiguous()
    dy = dy.contiguous()
    z_2d = _to_2d(z)
    dy_2d = _to_2d(dy)
    dz = _nelu_cuda.backward(z_2d, rho, dy_2d)
    return dz.reshape(z.shape)


@_nelu_bwd_op.register_fake
def _nelu_bwd_fake(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor):
    return torch.empty_like(z)


# ── Autograd wiring ─────────────────────────────────────────────

def _nelu_setup_context(ctx, inputs, output):
    z, _eps = inputs
    _y, rho = output
    ctx.save_for_backward(z, rho)


def _nelu_backward(ctx, grad_y, grad_rho):
    z, rho = ctx.saved_tensors
    grad_z = _nelu_bwd_op(z, rho, grad_y)
    return grad_z, None  # eps is not differentiable


torch.library.register_autograd(
    "nelu::fwd",
    _nelu_backward,
    setup_context=_nelu_setup_context,
)


# ── Public API ──────────────────────────────────────────────────

def nelu_cuda(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y, _rho = _nelu_fwd_op(z, float(eps))
    return y


class NELUCUDA(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nelu_cuda(z, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, backend=custom_op"
