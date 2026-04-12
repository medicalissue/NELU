"""Fused CUDA kernel for NiLU — Stop-Gradient (SG) variant.

Forward identical to NoSG: y_i = z_i * sigma(z_i / rho).
Backward: dz_j = g_j * (sigma(t_j) + t_j * sigma'(t_j))  — NO cross-term.
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


@torch.library.custom_op("nilu_sg::fwd", mutates_args=(), device_types="cuda")
def _fwd_op(z: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.contiguous()
    z_2d = _to_2d(z)
    y, rho = _nilu_cuda.forward(z_2d, eps)
    return y.reshape(z.shape), rho


@_fwd_op.register_fake
def _fwd_fake(z, eps):
    return (
        torch.empty_like(z),
        torch.empty(_rho_size(z), dtype=torch.float32, device=z.device),
    )


@torch.library.custom_op("nilu_sg::bwd", mutates_args=(), device_types="cuda")
def _bwd_op(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    z = z.contiguous()
    dy = dy.contiguous()
    z_2d = _to_2d(z)
    dy_2d = _to_2d(dy)
    dz = _nilu_cuda.backward_sg(z_2d, rho, dy_2d)
    return dz.reshape(z.shape)


@_bwd_op.register_fake
def _bwd_fake(z, rho, dy):
    return torch.empty_like(z)


def _setup_ctx(ctx, inputs, output):
    z, _eps = inputs
    _y, rho = output
    ctx.save_for_backward(z, rho)


def _backward(ctx, grad_y, grad_rho):
    z, rho = ctx.saved_tensors
    return _bwd_op(z, rho, grad_y), None


torch.library.register_autograd(
    "nilu_sg::fwd", _backward, setup_context=_setup_ctx)


def nilu_cuda_sg(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y, _ = _fwd_op(z, float(eps))
    return y


class NiLUCUDA_SG(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nilu_cuda_sg(z, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, backend=custom_op_sg"
