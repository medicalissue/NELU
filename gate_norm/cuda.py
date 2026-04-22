"""Fused CUDA backends for Gate Normalization.

The two kernels (NELU, NiLU) share identical plumbing — a forward op that
returns `(y, rho)`, a backward op that returns `(dz, dgamma)`, and autograd
glue that threads `rho` from forward to backward. Both kernels reduce over
the trailing axis only; the wrapper permutes and flattens the requested
reduction axes before dispatch and restores the original layout afterwards.

γ is always a single learnable scalar. The kernels accept a length-N gamma
vector (one entry per reduced feature), so we broadcast the scalar to the
required length immediately before the launch. No per-channel path is
exposed at the Python level.
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import torch

from .reduction import (
    DimsLike,
    ReductionLayout,
    RmsMode,
    flatten_for_reduction,
    rms_axes,
    restore,
)


_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")


def _load_extension(name: str, source: str):
    """JIT-compile a CUDA extension shipped in `gate_norm/csrc/`."""
    from torch.utils.cpp_extension import load

    return load(
        name=name,
        sources=[os.path.join(_CSRC, source)],
        verbose=False,
    )


_nelu_ext = None
_nilu_ext = None


def _get_ext(op: str):
    global _nelu_ext, _nilu_ext
    if op == "nelu":
        if _nelu_ext is None:
            _nelu_ext = _load_extension("gate_norm_nelu", "nelu_cuda.cu")
        return _nelu_ext
    if op == "nilu":
        if _nilu_ext is None:
            _nilu_ext = _load_extension("gate_norm_nilu", "nilu_cuda.cu")
        return _nilu_ext
    raise ValueError(f"unknown op: {op}")


# ── Custom ops & autograd ────────────────────────────────────────────────
#
# We register one pair of torch.library.custom_op's per kernel. This is
# required so `torch.compile` can FakeTensor-trace the extension without
# graph-breaking. The per-op registration is one-time; guard with a flag.

_REGISTERED: set[str] = set()


def _register(op: str) -> None:
    if op in _REGISTERED:
        return
    _REGISTERED.add(op)

    fwd_name = f"gate_norm::{op}_fwd"
    bwd_name = f"gate_norm::{op}_bwd"

    @torch.library.custom_op(fwd_name, mutates_args=(), device_types="cuda")
    def _fwd(z: torch.Tensor, gamma: torch.Tensor, eps: float
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        ext = _get_ext(op)
        z = z.contiguous()
        g = gamma.contiguous().to(torch.float32)
        y, rho = ext.forward(z, g, eps)
        return y, rho

    @_fwd.register_fake
    def _(z, gamma, eps):
        m = z.numel() // z.size(-1)
        return (
            torch.empty_like(z),
            torch.empty(m, dtype=torch.float32, device=z.device),
        )

    @torch.library.custom_op(bwd_name, mutates_args=(), device_types="cuda")
    def _bwd(z: torch.Tensor, rho: torch.Tensor, dy: torch.Tensor,
             gamma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ext = _get_ext(op)
        dz, dgamma = ext.backward(
            z.contiguous(), rho, dy.contiguous(),
            gamma.contiguous().to(torch.float32),
        )
        return dz, dgamma

    @_bwd.register_fake
    def _(z, rho, dy, gamma):
        return (
            torch.empty_like(z),
            torch.empty(gamma.numel(), dtype=torch.float32, device=z.device),
        )

    def _setup(ctx, inputs, output):
        z, gamma, _eps = inputs
        _y, rho = output
        ctx.save_for_backward(z, rho, gamma)

    def _backward(ctx, grad_y, grad_rho):
        z, rho, gamma = ctx.saved_tensors
        grad_z, dgamma = torch.ops.gate_norm.__getattr__(f"{op}_bwd")(
            z, rho, grad_y, gamma
        )
        return grad_z, dgamma.to(gamma.dtype).reshape(gamma.shape), None

    torch.library.register_autograd(fwd_name, _backward, setup_context=_setup)


def _call_fused(op: str, z: torch.Tensor, gamma: torch.Tensor, eps: float):
    _register(op)
    fn = torch.ops.gate_norm.__getattr__(f"{op}_fwd")
    y, _rho = fn(z, gamma, eps)
    return y


# ── Public dispatch ──────────────────────────────────────────────────────


def _expand_scalar_gamma(gamma: torch.Tensor, reduced: int,
                         device: torch.device) -> torch.Tensor:
    """Scalar γ → length-`reduced` broadcast view on the right device."""
    if gamma.numel() != 1:
        raise ValueError(
            "gate_norm CUDA wrapper only accepts a scalar gamma "
            f"(shape (1,)), got shape {tuple(gamma.shape)}"
        )
    if gamma.device != device:
        gamma = gamma.to(device)
    return gamma.reshape(1).expand(reduced)


def _fused(op: str, z: torch.Tensor, gamma: torch.Tensor,
           axes: tuple[int, ...], eps: float) -> torch.Tensor:
    z_flat, layout = flatten_for_reduction(z, axes)
    gvec = _expand_scalar_gamma(gamma, z_flat.size(-1), z.device)
    y_flat = _call_fused(op, z_flat, gvec, float(eps))
    return restore(y_flat, layout)


def nelu_cuda(z: torch.Tensor, gamma: torch.Tensor, *,
              axes: tuple[int, ...], eps: float = 1e-6) -> torch.Tensor:
    """Forward NELU on CUDA: y = z * Φ(γ · z / rms_axes(z)).

    `gamma` must be a scalar tensor of shape `(1,)`. Reduction axes must be
    pre-canonicalized (use `reduction.rms_axes`).
    """
    return _fused("nelu", z, gamma, axes, eps)


def nilu_cuda(z: torch.Tensor, gamma: torch.Tensor, *,
              axes: tuple[int, ...], eps: float = 1e-6) -> torch.Tensor:
    """Forward NiLU on CUDA: y = z * σ(γ · z / rms_axes(z))."""
    return _fused("nilu", z, gamma, axes, eps)
