"""Fused CUDA backend for Gate Normalization.

Forward returns ``(y, μ, rσ)`` and stashes ``(μ, rσ)`` in the autograd
context so the backward reuses them instead of recomputing row statistics.

γ and β are scalar parameters in the Python module. Just before the kernel
we reshape each to a length-N broadcast view; the kernel then accumulates
the per-feature gradients into length-N buffers and the ``ExpandBackward``
around the scalar input sums them to shape ``(1,)`` in the autograd graph.
Registration happens once at import time; :func:`fused_forward` is the sole
public entry point, parameterized by the GateNorm op name.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from .layout import flatten_for_reduction, restore


_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

_GATE_KIND = {"nelu": 0, "nilu": 1}  # kept in sync with csrc/gate_norm.cu


# ── Extension loading ────────────────────────────────────────────────────

_ext = None


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load

    _ext = load(
        name="gate_norm_fused",
        sources=[os.path.join(_CSRC, "gate_norm.cu")],
        verbose=False,
    )
    return _ext


# ── Custom ops ───────────────────────────────────────────────────────────

_FWD = "gate_norm::fused_fwd"
_BWD = "gate_norm::fused_bwd"


@torch.library.custom_op(_FWD, mutates_args=(), device_types="cuda")
def _fused_fwd(
    z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    kind: int, eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    return ext.forward(
        z.contiguous(),
        gamma.contiguous().to(torch.float32),
        beta.contiguous().to(torch.float32),
        int(kind), float(eps),
    )


@_fused_fwd.register_fake
def _(z, gamma, beta, kind, eps):
    m = z.numel() // z.size(-1)
    return (
        torch.empty_like(z),
        torch.empty(m, dtype=torch.float32, device=z.device),
        torch.empty(m, dtype=torch.float32, device=z.device),
    )


@torch.library.custom_op(_BWD, mutates_args=(), device_types="cuda")
def _fused_bwd(
    z: torch.Tensor, mu: torch.Tensor, rsigma: torch.Tensor,
    dy: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    kind: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    return ext.backward(
        z.contiguous(), mu, rsigma, dy.contiguous(),
        gamma.contiguous().to(torch.float32),
        beta.contiguous().to(torch.float32),
        int(kind),
    )


@_fused_bwd.register_fake
def _(z, mu, rsigma, dy, gamma, beta, kind):
    return (
        torch.empty_like(z),
        torch.empty(gamma.numel(), dtype=torch.float32, device=z.device),
        torch.empty(beta.numel(),  dtype=torch.float32, device=z.device),
    )


def _setup_ctx(ctx, inputs, output):
    z, gamma, beta, kind, _eps = inputs
    _y, mu, rsigma = output
    ctx.save_for_backward(z, mu, rsigma, gamma, beta)
    ctx.kind = int(kind)


def _backward(ctx, grad_y, grad_mu, grad_rsigma):
    z, mu, rsigma, gamma, beta = ctx.saved_tensors
    dz, dgamma, dbeta = torch.ops.gate_norm.fused_bwd(
        z, mu, rsigma, grad_y, gamma, beta, ctx.kind,
    )
    return (
        dz,
        dgamma.to(gamma.dtype),
        dbeta.to(beta.dtype),
        None,  # kind
        None,  # eps
    )


torch.library.register_autograd(_FWD, _backward, setup_context=_setup_ctx)


# ── Public entry point ───────────────────────────────────────────────────


def _expand_scalar(
    p: torch.Tensor, reduced: int, device: torch.device, name: str
) -> torch.Tensor:
    """Scalar → length-``reduced`` broadcast view on the right device."""
    if p.numel() != 1:
        raise ValueError(
            f"gate_norm CUDA wrapper only accepts a scalar {name} "
            f"(shape (1,)), got shape {tuple(p.shape)}"
        )
    if p.device != device:
        p = p.to(device)
    return p.reshape(1).expand(reduced)


def fused_forward(
    op: str,
    z: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    axes: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    """Run the fused GateNorm kernel and return ``y`` in the caller's layout.

    ``op`` selects the gate function (``"nelu"`` → Gaussian CDF,
    ``"nilu"`` → sigmoid). ``axes`` must be pre-canonicalized via
    :func:`gate_norm.layout.resolve_axes`.
    """
    if op not in _GATE_KIND:
        raise ValueError(f"unknown gate op: {op!r}")
    z_flat, layout = flatten_for_reduction(z, axes)
    reduced = z_flat.size(-1)
    gvec = _expand_scalar(gamma, reduced, z.device, "gamma")
    bvec = _expand_scalar(beta,  reduced, z.device, "beta")
    y_flat, _mu, _rsigma = torch.ops.gate_norm.fused_fwd(
        z_flat, gvec, bvec, _GATE_KIND[op], float(eps),
    )
    return restore(y_flat, layout)
