"""Fused CUDA backend for the v0.4 RMS-only Gate Normalization form.

The kernel implements::

    rsigma[m] = 1 / sqrt(mean(x[m,:]²) + eps)
    y[m,n]    = x[m,n] · g(γ · x[m,n] · rsigma[m])

with ``g`` selected by an integer kind (``0`` → Φ, NELU; ``1`` → σ, NiLU)
and ``γ`` passed as a Python float (the buffer value at trace time).

We register a single ``torch.library.custom_op`` for the forward and an
explicit autograd backward; this is enough for ``torch.compile`` to flow
through the op without graph breaks. The Python module
:class:`gate_norm.GateNorm` opts in by setting ``_CUDA_OP`` to the gate
name; subclasses ``NELU`` / ``NiLU`` toggle this on a per-class basis.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from .layout import flatten_for_reduction, restore


_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

_GATE_KIND = {"nelu": 0, "nilu": 1}


# ── Extension loading ────────────────────────────────────────────────────

_ext = None


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    _ext = load(
        name="gate_norm_fused_v04",
        sources=[os.path.join(_CSRC, "gate_norm.cu")],
        verbose=False,
    )
    return _ext


# ── Custom ops ───────────────────────────────────────────────────────────

_FWD = "gate_norm::fused_fwd_v04"
_BWD = "gate_norm::fused_bwd_v04"


@torch.library.custom_op(_FWD, mutates_args=(), device_types="cuda")
def _fused_fwd(
    x: torch.Tensor, gamma: float, kind: int, eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    y, rsigma = ext.forward(x.contiguous(), float(gamma), int(kind), float(eps))
    return y, rsigma


@_fused_fwd.register_fake
def _(x, gamma, kind, eps):
    m = x.size(0)
    return (
        torch.empty_like(x),
        torch.empty(m, dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(_BWD, mutates_args=(), device_types="cuda")
def _fused_bwd(
    x: torch.Tensor, rsigma: torch.Tensor, dy: torch.Tensor,
    gamma: float, kind: int,
) -> torch.Tensor:
    ext = _load_extension()
    return ext.backward(
        x.contiguous(), rsigma, dy.contiguous(),
        float(gamma), int(kind),
    )


@_fused_bwd.register_fake
def _(x, rsigma, dy, gamma, kind):
    return torch.empty_like(x)


def _setup_ctx(ctx, inputs, output):
    x, gamma, kind, _eps = inputs
    _y, rsigma = output
    ctx.save_for_backward(x, rsigma)
    ctx.gamma = float(gamma)
    ctx.kind = int(kind)


def _backward(ctx, grad_y, grad_rsigma):
    x, rsigma = ctx.saved_tensors
    dx = torch.ops.gate_norm.fused_bwd_v04(
        x, rsigma, grad_y, ctx.gamma, ctx.kind,
    )
    return (dx, None, None, None)  # x, gamma, kind, eps


torch.library.register_autograd(_FWD, _backward, setup_context=_setup_ctx)


# ── Public entry ─────────────────────────────────────────────────────────


def fused_forward(
    op: str,
    z: torch.Tensor,
    gamma: torch.Tensor,
    axes: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    """Run the fused GateNorm kernel and return ``y`` in the caller's layout.

    ``op`` selects the gate (``"nelu"`` → Φ, ``"nilu"`` → σ). ``gamma`` is
    a length-1 buffer; we pass its scalar value in as a Python float so
    inductor specializes the kernel on it (this matches the assumption
    that γ is non-trainable and only changes via the warmup scheduler at
    epoch boundaries).
    """
    if op not in _GATE_KIND:
        raise ValueError(f"unknown gate op: {op!r}")
    z_flat, layout = flatten_for_reduction(z, axes)
    # The CUDA kernel expects a 2-D (M, N) view; flatten the kept leading
    # dims into M before dispatching, then reshape the kernel's 2-D output
    # back into the caller's layout via restore().
    flat_shape = z_flat.shape  # (kept..., N)
    n_red = flat_shape[-1]
    z_2d = z_flat.reshape(-1, n_red)
    g = float(gamma.detach().reshape(()).item())
    y_2d, _rsigma = torch.ops.gate_norm.fused_fwd_v04(
        z_2d, g, _GATE_KIND[op], float(eps),
    )
    y_flat = y_2d.reshape(flat_shape)
    return restore(y_flat, layout)
