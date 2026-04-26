"""Fused CUDA backend for Gate Normalization.

Two custom ops are exported through ``torch.library``:

* ``gate_norm::fused_fwd(z, gamma, kind, eps) -> (y, rsigma)``
* ``gate_norm::fused_bwd(z, rsigma, dy, gamma, kind) -> (dz, dgamma)``

Both operate on a 2-D ``(M, N)`` view; the wrapper in
:func:`gate_norm_cuda_forward` flattens arbitrary reduction axes via
:func:`gate_norm.layout.flatten_for_reduction` and restores the original
shape on the way out.

``γ`` is a learnable scalar ``(1,)`` fp32 tensor. Forward saves
``rsigma`` (per-row reciprocal RMS, fp32) for the backward; backward
returns both ``dz`` (in z's dtype) and ``dgamma`` (scalar fp32, summed
across all rows).

The CUDA extension is compiled lazily on first call via
:func:`torch.utils.cpp_extension.load`. CUDA-less platforms (e.g. macOS
during development) never reach that path because :mod:`gate_norm.dispatch`
gates the import inside :meth:`gate_norm.GateNorm.forward`.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from .layout import flatten_for_reduction, restore


_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

# Kept in sync with the GateKind enum in gate_norm_common.cuh.
GATE_KIND_PHI: int = 0      # NELU
GATE_KIND_SIGMOID: int = 1  # NiLU


# ── Lazy extension loading ───────────────────────────────────────────────

_ext = None


def _load_extension():
    """JIT-compile the fused kernel. First call may take 30–60 s; cached
    in ``~/.cache/torch_extensions/`` thereafter so subsequent imports
    are essentially free.
    """
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    _ext = load(
        name="gate_norm_fused",
        sources=[os.path.join(_CSRC, "gate_norm.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-Xptxas=-v",
        ],
        verbose=False,
    )
    return _ext


# ── Custom ops + autograd ────────────────────────────────────────────────

_FWD = "gate_norm::fused_fwd"
_BWD = "gate_norm::fused_bwd"


@torch.library.custom_op(_FWD, mutates_args=(), device_types="cuda")
def _fused_fwd(
    z: torch.Tensor, gamma: torch.Tensor, kind: int, eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    y, rsigma = ext.forward(
        z.contiguous(),
        gamma.contiguous().to(torch.float32),
        int(kind),
        float(eps),
    )
    return y, rsigma


@_fused_fwd.register_fake
def _(z, gamma, kind, eps):
    m = z.size(0)
    return (
        torch.empty_like(z),
        torch.empty(m, dtype=torch.float32, device=z.device),
    )


@torch.library.custom_op(_BWD, mutates_args=(), device_types="cuda")
def _fused_bwd(
    z: torch.Tensor, rsigma: torch.Tensor, dy: torch.Tensor,
    gamma: torch.Tensor, kind: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    dz, dgamma = ext.backward(
        z.contiguous(),
        rsigma,
        dy.contiguous(),
        gamma.contiguous().to(torch.float32),
        int(kind),
    )
    return dz, dgamma


@_fused_bwd.register_fake
def _(z, rsigma, dy, gamma, kind):
    return (
        torch.empty_like(z),
        torch.empty(1, dtype=torch.float32, device=z.device),
    )


def _setup_ctx(ctx, inputs, output):
    z, gamma, kind, _eps = inputs
    _y, rsigma = output
    ctx.save_for_backward(z, gamma, rsigma)
    ctx.kind = int(kind)


def _backward(ctx, grad_y, grad_rsigma):
    """Autograd backward. ``grad_rsigma`` is unused — rsigma is an
    auxiliary that flows back through forward into our backward kernel,
    not a user-visible output."""
    z, gamma, rsigma = ctx.saved_tensors
    dz, dgamma = torch.ops.gate_norm.fused_bwd(
        z, rsigma, grad_y, gamma, ctx.kind,
    )
    # dgamma is fp32 scalar (1,); cast back to gamma's dtype/shape so the
    # autograd engine accumulates it into gamma.grad.
    return (dz, dgamma.to(gamma.dtype).reshape(gamma.shape), None, None)


torch.library.register_autograd(_FWD, _backward, setup_context=_setup_ctx)


# ── Public entry — used by gate_norm.core.GateNorm.forward ───────────────


def gate_norm_cuda_forward(
    z: torch.Tensor,
    gamma: torch.Tensor,
    kind: int,
    axes: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    """Run the fused GateNorm kernel and return ``y`` in the caller's layout.

    Parameters
    ----------
    z : torch.Tensor
        Input tensor of arbitrary rank.
    gamma : torch.Tensor
        Scalar learnable parameter, shape ``(1,)``.
    kind : int
        Gate kind (``0`` → NELU/Φ, ``1`` → NiLU/σ).
    axes : tuple of int
        Reduction axes (already canonicalized via ``layout.resolve_axes``).
    eps : float
        Numerical floor inside the RMS.

    Returns
    -------
    torch.Tensor
        ``y`` with the same shape and dtype as ``z``.
    """
    z_flat, layout = flatten_for_reduction(z, axes)
    flat_shape = z_flat.shape           # (kept..., N)
    N = flat_shape[-1]
    z_2d = z_flat.reshape(-1, N).contiguous()
    y_2d, _rsigma = torch.ops.gate_norm.fused_fwd(z_2d, gamma, int(kind), float(eps))
    return restore(y_2d.reshape(flat_shape), layout)
