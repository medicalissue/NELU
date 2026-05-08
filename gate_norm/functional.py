"""Functional helpers for quick experiments — RMS-only legacy form.

``nelu(z, γ)`` and ``nilu(z, γ)`` mirror the *legacy* RMS-only,
scalar-γ activation (matching :class:`gate_norm.NELU_RMS` /
:class:`gate_norm.NiLU_RMS`). They are kept for unit tests and
quick standalone experiments — the default :class:`gate_norm.NELU` /
:class:`gate_norm.NiLU` modules use LN-normalize + per-channel γ_c, β_c,
which doesn't fit a small functional helper signature.
"""

from __future__ import annotations

import math

import torch

from .layout import DimsLike, NormAxes, resolve_axes
from .stats import layer_stats


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _gated(z, gate_fn, gamma, axes, eps: float) -> torch.Tensor:
    rsigma = layer_stats(z, axes, eps)
    # Outer multiplication stays in the caller's dtype; statistics and
    # the gate are resolved in float32 to match the module path.
    z32 = z.float() if z.dtype != torch.float32 else z
    gate = gate_fn(gamma * z32 * rsigma)
    return z * gate.to(z.dtype)


def nelu(
    z: torch.Tensor,
    gamma: float | torch.Tensor = 1.0,
    *,
    norm_axes: NormAxes | DimsLike = "position",
    eps: float = 1e-6,
) -> torch.Tensor:
    """``z · Φ(γ · z / rms(z))`` with Φ the Gaussian CDF."""
    axes = resolve_axes(z.ndim, norm_axes)
    return _gated(
        z,
        lambda t: 0.5 * (1.0 + torch.erf(t * _INV_SQRT2)),
        gamma, axes, eps,
    )


def nilu(
    z: torch.Tensor,
    gamma: float | torch.Tensor = 1.0,
    *,
    norm_axes: NormAxes | DimsLike = "position",
    eps: float = 1e-6,
) -> torch.Tensor:
    """``z · σ(γ · z / rms(z))`` with σ the logistic sigmoid."""
    axes = resolve_axes(z.ndim, norm_axes)
    return _gated(z, torch.sigmoid, gamma, axes, eps)
