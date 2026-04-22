"""Functional helpers for quick experiments.

These mirror ``torch.nn.functional.gelu`` / ``torch.nn.functional.silu`` but
apply the Gate Normalization rescaling with a user-supplied γ. Production
code should use the :class:`gate_norm.NELU` and :class:`gate_norm.NiLU`
modules so γ is a learnable parameter.
"""

from __future__ import annotations

import math

import torch

from .reduction import DimsLike, RmsMode, rms, rms_axes


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def nelu(
    z: torch.Tensor,
    gamma: float | torch.Tensor = 1.0,
    *,
    rms_mode: RmsMode | DimsLike = "per_token",
    eps: float = 1e-6,
) -> torch.Tensor:
    """``z · Φ(γ · z / rms(z))`` with Φ the Gaussian CDF."""
    axes = rms_axes(z.ndim, rms_mode)
    rho = rms(z, axes, eps)
    t = gamma * z / rho
    return z * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


def nilu(
    z: torch.Tensor,
    gamma: float | torch.Tensor = 1.0,
    *,
    rms_mode: RmsMode | DimsLike = "per_token",
    eps: float = 1e-6,
) -> torch.Tensor:
    """``z · σ(γ · z / rms(z))`` with σ the logistic sigmoid."""
    axes = rms_axes(z.ndim, rms_mode)
    rho = rms(z, axes, eps)
    return z * torch.sigmoid(gamma * z / rho)
