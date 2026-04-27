"""NELU variant without internal normalization — pure affine gate input.

Where standard NELU normalizes the gate input by the RMS of the
activations, and ``NELU_LN`` uses LayerNorm-style stats, this variant
strips normalization entirely:

    y = x · g(γ · x + β)

The gate sees the raw activation magnitude. This is meaningful only
because every CIFAR backbone in our zoo runs BatchNorm immediately
before the activation — so ``x`` already has roughly unit variance per
channel, and the affine ``γ x + β`` is a learnable rescale + shift on
top of that. With γ=1, β=0 this is exactly SiLU; learning γ and β as
scalars generalizes to the family of "Swish-β" activations while
keeping the channel-mixing-aware NELU bookkeeping (norm_axes, etc.)
unchanged for compatibility with our swap policy.

β's role here is the most direct of the three variants: it's the
absolute decision boundary of the gate. β > 0 keeps weak negatives
alive (wide gating, eff-rank ↑); β < 0 only lets strong positives
through (sparse gating, Fisher ↑).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core import GateNorm
from .layout import DimsLike, NormAxes


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _phi(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


class _GateAffine(GateNorm):
    """Affine-only gate: ``y = x · g(γ·x + β)``. Subclasses pick ``g``.

    Inherits ``gamma`` from :class:`GateNorm`, adds a single learnable
    scalar ``beta``, and overrides forward to skip normalization. The
    fused CUDA kernel is disabled (``_CUDA_KIND = None``).
    """

    _CUDA_KIND = None

    def __init__(
        self,
        norm_axes: NormAxes | DimsLike = "channel",
        *,
        eps: float = 1e-6,
        gamma_init: float = 1.0,
        beta_init: float = 0.0,
    ) -> None:
        super().__init__(norm_axes, eps=eps, gamma_init=gamma_init)
        self.beta = nn.Parameter(
            torch.full((1,), float(beta_init), dtype=torch.float32)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z32 = z.float() if z.dtype != torch.float32 else z
        gate = type(self)._gate_python(self.gamma * z32 + self.beta)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        return (
            f"norm_axes={self.norm_axes!r}, "
            f"gamma={self.gamma.item():.3e}, beta={self.beta.item():.3e}"
        )


class NELU_AFF(_GateAffine):
    """NELU variant without normalization: ``y = x · Φ(γ·x + β)``."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU_AFF(_GateAffine):
    """NiLU variant without normalization: ``y = x · σ(γ·x + β)`` (= Swish-β)."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)
