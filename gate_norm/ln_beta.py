"""NELU variant with LayerNorm-style stats and a learnable bias β.

Standard NELU is

    y = x · g(γ · x / rms(x))

where the gate input is *only* scaled (RMSNorm-style). This variant uses
the LayerNorm-style normalize (subtract mean, divide by std) and adds a
learnable bias β to the gate input:

    y = x · g(γ · (x − μ) / σ + β)

The β shifts the sigmoid/Φ operating point. β << 0 → sparse gating
(strong-channel selection), β = 0 → NELU-like balanced gating, β >> 0 →
wide gating (most channels pass through, recovering effective rank).

Both γ and β are scalar learnable parameters per module (channel-wise
β was considered but kept scalar for parity with the rest of the
codebase and minimal parameter overhead). Gradient flow drives them
jointly with the rest of the model.

This module is a drop-in replacement for ``NELU`` / ``NiLU`` and is
swapped in via ``train.swap.replace_activation_auto_axes`` with
``cls=NELU_LN`` or ``cls=NiLU_LN``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core import GateNorm
from .layout import DimsLike, NormAxes, resolve_axes


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _phi(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


class _GateNormLN(GateNorm):
    """LN-stats + β-shifted gate. Subclasses pick the gate function.

    Inherits the ``gamma`` parameter and constructor from :class:`GateNorm`
    and adds a single learnable scalar ``beta`` that shifts the gate's
    operating point. Disables the fused CUDA path (``_CUDA_KIND = None``)
    so the LN-style stats and β take the python forward.
    """

    _CUDA_KIND = None

    def __init__(
        self,
        norm_axes: NormAxes | DimsLike = "position",
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
        axes = resolve_axes(z.ndim, self.norm_axes)
        z32 = z.float() if z.dtype != torch.float32 else z
        mu = z32.mean(dim=axes, keepdim=True)
        var = z32.var(dim=axes, keepdim=True, unbiased=False)
        rsigma = (var + self.eps).rsqrt()
        z_norm = (z32 - mu) * rsigma
        gate = type(self)._gate_python(self.gamma * z_norm + self.beta)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        return (
            f"norm_axes={self.norm_axes!r}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e}, beta={self.beta.item():.3e}"
        )


class NELU_LN(_GateNormLN):
    """NELU with LayerNorm-style stats and a learnable bias β."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU_LN(_GateNormLN):
    """NiLU with LayerNorm-style stats and a learnable bias β."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)
