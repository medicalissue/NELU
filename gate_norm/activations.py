"""Gate-normalized instances of GELU and SiLU.

* ``NELU`` — gate function is the Gaussian CDF Φ, so ``NELU`` is the
  scale-invariant counterpart of GELU.
* ``NiLU`` — gate function is the sigmoid, making ``NiLU`` the
  scale-invariant counterpart of SiLU.

Both are drop-in replacements for their baselines: construct one in place of
``nn.GELU()`` / ``nn.SiLU()`` and pass ``norm_axes`` matching the mixing
axes of the preceding linear operation. ``γ`` is a non-learnable buffer
driven by the trainer's warmup scheduler (see :mod:`gate_norm.scheduler`).
"""

from __future__ import annotations

import math

import torch

from .core import GateNorm


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _phi(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


class NELU(GateNorm):
    """GELU with gate normalization: ``y = z · Φ(γ · z / rms(z))``."""

    _CUDA_OP = "nelu"

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU(GateNorm):
    """SiLU with gate normalization: ``y = z · σ(γ · z / rms(z))``."""

    _CUDA_OP = "nilu"

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)
