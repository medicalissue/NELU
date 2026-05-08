"""Default NELU and NiLU activations.

NELU and NiLU are the canonical gate-normalized activations of the
paper. Both share the same recipe — pool over the position axis, get a
per-channel statistic, normalize, gate — and differ only in the gate
function:

* :class:`NELU` — Gaussian CDF Φ. Counterpart of GELU.
* :class:`NiLU` — sigmoid σ.    Counterpart of SiLU.

Form
----
::

    μ_c, var_c = pool over position axis (CNN: H×W; Transformer: T)
    z_norm     = (x − μ_c) / sqrt(var_c + ε)
    gate       = g(γ_c · z_norm + β_c)        # γ_c, β_c learnable per channel
    y          = x · gate

The position axis is rank-dispatched at runtime via the
``"position"`` alias in :func:`gate_norm.layout.resolve_axes`. The
per-channel parameters are materialized lazily on the first forward.

Backward-compatible aliases
---------------------------
The previous RMS-only, scalar-γ NELU is kept available as
:class:`NELU_RMS` / :class:`NiLU_RMS` for ablation. The fused CUDA
kernel covers only those, since it expects scalar γ.
"""

from __future__ import annotations

import math

import torch

from .core import GateNorm
from .ln_beta import _GateNormLN


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _phi(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


# ── Default: LN-normalize + channel-wise affine ──────────────────────────


class NELU(_GateNormLN):
    """Default NELU: ``y = x · Φ(γ_c · LN_c(x) + β_c)``."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU(_GateNormLN):
    """Default NiLU: ``y = x · σ(γ_c · LN_c(x) + β_c)``."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)


# ── Legacy RMS-only, scalar γ (kept for ablation) ────────────────────────


class NELU_RMS(GateNorm):
    """Legacy RMS-only NELU: ``y = x · Φ(γ · x / rms(x))``.

    The earlier paper draft used this scalar-γ, RMSNorm-style gate. Kept
    here so the fused CUDA kernel and ablations have a clean reference.
    """

    _CUDA_KIND = 0  # GATE_PHI

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU_RMS(GateNorm):
    """Legacy RMS-only NiLU: ``y = x · σ(γ · x / rms(x))``."""

    _CUDA_KIND = 1  # GATE_SIGMOID

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)
