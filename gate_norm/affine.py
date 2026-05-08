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


class _GateAffineCW(GateNorm):
    """Channel-wise affine gate: ``y = x · g(γ_c · x + β_c)``.

    γ and β are per-channel learnable vectors, allowing each channel to
    pick its own sigmoid sharpness and operating point. Parameters are
    initialized lazily on the first forward (we don't know the channel
    count at swap time without changing the swap policy).

    Memory cost vs scalar variant: ``2 × C`` extra floats per layer,
    negligible compared to the surrounding conv weights.
    """

    _CUDA_KIND = None

    def __init__(
        self,
        norm_axes: NormAxes | DimsLike = "position",
        *,
        eps: float = 1e-6,
        gamma_init: float = 1.0,
        beta_init: float = 0.0,
        num_channels: int | None = None,
    ) -> None:
        # Skip GateNorm's gamma init (it's a scalar there); we'll either
        # eagerly create per-channel γ_c, β_c (when num_channels is
        # supplied by the swap policy) or fall back to lazy init.
        nn.Module.__init__(self)
        self.norm_axes = norm_axes
        self.eps = eps
        self._gate_norm_module = True
        self._gamma_init = float(gamma_init)
        self._beta_init = float(beta_init)
        if num_channels is not None:
            self.gamma = nn.Parameter(
                torch.full((int(num_channels),), float(gamma_init), dtype=torch.float32)
            )
            self.beta = nn.Parameter(
                torch.full((int(num_channels),), float(beta_init), dtype=torch.float32)
            )
        else:
            self.gamma = nn.UninitializedParameter()
            self.beta = nn.UninitializedParameter()

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _materialize(self, n_channels: int, device, dtype):
        if isinstance(self.gamma, nn.UninitializedParameter):
            self.gamma.materialize((n_channels,), device=device, dtype=torch.float32)
            with torch.no_grad():
                self.gamma.fill_(self._gamma_init)
        if isinstance(self.beta, nn.UninitializedParameter):
            self.beta.materialize((n_channels,), device=device, dtype=torch.float32)
            with torch.no_grad():
                self.beta.fill_(self._beta_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Channel axis is dim=1 for NCHW (4D) and dim=-1 for (B, D) / (B, L, D).
        # We use the convention that channels are the last axis the gate
        # broadcasts to — dim=1 for 4D, dim=-1 otherwise.
        if z.ndim == 4:
            channel_dim = 1
            shape = (1, z.size(1), 1, 1)
        else:
            channel_dim = -1
            shape = (1,) * (z.ndim - 1) + (z.size(-1),)
        n_channels = z.size(channel_dim)
        self._materialize(n_channels, z.device, z.dtype)
        z32 = z.float() if z.dtype != torch.float32 else z
        gamma = self.gamma.view(shape)
        beta = self.beta.view(shape)
        gate = type(self)._gate_python(gamma * z32 + beta)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        if isinstance(self.gamma, nn.UninitializedParameter):
            return f"norm_axes={self.norm_axes!r}, channel-wise (lazy)"
        return (
            f"norm_axes={self.norm_axes!r}, "
            f"gamma=Vec[{self.gamma.numel()}] (mean={self.gamma.mean().item():.3e}), "
            f"beta=Vec[{self.beta.numel()}] (mean={self.beta.mean().item():.3e})"
        )


class NELU_AFFCW(_GateAffineCW):
    """Channel-wise NELU_AFF: ``y = x · Φ(γ_c·x + β_c)``."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU_AFFCW(_GateAffineCW):
    """Channel-wise NiLU_AFF: ``y = x · σ(γ_c·x + β_c)``."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)
