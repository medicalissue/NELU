"""NELU variant: LayerNorm-style normalize + channel-wise affine gate.

The default NELU activation:

    y = x · g(γ_c · LN_c(x) + β_c)

where:
  * ``LN_c(x) = (x − μ_c) / sqrt(var_c + ε)`` with ``μ_c, var_c`` pooled
    over the **position axis** (CNN: spatial H×W; Transformer: tokens T;
    rank-dispatched by :func:`gate_norm.layout.resolve_axes`).
  * ``γ_c, β_c`` are **channel-wise** learnable vectors (per-channel
    sharpness and operating point). Materialized lazily on the first
    forward — we don't know the channel count at swap time.
  * ``g`` is a pointwise gate (Φ for NELU, σ for NiLU).

Why per-channel γ, β
--------------------
Position-axis pooling already extracts a per-channel statistic; giving
each channel its own ``(γ_c, β_c)`` lets the gate be channel-aware in
both sharpness and threshold. This is a strict generalization of the
older scalar-γ NELU.

Why mean-subtract (LN), not RMS-only
-------------------------------------
The "pool over position → broadcast back" picture is strictly
shift-invariant: ``x → x + c`` should not change the gate. Subtracting
the per-channel mean is the natural realization of that invariant; the
``+ β_c`` term re-introduces a learnable, channel-wise threshold so we
don't lose the bias absorption.

Notes
-----
``_CUDA_KIND = None`` disables the fused kernel for this module — the
LN stats and per-channel parameters take the Python forward.
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
    """LN normalize + channel-wise affine gate. Subclasses pick the gate.

    Forward::

        μ_c, var_c = mean / var over position axis
        z_norm = (z − μ_c) / sqrt(var_c + ε)
        gate = g(γ_c · z_norm + β_c)
        y = z * gate

    γ_c, β_c are length-C learnable vectors, materialized lazily.
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
        # Skip GateNorm's scalar gamma init — we use lazy per-channel γ_c, β_c.
        nn.Module.__init__(self)
        self.norm_axes = norm_axes
        self.eps = eps
        self._gate_norm_module = True
        self._gamma_init = float(gamma_init)
        self._beta_init = float(beta_init)
        self.gamma = nn.UninitializedParameter()
        self.beta = nn.UninitializedParameter()

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _materialize(self, n_channels: int, device, dtype) -> None:
        if isinstance(self.gamma, nn.UninitializedParameter):
            self.gamma.materialize((n_channels,), device=device, dtype=torch.float32)
            with torch.no_grad():
                self.gamma.fill_(self._gamma_init)
        if isinstance(self.beta, nn.UninitializedParameter):
            self.beta.materialize((n_channels,), device=device, dtype=torch.float32)
            with torch.no_grad():
                self.beta.fill_(self._beta_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        axes = resolve_axes(z.ndim, self.norm_axes)
        # Channel axis convention: 1 for NCHW (4D), -1 otherwise (B,T,C / B,C).
        if z.ndim == 4:
            channel_dim = 1
            shape = (1, z.size(1), 1, 1)
        else:
            channel_dim = -1
            shape = (1,) * (z.ndim - 1) + (z.size(-1),)
        n_channels = z.size(channel_dim)
        self._materialize(n_channels, z.device, z.dtype)

        z32 = z.float() if z.dtype != torch.float32 else z
        mu = z32.mean(dim=axes, keepdim=True)
        var = z32.var(dim=axes, keepdim=True, unbiased=False)
        rsigma = (var + self.eps).rsqrt()
        z_norm = (z32 - mu) * rsigma

        gamma = self.gamma.view(shape)
        beta = self.beta.view(shape)
        gate = type(self)._gate_python(gamma * z_norm + beta)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        if isinstance(self.gamma, nn.UninitializedParameter):
            return f"norm_axes={self.norm_axes!r}, eps={self.eps}, channel-wise (lazy)"
        return (
            f"norm_axes={self.norm_axes!r}, eps={self.eps}, "
            f"gamma=Vec[{self.gamma.numel()}] (mean={self.gamma.mean().item():.3e}), "
            f"beta=Vec[{self.beta.numel()}]  (mean={self.beta.mean().item():.3e})"
        )


class NELU_LN(_GateNormLN):
    """NELU: LN-normalize + channel-wise affine + Φ gate."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return _phi(t)


class NiLU_LN(_GateNormLN):
    """NiLU: LN-normalize + channel-wise affine + σ gate."""

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)
