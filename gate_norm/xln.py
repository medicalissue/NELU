"""xLN — LayerNorm as a multiplicative self-gate.

Where standard NELU squashes the gate input through a pointwise function
(Φ for NELU, σ for NiLU), this variant uses LayerNorm itself as the
"squashing" function: the gate is the layer-normalized value of the
input, and the activation multiplies the input by it.

    xLN(x) = γ_c · (x · normalize_axes(x)) + β_c

where:
  * ``normalize_axes(x) = (x - μ) / σ`` over the swap-policy-resolved
    axes (matches the mixing axes of the preceding linear op — same
    convention as NELU).
  * ``γ_c, β_c`` are *outer*, channel-wise learnable parameters (the
    LayerNorm-standard affine, lifted out of the gate so it doesn't get
    absorbed into the normalization).

Init
----
``γ_c₀ = 0`` is the LayerScale-style choice (DeiT-III, ConvNeXt): the
multiplicative branch starts as zero, so each residual block begins as
the identity, and γ_c grows to switch the branch on as training
progresses. ``γ_c₀ = 1`` is the more conventional "active from the
start" choice. Both are exposed via ``gamma_init`` for ablation.

Why outer affine, not inner
---------------------------
``γ · normalize(x) + β`` keeps the affine effective. Putting it inside
(``normalize(γ·x + β)``) makes both parameters dead weight: the mean
absorbs β and the std absorbs γ. The dual-of-DyT analogy goes the same
way — DyT is ``γ · tanh(α·x) + β`` (outer affine), so xLN keeps its
affine outer too.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .core import GateNorm
from .layout import DimsLike, NormAxes, resolve_axes


class _xLN_Base(GateNorm):
    """Multiplicative LayerNorm gate with outer channel-wise affine.

    Subclasses pick a gate-output transform (typically identity for the
    classic ``x · LN(x)`` form). γ_c, β_c are length-C vectors,
    materialized lazily on the first forward (the swap policy doesn't
    surface num_features, so we infer it from input shape).
    """

    _CUDA_KIND = None

    def __init__(
        self,
        norm_axes: NormAxes | DimsLike = "channel",
        *,
        eps: float = 1e-6,
        gamma_init: float = 0.0,   # LayerScale-style by default
        beta_init: float = 0.0,
    ) -> None:
        # Skip GateNorm's scalar gamma init; xLN wants a per-channel
        # vector so we materialize lazily.
        nn.Module.__init__(self)
        self.norm_axes = norm_axes
        self.eps = eps
        self._gate_norm_module = True
        self._gamma_init = float(gamma_init)
        self._beta_init = float(beta_init)
        self.gamma = nn.UninitializedParameter()
        self.beta = nn.UninitializedParameter()

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
        axes = resolve_axes(z.ndim, self.norm_axes)
        z32 = z.float() if z.dtype != torch.float32 else z
        # Channel axis: 1 for NCHW (4D), -1 otherwise. Match NELU convention.
        channel_dim = 1 if z.ndim == 4 else -1
        n_channels = z.size(channel_dim)
        self._materialize(n_channels, z.device, z.dtype)

        # Normalize over the swap-resolved axes (no inner affine).
        mu = z32.mean(dim=axes, keepdim=True)
        var = z32.var(dim=axes, keepdim=True, unbiased=False)
        rsigma = (var + self.eps).rsqrt()
        z_norm = (z32 - mu) * rsigma

        # Multiplicative gate: x · normalize(x), then outer channel-wise affine.
        gated = z32 * z_norm
        if z.ndim == 4:
            shape = (1, n_channels, 1, 1)
        else:
            shape = (1,) * (z.ndim - 1) + (n_channels,)
        gamma = self.gamma.view(shape)
        beta = self.beta.view(shape)
        out = gamma * gated + beta
        return out.to(z.dtype)

    def extra_repr(self) -> str:
        if isinstance(self.gamma, nn.UninitializedParameter):
            return (
                f"norm_axes={self.norm_axes!r}, gamma_init={self._gamma_init}, "
                f"beta_init={self._beta_init} (lazy)"
            )
        return (
            f"norm_axes={self.norm_axes!r}, "
            f"gamma=Vec[{self.gamma.numel()}] (mean={self.gamma.mean().item():.3e}), "
            f"beta=Vec[{self.beta.numel()}] (mean={self.beta.mean().item():.3e})"
        )


class xLN(_xLN_Base):
    """LayerNorm as a multiplicative self-gate.

    Equivalent to ``γ_c · (x · ((x - μ) / σ)) + β_c`` where (μ, σ) reduce
    over the axes the upstream linear op mixed.
    """

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        # Not used — xLN's gate is structural, not pointwise.
        raise NotImplementedError("xLN does not have a pointwise gate function")
