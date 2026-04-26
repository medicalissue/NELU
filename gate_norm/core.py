"""Gate Normalization — the core building block.

Gate Normalization (GN) generalizes self-gated activations of the form
``x · g(x)`` by normalizing the gate input by its root-mean-square:

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise squashing function (sigmoid, Gaussian CDF, …)
and ``γ`` is a learnable scalar with a soft positivity constraint via
softplus reparameterization: ``γ_eff = softplus(γ_raw)``. This keeps γ
strictly positive (anti-gate sign-flip is unphysical for self-gated
activations — `Φ(γ x̂)` with γ < 0 inverts the gate so the layer outputs
are non-positive, breaking the "save positives, drop negatives"
inductive bias of ReLU/GELU/SiLU) while leaving γ_eff = 0 unreachable
in finite training so the activation never collapses to a pure linear
pass. The outer multiplication by ``x`` preserves the feature's DC
component so the activation retains the family's deactivation bias.

The forward pass is pure PyTorch so it composes cleanly with torch.compile,
AMP autocast, and arbitrary reduction axes (including non-contiguous subsets
such as ``(2, 3)`` used after depthwise convolutions). Statistics are upcast
to float32 internally to avoid fp16/bf16 underflow.

Subclasses implement :meth:`_gate_python` and advertise their fused kernel
name via ``_CUDA_OP``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dispatch import should_use_cuda
from .layout import DimsLike, NormAxes, resolve_axes
from .stats import layer_stats


_DEFAULT_GAMMA_INIT = 0.0
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class GateNorm(nn.Module):
    """Scale-invariant self-gated activation with a learnable scalar γ.

    Parameters
    ----------
    norm_axes : str or tuple of int, default ``"channel"``
        Axes over which the RMS is computed.
    eps : float, default ``1e-6``
        Numerical floor added inside the RMS before sqrt.
    gamma_init : float, default ``0.0``
        Initial value of the *raw* parameter γ_raw (the unconstrained
        scalar). The effective gate temperature is
        ``γ_eff = softplus(γ_raw)``. Defaults: γ_raw=0 → γ_eff=ln 2 ≈
        0.693, gradient gate sigmoid(0)=0.5 (well-conditioned).
        Pick γ_raw=1 for γ_eff≈1.31, γ_raw=-2 for γ_eff≈0.13, etc.
    """

    _CUDA_OP: str | None = None

    def __init__(
        self,
        norm_axes: NormAxes | DimsLike = "channel",
        *,
        eps: float = 1e-6,
        gamma_init: float = _DEFAULT_GAMMA_INIT,
    ) -> None:
        super().__init__()
        self.norm_axes = norm_axes
        self.eps = eps
        # gamma_init now sets γ_raw directly (no inv_softplus).
        self.gamma_raw = nn.Parameter(
            torch.full((1,), float(gamma_init), dtype=torch.float32),
            requires_grad=True,
        )
        self._gate_norm_module = True

    @property
    def gamma(self) -> torch.Tensor:
        """Effective γ used by the gate: softplus(γ_raw), strictly positive."""
        return F.softplus(self.gamma_raw)

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _axes(self, z: torch.Tensor) -> tuple[int, ...]:
        return resolve_axes(z.ndim, self.norm_axes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # The CUDA fused path bakes γ in as a Python float and drops the
        # gradient, so always use the pure-PyTorch path. inductor still fuses.
        axes = self._axes(z)
        gamma_eff = self.gamma
        rsigma = layer_stats(z, axes, self.eps)
        z32 = z.float() if z.dtype != torch.float32 else z
        t = gamma_eff * z32 * rsigma
        gate = type(self)._gate_python(t)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        return (
            f"norm_axes={self.norm_axes!r}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e} (raw={self.gamma_raw.item():.3e})"
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Back-compat: legacy `gamma` (effective γ stored directly) → gamma_raw.
        gamma_key = prefix + "gamma"
        gamma_raw_key = prefix + "gamma_raw"
        if gamma_key in state_dict and gamma_raw_key not in state_dict:
            t = state_dict.pop(gamma_key)
            if isinstance(t, torch.Tensor):
                t = t.reshape(1).float().clamp(min=1e-12)
                state_dict[gamma_raw_key] = torch.log(torch.expm1(t))
        beta_key = prefix + "beta"
        if beta_key in state_dict:
            state_dict.pop(beta_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


# ── Functional form (γ explicit, for quick tests) ─────────────────────────


def gate_norm(
    z: torch.Tensor,
    gate_fn: Callable[[torch.Tensor], torch.Tensor],
    gamma: float | torch.Tensor = 1.0,
    *,
    norm_axes: NormAxes | DimsLike = "channel",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional form: ``z · gate_fn(γ · z / rms(z))``."""
    axes = resolve_axes(z.ndim, norm_axes)
    rsigma = layer_stats(z, axes, eps)
    if isinstance(gamma, torch.Tensor):
        if gamma.numel() != 1:
            raise ValueError(f"gamma must be a scalar, got shape {tuple(gamma.shape)}")
        gamma = gamma.reshape(())
    z32 = z.float() if z.dtype != torch.float32 else z
    return z * gate_fn(gamma * z32 * rsigma).to(z.dtype)
