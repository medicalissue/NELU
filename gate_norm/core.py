"""Gate Normalization — the core building block.

Gate Normalization (GN) generalizes self-gated activations of the form
``x · g(x)`` by normalizing the gate input by its root-mean-square:

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise squashing function (sigmoid, Gaussian CDF, …)
and ``γ`` is a learnable scalar with a soft positivity constraint:
``γ_eff = softplus(γ_raw)``. The softplus reparameterization keeps γ
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


_DEFAULT_GAMMA_INIT = 1e-6
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _inv_softplus(y: float) -> float:
    """Inverse of softplus: solve softplus(x) = y for x."""
    if y <= 0:
        raise ValueError(f"softplus output must be > 0, got {y}")
    # x = log(exp(y) - 1); use expm1 for numerical stability near y=0.
    return math.log(math.expm1(y))


class GateNorm(nn.Module):
    """Scale-invariant self-gated activation with a learnable scalar γ.

    Parameters
    ----------
    norm_axes : str or tuple of int, default ``"channel"``
        Axes over which the RMS is computed. Either a preset alias
        (``"channel"`` for channel-mixing inputs, ``"sample"`` for NCHW
        convolutions whose preceding linear mixes both channel and space)
        or an explicit axis tuple matching the mixing axes of the preceding
        linear operation (e.g. ``(2, 3)`` after a depthwise convolution).
    eps : float, default ``1e-6``
        Numerical floor added inside the RMS before sqrt.
    gamma_init : float, default ``1e-6``
        Initial *effective* γ (i.e. γ_eff = softplus(γ_raw)). The raw
        parameter is set to inv_softplus(gamma_init).
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
        # γ_raw is the unconstrained learnable parameter; γ_eff =
        # softplus(γ_raw) is the value used in the forward gate.
        raw = _inv_softplus(float(gamma_init))
        self.gamma_raw = nn.Parameter(
            torch.full((1,), raw, dtype=torch.float32),
            requires_grad=True,
        )
        self._gate_norm_module = True

    @property
    def gamma(self) -> torch.Tensor:
        """Effective γ used by the gate: softplus(γ_raw), strictly positive."""
        return F.softplus(self.gamma_raw)

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        """Pointwise squashing function applied to ``γ · x / rms(x)``."""
        raise NotImplementedError

    def _axes(self, z: torch.Tensor) -> tuple[int, ...]:
        return resolve_axes(z.ndim, self.norm_axes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        axes = self._axes(z)
        gamma_eff = self.gamma  # softplus(γ_raw)
        # CUDA fused path bakes γ into the kernel as a Python float and
        # therefore drops the gradient w.r.t. γ_raw. Skip it whenever γ
        # is learnable and we're training.
        gamma_grad_active = self.gamma_raw.requires_grad and torch.is_grad_enabled()
        if (
            self._CUDA_OP is not None
            and should_use_cuda(z)
            and not gamma_grad_active
        ):
            from . import cuda
            return cuda.fused_forward(
                self._CUDA_OP, z, gamma_eff, axes, self.eps,
            )

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
        # Back-compat: older checkpoints stored `gamma` directly (the
        # effective γ value). Convert to `gamma_raw = inv_softplus(γ)`.
        gamma_key = prefix + "gamma"
        gamma_raw_key = prefix + "gamma_raw"
        if gamma_key in state_dict and gamma_raw_key not in state_dict:
            t = state_dict.pop(gamma_key)
            if isinstance(t, torch.Tensor):
                t = t.reshape(1).float()
                t = t.clamp(min=1e-12)  # softplus output must be > 0
                raw = torch.log(torch.expm1(t))
                state_dict[gamma_raw_key] = raw
        # Drop legacy β key if present.
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
    """Functional form: ``z · gate_fn(γ · z / rms(z))``.

    Note: the functional form takes γ directly (not γ_raw); use the
    :class:`GateNorm` module if you want the softplus reparameterization.
    """
    axes = resolve_axes(z.ndim, norm_axes)
    rsigma = layer_stats(z, axes, eps)

    if isinstance(gamma, torch.Tensor):
        if gamma.numel() != 1:
            raise ValueError(
                f"gamma must be a scalar, got shape {tuple(gamma.shape)}"
            )
        gamma = gamma.reshape(())

    z32 = z.float() if z.dtype != torch.float32 else z
    return z * gate_fn(gamma * z32 * rsigma).to(z.dtype)
