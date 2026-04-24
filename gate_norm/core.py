"""Gate Normalization — the core building block.

Gate Normalization (GN) generalizes self-gated activations of the form
``x · g(x)`` by centering and rescaling the gate input:

    y = x · g(γ · (x - μ) / σ + β)

where ``g`` is a pointwise squashing function (sigmoid, Gaussian CDF, …),
``μ, σ`` are the mean and standard deviation of ``x`` computed over
architecture-specific axes, and ``γ``, ``β`` are learnable scalars. The outer
multiplication by ``x`` preserves the feature's DC component; only the gate
decision is shift- and scale-invariant. ``γ`` controls how strongly the
normalized gate participates; ``β`` shifts the gate's operating point away
from ``t = 0``. At ``γ = β = 0`` the module reduces to ``x · g(0)``.

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

from .dispatch import should_use_cuda
from .layout import DimsLike, NormAxes, resolve_axes
from .stats import layer_stats


_DEFAULT_GAMMA_INIT = 0.0
_DEFAULT_BETA_INIT = 0.0
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class GateNorm(nn.Module):
    """Shift- and scale-invariant self-gated activation with learnable γ, β.

    Parameters
    ----------
    norm_axes : str or tuple of int, default ``"channel"``
        Axes over which mean/variance are computed. Either a preset alias
        (``"channel"`` for channel-mixing inputs, ``"sample"`` for NCHW
        convolutions whose preceding linear mixes both channel and space)
        or an explicit axis tuple matching the mixing axes of the preceding
        linear operation (e.g. ``(2, 3)`` after a depthwise convolution).
    eps : float, default ``1e-6``
        Numerical floor added inside the variance before sqrt.
    gamma_init : float, default ``0.0``
        Initial value of γ. Zero init gives an exact ``y = x · g(0)``
        identity at start; training grows ``|γ|`` as needed.
    beta_init : float, default ``0.0``
        Initial value of β. Zero keeps the gate's operating point at
        ``t = 0`` at init, where ``g(0)`` matches the identity-at-init
        behavior of GELU/SiLU.
    """

    # Name of the fused CUDA op registered in :mod:`gate_norm.cuda`. Subclasses
    # set this to opt into the kernel; ``None`` keeps the module on the
    # pure-PyTorch path.
    _CUDA_OP: str | None = None

    def __init__(
        self,
        norm_axes: NormAxes | DimsLike = "channel",
        *,
        eps: float = 1e-6,
        gamma_init: float = _DEFAULT_GAMMA_INIT,
        beta_init: float = _DEFAULT_BETA_INIT,
    ) -> None:
        super().__init__()
        self.norm_axes = norm_axes
        self.eps = eps
        self.gamma = nn.Parameter(
            torch.full((1,), float(gamma_init), dtype=torch.float32)
        )
        self.beta = nn.Parameter(
            torch.full((1,), float(beta_init), dtype=torch.float32)
        )
        # Marker so `collect_gamma_stats` can discover us generically.
        self._gate_norm_module = True

    # Subclass hook ─────────────────────────────────────────────────

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        """Pointwise squashing function applied to ``γ · (x - μ)/σ + β``."""
        raise NotImplementedError

    # Forward ───────────────────────────────────────────────────────

    def _axes(self, z: torch.Tensor) -> tuple[int, ...]:
        return resolve_axes(z.ndim, self.norm_axes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        axes = self._axes(z)
        if self._CUDA_OP is not None and should_use_cuda(z):
            from . import cuda  # Lazy: only imported on CUDA devices.
            return cuda.fused_forward(
                self._CUDA_OP, z, self.gamma, self.beta, axes, self.eps,
            )

        mu, rsigma = layer_stats(z, axes, self.eps)
        # The outer multiplication keeps the caller's dtype so the AMP
        # autocast contract is preserved; the gate path stays in float32.
        z32 = z.float() if z.dtype != torch.float32 else z
        t = self.gamma * (z32 - mu) * rsigma + self.beta
        gate = type(self)._gate_python(t)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        return (
            f"norm_axes={self.norm_axes!r}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e}, beta={self.beta.item():.3e}"
        )

    # Checkpoint compatibility ─────────────────────────────────────

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Older checkpoints saved γ as a 0-dim scalar; reshape to (1,).
        gamma_key = prefix + "gamma"
        t = state_dict.get(gamma_key)
        if isinstance(t, torch.Tensor) and t.ndim == 0:
            state_dict[gamma_key] = t.reshape(1)
        # β was introduced in gate_norm v0.3. Older checkpoints don't carry
        # it; synthesize a zero value so strict loading still succeeds and
        # resume from pre-β checkpoints is seamless.
        beta_key = prefix + "beta"
        if beta_key not in state_dict:
            state_dict[beta_key] = torch.zeros(1, dtype=torch.float32)
        else:
            t = state_dict[beta_key]
            if isinstance(t, torch.Tensor) and t.ndim == 0:
                state_dict[beta_key] = t.reshape(1)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


# ── Functional form (γ, β required, for quick tests) ──────────────────────


def gate_norm(
    z: torch.Tensor,
    gate_fn: Callable[[torch.Tensor], torch.Tensor],
    gamma: float | torch.Tensor = 1.0,
    beta: float | torch.Tensor = 0.0,
    *,
    norm_axes: NormAxes | DimsLike = "channel",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional form: ``z · gate_fn(γ · (z - μ(z)) / σ(z) + β)``.

    Provided primarily for unit tests and experiments; production code should
    use :class:`GateNorm` so ``γ`` and ``β`` are learnable parameters.
    """
    axes = resolve_axes(z.ndim, norm_axes)
    mu, rsigma = layer_stats(z, axes, eps)

    def _reshape_scalar(x, name):
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError(
                    f"{name} must be a scalar, got shape {tuple(x.shape)}"
                )
            return x.reshape(())
        return x

    gamma = _reshape_scalar(gamma, "gamma")
    beta = _reshape_scalar(beta, "beta")
    z32 = z.float() if z.dtype != torch.float32 else z
    return z * gate_fn(gamma * (z32 - mu) * rsigma + beta).to(z.dtype)
