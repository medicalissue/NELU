"""Gate Normalization — the core building block.

Gate Normalization (GN) generalizes self-gated activations of the form
`x · g(x)` by normalizing the gate input:

    y = x · g(γ · x / rms(x))

where `g` is a pointwise squashing function (sigmoid, Gaussian CDF, …),
`rms(x)` is the root-mean-square of `x` computed over architecture-specific
axes, and `γ` is a single learnable scalar initialized near zero so that at
initialization the module recovers the near-linear behavior of `x · g(0)`.

Subclasses define a concrete `g` and optionally a fused CUDA backend by
implementing `_gate_python` and `_fused_backend`.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from .reduction import DimsLike, RmsMode, rms, rms_axes


_DEFAULT_GAMMA_INIT = 1e-6


class GateNorm(nn.Module):
    """Scale-invariant self-gated activation with a learnable scalar γ.

    Parameters
    ----------
    rms_mode : str or tuple of int, default ``"per_token"``
        Axes over which the RMS is computed. Either a preset alias
        (``"per_token"`` for channels-last / token inputs, ``"per_sample"``
        for NCHW convolutions) or an explicit axis tuple.
    eps : float, default ``1e-6``
        Numerical floor added inside the RMS square-root.
    gamma_init : float, default ``1e-6``
        Initial value of γ. Small values keep the gate near its zero-input
        baseline at initialization; training grows γ as needed.
    """

    def __init__(
        self,
        rms_mode: RmsMode | DimsLike = "per_token",
        *,
        eps: float = 1e-6,
        gamma_init: float = _DEFAULT_GAMMA_INIT,
    ) -> None:
        super().__init__()
        self.rms_mode = rms_mode
        self.eps = eps
        self.gamma = nn.Parameter(
            torch.full((1,), float(gamma_init), dtype=torch.float32)
        )
        # Marker so `collect_gamma_stats` can discover us generically.
        self._gate_norm_module = True

    # Subclass hooks ────────────────────────────────────────────────

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        """Pointwise squashing function applied to γ · x / rms(x)."""
        raise NotImplementedError

    def _fused_backend(
        self,
    ) -> Optional[
        Callable[..., torch.Tensor]
    ]:
        """Return a CUDA kernel callable, or None for the Python path."""
        return None

    # Forward ───────────────────────────────────────────────────────

    def _axes(self, z: torch.Tensor) -> tuple[int, ...]:
        return rms_axes(z.ndim, self.rms_mode)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        axes = self._axes(z)
        if z.is_cuda:
            fused = self._fused_backend()
            if fused is not None:
                return fused(z, self.gamma, axes=axes, eps=self.eps)
        rho = rms(z, axes, self.eps)
        t = self.gamma * z / rho
        return z * type(self)._gate_python(t)

    def extra_repr(self) -> str:
        return (
            f"rms_mode={self.rms_mode!r}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e}"
        )

    # Checkpoint compatibility ─────────────────────────────────────

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Older checkpoints saved γ as a 0-dim scalar; reshape to (1,).
        key = prefix + "gamma"
        t = state_dict.get(key)
        if isinstance(t, torch.Tensor) and t.ndim == 0:
            state_dict[key] = t.reshape(1)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


# ── Functional interface (γ-free, for quick tests) ────────────────────────


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def gate_norm(
    z: torch.Tensor,
    gate_fn: Callable[[torch.Tensor], torch.Tensor],
    gamma: float | torch.Tensor = 1.0,
    *,
    rms_mode: RmsMode | DimsLike = "per_token",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional form of Gate Normalization.

    Provided primarily for unit tests and experiments; production code should
    use the `GateNorm` module so γ is a learnable parameter.
    """
    axes = rms_axes(z.ndim, rms_mode)
    rho = rms(z, axes, eps)
    if isinstance(gamma, torch.Tensor):
        if gamma.numel() != 1:
            raise ValueError(
                f"gamma must be a scalar, got shape {tuple(gamma.shape)}"
            )
        gamma = gamma.reshape(())
    return z * gate_fn(gamma * z / rho)
