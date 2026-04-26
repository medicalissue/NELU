"""Gate Normalization — the core building block.

Gate Normalization (GN) generalizes self-gated activations of the form
``x · g(x)`` by normalizing the gate input by its root-mean-square:

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise squashing function (sigmoid, Gaussian CDF, …)
and ``γ`` is a single learnable scalar shared per module. The optimizer
drives ``γ`` together with the rest of the model; in practice gradient
flow keeps it positive — anti-gating (``γ < 0``) inverts the gate and
flips the "keep positives, drop negatives" inductive bias, which yields
no useful loss signal — so no explicit positivity constraint is needed.

The forward pass is plain PyTorch so it composes cleanly with
torch.compile, AMP autocast, and arbitrary reduction axes (including
non-contiguous subsets such as ``(2, 3)`` used after depthwise
convolutions). Statistics are upcast to float32 internally to avoid
fp16/bf16 underflow.

Subclasses implement :meth:`_gate_python` to select the pointwise gate
function.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from .dispatch import should_use_cuda
from .layout import DimsLike, NormAxes, resolve_axes
from .stats import layer_stats


_DEFAULT_GAMMA_INIT = 1.0
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class GateNorm(nn.Module):
    """Scale-invariant self-gated activation with a single learnable γ.

    Parameters
    ----------
    norm_axes : str or tuple of int, default ``"channel"``
        Axes over which the RMS is computed.
    eps : float, default ``1e-6``
        Numerical floor added inside the RMS before sqrt.
    gamma_init : float, default ``1.0``
        Initial value of the learnable γ at step 0.

    Subclasses provide ``_gate_python`` (the pure-PyTorch reference
    implementation, used on CPU / MPS / autocast-disabled paths) and
    optionally ``_CUDA_KIND`` — the integer enum understood by the fused
    CUDA backend (``0`` for NELU/Φ, ``1`` for NiLU/σ). When set, and the
    runtime conditions in :func:`gate_norm.dispatch.should_use_cuda` are
    met, the fused kernel takes the call. Subclasses that don't set
    ``_CUDA_KIND`` (e.g. user-defined gates) stay on the PyTorch path.
    """

    # CUDA gate kind for this module. None disables the fused path; the
    # forward then falls back to ``_gate_python``. Subclasses override.
    _CUDA_KIND: int | None = None

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
        self.gamma = nn.Parameter(
            torch.full((1,), float(gamma_init), dtype=torch.float32)
        )
        self._gate_norm_module = True

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _axes(self, z: torch.Tensor) -> tuple[int, ...]:
        return resolve_axes(z.ndim, self.norm_axes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        axes = self._axes(z)
        if self._CUDA_KIND is not None and should_use_cuda(z):
            # Fused kernel path: imports lazily so CUDA-less environments
            # never reach the cpp_extension build.
            from . import cuda as _cuda
            return _cuda.gate_norm_cuda_forward(
                z, self.gamma, self._CUDA_KIND, axes, float(self.eps),
            )
        # Pure PyTorch fallback.
        rsigma = layer_stats(z, axes, self.eps)
        z32 = z.float() if z.dtype != torch.float32 else z
        gate = type(self)._gate_python(self.gamma * z32 * rsigma)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        return (
            f"norm_axes={self.norm_axes!r}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e}"
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
