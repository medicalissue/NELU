"""Gate Normalization — the core building block.

Gate Normalization (GN) generalizes self-gated activations of the form
``x · g(x)`` by normalizing the gate input by its root-mean-square:

    y = x · g(γ · x / rms(x))

where ``g`` is a pointwise squashing function (sigmoid, Gaussian CDF, …)
and ``γ`` is a non-learnable scalar buffer scheduled by the trainer
(typically warmed up from 0 → 1 alongside the LR warmup, then held at 1
for the rest of training). The outer multiplication by ``x`` preserves
the feature's DC component so the activation retains the "deactivate the
negative side" inductive bias of ReLU/GELU/SiLU.

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
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class GateNorm(nn.Module):
    """Scale-invariant self-gated activation with a scheduled scalar γ.

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
    gamma_init : float, default ``0.0``
        Initial value of γ. The trainer's warmup schedule is responsible
        for ramping γ to its final value (typically 1.0). At γ = 0 the
        module reduces to ``y = x · g(0) = 0.5 · x`` (linear identity-ish).
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
    ) -> None:
        super().__init__()
        self.norm_axes = norm_axes
        self.eps = eps
        # γ is a non-learnable scalar driven externally by a warmup
        # scheduler. It rides on the module as a buffer so it ships with
        # state_dict and DDP broadcasts; gradients are not computed.
        self.register_buffer(
            "gamma",
            torch.full((1,), float(gamma_init), dtype=torch.float32),
        )
        # Marker so `collect_gamma_stats` can discover us generically.
        self._gate_norm_module = True

    # Subclass hook ─────────────────────────────────────────────────

    @staticmethod
    def _gate_python(t: torch.Tensor) -> torch.Tensor:
        """Pointwise squashing function applied to ``γ · x / rms(x)``."""
        raise NotImplementedError

    # Forward ───────────────────────────────────────────────────────

    def _axes(self, z: torch.Tensor) -> tuple[int, ...]:
        return resolve_axes(z.ndim, self.norm_axes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        axes = self._axes(z)
        if self._CUDA_OP is not None and should_use_cuda(z):
            from . import cuda  # Lazy: only imported on CUDA devices.
            return cuda.fused_forward(
                self._CUDA_OP, z, self.gamma, axes, self.eps,
            )

        rsigma = layer_stats(z, axes, self.eps)
        # The outer multiplication keeps the caller's dtype so the AMP
        # autocast contract is preserved; the gate path stays in float32.
        z32 = z.float() if z.dtype != torch.float32 else z
        t = self.gamma * z32 * rsigma
        gate = type(self)._gate_python(t)
        return z * gate.to(z.dtype)

    def extra_repr(self) -> str:
        return (
            f"norm_axes={self.norm_axes!r}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e}"
        )

    # Checkpoint compatibility ─────────────────────────────────────

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Older centred-and-learnable checkpoints saved γ as a Parameter
        # (1-D) and a separate β scalar. We load γ if present, drop β.
        gamma_key = prefix + "gamma"
        t = state_dict.get(gamma_key)
        if isinstance(t, torch.Tensor) and t.ndim == 0:
            state_dict[gamma_key] = t.reshape(1)
        # Discard any β key so strict loading still succeeds; the new
        # form has no β.
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

    Provided primarily for unit tests and experiments; production code should
    use :class:`GateNorm` so ``γ`` rides through state_dict and the trainer's
    warmup scheduler.
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
