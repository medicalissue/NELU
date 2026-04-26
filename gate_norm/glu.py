"""Gated Linear Unit variants with Gate Normalization on the gate branch.

Standard SwiGLU (used in LLaMA, Mistral, …) computes::

    y = W_down( silu(W_gate(x)) · W_up(x) )

We normalize the gate branch the same way as the pointwise activations::

    NiLUGLU(x) = W_down( g · σ(γ · g / rms(g)) · W_up(x) )
    NELUGLU(x) = W_down( g · Φ(γ · g / rms(g)) · W_up(x) )

where ``g = W_gate(x)``. Parameter count matches SwiGLU exactly (γ is a
single scalar per module). The up branch is untouched.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import _DEFAULT_GAMMA_INIT, _INV_SQRT2
from .dispatch import should_use_cuda
from .stats import layer_stats


def _llama_hidden(dim: int) -> int:
    """LLaMA's FFN width: round 8/3 · dim up to a multiple of 256."""
    h = int(dim * 8 / 3)
    return (h + 255) // 256 * 256


class SwiGLU(nn.Module):
    """Standard SwiGLU FFN — provided as a baseline reference."""

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or _llama_hidden(dim)
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class _GatedGLU(nn.Module):
    """Shared plumbing for NiLUGLU / NELUGLU.

    Subclasses set ``_CUDA_KIND`` (matches :class:`gate_norm.GateNorm`)
    to opt into the fused CUDA path on the gate branch.
    """

    _gate_fn: Callable[[torch.Tensor], torch.Tensor]
    _CUDA_KIND: int | None = None

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
        *,
        eps: float = 1e-6,
        gamma_init: float = _DEFAULT_GAMMA_INIT,
    ):
        super().__init__()
        hidden_dim = hidden_dim or _llama_hidden(dim)
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.eps = eps
        self.gamma = nn.Parameter(
            torch.full((1,), float(gamma_init), dtype=torch.float32)
        )
        self._gate_norm_module = True

    def _gated(self, g: torch.Tensor) -> torch.Tensor:
        """Compute ``g · gate(γ · g / rms(g))`` over the trailing axis."""
        axes = (g.ndim - 1,)
        if self._CUDA_KIND is not None and should_use_cuda(g):
            from . import cuda as _cuda
            return _cuda.gate_norm_cuda_forward(
                g, self.gamma, self._CUDA_KIND, axes, float(self.eps),
            )
        rsigma = layer_stats(g, axes, self.eps)
        g32 = g.float() if g.dtype != torch.float32 else g
        gate = type(self)._gate_fn(self.gamma * g32 * rsigma)
        return g * gate.to(g.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)
        u = self.w_up(x)
        h = self._gated(g)
        return self.w_down(h * u)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e}"
        )


class NiLUGLU(_GatedGLU):
    """SwiGLU with Gate Normalization on the gate branch."""

    _CUDA_KIND = 1  # GATE_SIGMOID

    @staticmethod
    def _gate_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)


class NELUGLU(_GatedGLU):
    """SwiGLU with Gate Normalization + Gaussian-CDF gate."""

    _CUDA_KIND = 0  # GATE_PHI

    @staticmethod
    def _gate_fn(t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))
