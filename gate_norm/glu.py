"""Gated Linear Unit variants with Gate Normalization on the gate branch.

Standard SwiGLU (used in LLaMA, Mistral, …) computes::

    y = W_down( silu(W_gate(x)) · W_up(x) )

We normalize the gate branch the same way as the pointwise activations::

    NiLUGLU(x) = W_down( g · σ(γ · g / rms(g)) · W_up(x) )
    NELUGLU(x) = W_down( g · Φ(γ · g / rms(g)) · W_up(x) )

where ``g = W_gate(x)``. Parameter count matches SwiGLU exactly — only the
gate's activation changes. The up branch is untouched. ``γ`` is a buffer
driven by the trainer's warmup scheduler (``GammaWarmup``).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import _DEFAULT_GAMMA_INIT, _INV_SQRT2
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
    """Shared plumbing for NiLUGLU / NELUGLU."""

    _gate_fn: Callable[[torch.Tensor], torch.Tensor]

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
        # γ_raw is learnable; γ_eff = softplus(γ_raw) is what the gate sees.
        # gamma_init sets γ_raw directly (no inv_softplus). See core.py.
        self.gamma_raw = nn.Parameter(
            torch.full((1,), float(gamma_init), dtype=torch.float32),
            requires_grad=True,
        )
        self._gate_norm_module = True

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)
        u = self.w_up(x)
        axes = (g.ndim - 1,)
        rsigma = layer_stats(g, axes, self.eps)
        g32 = g.float() if g.dtype != torch.float32 else g
        gate = type(self)._gate_fn(self.gamma * g32 * rsigma)
        h = g * gate.to(g.dtype)
        return self.w_down(h * u)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, eps={self.eps}, "
            f"gamma={self.gamma.item():.3e} (raw={self.gamma_raw.item():.3e})"
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Back-compat: convert legacy `gamma` (effective) → `gamma_raw`.
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


class NiLUGLU(_GatedGLU):
    """SwiGLU with Gate Normalization on the gate branch."""

    @staticmethod
    def _gate_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)


class NELUGLU(_GatedGLU):
    """SwiGLU with Gate Normalization + Gaussian-CDF gate."""

    @staticmethod
    def _gate_fn(t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))
