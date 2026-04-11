"""Gated Linear Unit (GLU) variants with and without RMS gate normalization.

Standard SwiGLU (used in LLaMA, Mistral, etc.):
    SwiGLU(x) = down( silu(gate_proj(x)) * up_proj(x) )

We define two RMS-gate-normalized variants — one for each pointwise base:

    NiLUGLU(x) = down( nilu(gate_proj(x)) * up_proj(x) )
              = down( g * sigmoid(g / rms(g)) * up )      # SwiGLU + RMS

    NELUGLU(x) = down( nelu(gate_proj(x)) * up_proj(x) )
              = down( g * Phi(g / rms(g))    * up )       # GELU-GLU + RMS

The `up_proj` output is left untouched — only the gate side is normalized.
Both keep parameter count identical to SwiGLU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _default_hidden(dim: int) -> int:
    """LLaMA-style FFN hidden dim: round 8/3 * dim up to a multiple of 256."""
    h = int(dim * 8 / 3)
    return (h + 255) // 256 * 256


# ── Baseline ─────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """Standard SwiGLU FFN block (LLaMA recipe).

        h = silu(x W_gate) * (x W_up)
        y = h W_down
    """

    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or _default_hidden(dim)
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ── Our variants ─────────────────────────────────────────────────

class NiLUGLU(nn.Module):
    """SwiGLU with NiLU on the gate projection (= SwiGLU + RMS gate norm).

        g = x W_gate
        h = g * sigmoid( g / rms(g) ) * (x W_up)      # NiLU(g) * up
        y = h W_down

    Parameter count identical to SwiGLU.
    """

    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        hidden_dim = hidden_dim or _default_hidden(dim)
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)
        u = self.w_up(x)
        rms = g.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        h = g * torch.sigmoid(g / rms)        # NiLU(g)
        return self.w_down(h * u)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, eps={self.eps}"


class NELUGLU(nn.Module):
    """GLU FFN block using NELU (Phi-based) on the gate projection.

        g = x W_gate
        h = g * Phi( g / rms(g) ) * (x W_up)          # NELU(g) * up
        y = h W_down

    Same parameter count as SwiGLU. Differs from NiLUGLU only in the
    nonlinearity used on the gate (Gaussian CDF vs sigmoid).
    """

    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        hidden_dim = hidden_dim or _default_hidden(dim)
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)
        u = self.w_up(x)
        rms = g.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        h = g * 0.5 * (1.0 + torch.erf((g / rms) * _INV_SQRT2))   # NELU(g)
        return self.w_down(h * u)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, eps={self.eps}"
