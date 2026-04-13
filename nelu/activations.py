"""RMS-gate-normalized activations.

A single one-line fix — divide the gate argument by rms(z) — restores
exact scale invariance for any self-gated activation. We instantiate
two members of the family:

    NELU(z)_i = z_i * Phi(z_i / rho)          GELU variant
    NiLU(z)_i = z_i * sigma(z_i / rho)        SiLU (Swish) variant
    (NiGLU for GLU blocks lives in nelu/glu.py)

where `rho = rms(z)` with gradient flowing through (no stop-grad).

All satisfy f(alpha z) = alpha f(z) exactly in forward. Backward
carries an O(1/N) cross-term that provides mild self-normalizing
feedback during training.

RMS reduction axis:
    2D/3D  (*, d)        ->  dim = -1        (feature axis)
    4D     (B, C, H, W)  ->  dim = (1,2,3)   (all but batch)

For CNN workloads, pair with torch.compile for best performance.
For Transformer workloads, use NELUCUDA (fused SRAM-cached kernel).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _rms(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample RMS over the feature axes (dim=(1,2,3) for 4D, else -1)."""
    dim = (1, 2, 3) if z.dim() == 4 else -1
    return z.pow(2).mean(dim=dim, keepdim=True).add(eps).sqrt()


class NELU(nn.Module):
    """Normalized Gaussian Error Linear Unit — GELU + RMS gate normalization.

    Drop-in replacement for nn.GELU(). No learnable parameters.
    At rms(z)=1, NELU reduces exactly to GELU.

        NELU(z)_i = z_i * Phi(z_i / rms(z))

    Args:
        eps: Small constant added inside sqrt for numerical stability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rho = _rms(z, self.eps)
        return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class NiLU(nn.Module):
    """Normalized SiLU — SiLU/Swish + RMS gate normalization.

    Drop-in replacement for nn.SiLU(). No learnable parameters.
    At rms(z)=1, NiLU reduces exactly to SiLU.

        NiLU(z)_i = z_i * sigmoid(z_i / rms(z))

    Args:
        eps: Small constant added inside sqrt for numerical stability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rho = _rms(z, self.eps)
        return z * torch.sigmoid(z / rho)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


# ── Functional interfaces ─────────────────────────────────────────

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
