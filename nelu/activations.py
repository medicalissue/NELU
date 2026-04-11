"""Pure PyTorch implementation of NELU.

    NELU(z)_i = z_i * Phi(z_i / rms(z))

The gate sees only relative magnitudes; absolute scale flows through the
output. The forward is exactly scale-invariant: NELU(alpha z) = alpha NELU(z).
Gradient flows through rms (no stop-gradient) — an O(1/N) cross-term
provides self-normalizing feedback that empirically improves training
stability on CNNs.

RMS reduction axis:
    2D/3D  (*, d)        ->  dim = -1        (feature axis)
    4D     (B, C, H, W)  ->  dim = (1,2,3)   (all but batch)

For CNN workloads, pair with torch.compile for best performance.
For Transformer workloads, use NELUCUDA (fused SRAM-cached kernel).
"""

import math

import torch
import torch.nn as nn


class NELU(nn.Module):
    """Normalized Gaussian Error Linear Unit.

    Drop-in replacement for nn.GELU(). No learnable parameters.
    At rms(z)=1, NELU reduces exactly to GELU.

    Args:
        eps: Small constant added inside sqrt for numerical stability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dim = (1, 2, 3) if z.dim() == 4 else -1
        rms = z.pow(2).mean(dim=dim, keepdim=True).add(self.eps).sqrt()
        # No stop-gradient: rms participates in backward via autograd.
        z_hat = z / rms
        return z * 0.5 * (1.0 + torch.erf(z_hat * _INV_SQRT2))

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional interface for NELU."""
    dim = (1, 2, 3) if z.dim() == 4 else -1
    rms = z.pow(2).mean(dim=dim, keepdim=True).add(eps).sqrt()
    z_hat = z / rms
    return z * 0.5 * (1.0 + torch.erf(z_hat * _INV_SQRT2))


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
