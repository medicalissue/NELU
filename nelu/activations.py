"""RMS-gate-normalized activations with a learnable per-channel gamma.

    NELU(z)_i = z_i * Phi(gamma_c * z_i / rho)
    NiLU(z)_i = z_i * sigma(gamma_c * z_i / rho)

where rho = rms(z) (over feature axes, see below) and gamma_c is a
per-channel learnable scalar (one per feature dim). gamma is initialized
to a small value so the module starts as approximately 0.5*z
(near-linear). Training grows each gamma_c to find the right gate
sharpness per channel.

Homogeneity is preserved because rho(alpha*z) = alpha*rho(z):
    f(alpha*z) = alpha*z * g(gamma * alpha*z / (alpha*rho))
               = alpha*z * g(gamma*z/rho)
               = alpha * f(z)

RMS reduction axis and gamma axis:
    Channel is always the LAST dim. Works for:
        (B, D)         — MLP head
        (B, L, D)      — ViT/DeiT tokens
        (B, H, W, D)   — ConvNeXt Block (NHWC inside the block)
    rho is computed over dim=-1; gamma has shape (D,) broadcast on -1.
    (NCHW is NOT supported — the repo never calls this class in that
    layout; ConvNeXt permutes to NHWC before the activation.)

gamma is lazily materialized on first forward so callers do not need
to specify a channel count at construction time.

Backend: if the CUDA extension compiles AND the input is 2D/3D, NELU/NiLU
delegate to the fused kernel. Otherwise they use a pure-PyTorch path.
(4D is handled in Python — the CUDA kernel currently supports last-dim
gamma only.)
"""

import math

import torch
import torch.nn as nn


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_DEFAULT_GAMMA_INIT = 1e-4


# ── CUDA backend detection (lazy) ───────────────────────────────

_NELU_CUDA_FN = None
_NILU_CUDA_FN = None

def _try_load_cuda_backends():
    global _NELU_CUDA_FN, _NILU_CUDA_FN
    if _NELU_CUDA_FN is not None and _NILU_CUDA_FN is not None:
        return
    try:
        from .cuda_kernel import nelu_cuda as _nelu_cu
        from .nilu_cuda_kernel import nilu_cuda as _nilu_cu
        _NELU_CUDA_FN = _nelu_cu
        _NILU_CUDA_FN = _nilu_cu
    except Exception:
        _NELU_CUDA_FN = False
        _NILU_CUDA_FN = False


def _rms(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token RMS over the last (channel) dim."""
    return z.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()


def _gamma_view(gamma: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Reshape (C,) gamma so it broadcasts over z on the last dim."""
    shape = [1] * z.dim()
    shape[-1] = gamma.numel()
    return gamma.view(*shape)


# ── Python reference ────────────────────────────────────────────

def _nelu_py(z: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rho = _rms(z, eps)
    g = _gamma_view(gamma, z)
    t = g * z / rho
    return z * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


def _nilu_py(z: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rho = _rms(z, eps)
    g = _gamma_view(gamma, z)
    t = g * z / rho
    return z * torch.sigmoid(t)


# ── Modules ────────────────────────────────────────────────────

class _GatedBase(nn.Module):
    """Base for activations with a per-channel learnable gamma."""

    def __init__(self, num_channels: int, eps: float = 1e-6,
                 gamma_init: float = _DEFAULT_GAMMA_INIT):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.full((int(num_channels),),
                                             float(gamma_init),
                                             dtype=torch.float32))

    def extra_repr(self) -> str:
        return (f"eps={self.eps}, C={self.gamma.numel()}, "
                f"gamma_mean={self.gamma.mean().item():.6f}")


class NELU(_GatedBase):
    """NELU with learnable per-channel gamma.

        f(z)_{...,c,...} = z * Phi(gamma_c * z / rho)
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NELU_CUDA_FN:
                return _NELU_CUDA_FN(z, self.gamma, self.eps)
        return _nelu_py(z, self.gamma, self.eps)


class NiLU(_GatedBase):
    """NiLU with learnable per-channel gamma.

        f(z)_{...,c,...} = z * sigma(gamma_c * z / rho)
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NILU_CUDA_FN:
                return _NILU_CUDA_FN(z, self.gamma, self.eps)
        return _nilu_py(z, self.gamma, self.eps)


# ── Functional interfaces (gamma=1, no learnable — for testing only) ──

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
