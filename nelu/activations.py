"""RMS-gate-normalized activations with a learnable per-activation gamma.

    NELU(z)_i = z_i * Phi(gamma * z_i / rho)
    NiLU(z)_i = z_i * sigma(gamma * z_i / rho)

where rho = rms(z) (over the last / feature dim) and gamma is a SINGLE
scalar learnable Parameter per module instance ("per-activation" — one
gamma per NELU module, NOT per channel). Initialized to a small value
so the module starts as approximately 0.5*z (near-linear). Training
grows gamma per layer to find the right gate sharpness.

Homogeneity is preserved because rho(alpha*z) = alpha*rho(z):
    f(alpha*z) = alpha*z * g(gamma*alpha*z/(alpha*rho))
               = alpha*z * g(gamma*z/rho)
               = alpha * f(z)

RMS / channel axis:
    Channel is always the LAST dim. Works for:
        (B, D)         — MLP head
        (B, L, D)      — ViT/DeiT tokens
        (B, H, W, D)   — ConvNeXt Block (NHWC inside the block)
    rho is computed over dim=-1.
    (NCHW is NOT supported — this repo always permutes to NHWC before
    the activation call.)

Backend: if the CUDA extension compiles, NELU/NiLU delegate to the
fused kernel. The CUDA kernel is written for a length-C gamma vector
so we broadcast the scalar gamma to shape (C,) via `.expand(C)` before
passing it in. autograd sums the (C,) dgamma back to a scalar gradient
for the underlying Parameter.
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


# ── Python reference (scalar gamma) ─────────────────────────────

def _nelu_py(z: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rho = _rms(z, eps)
    t = gamma * z / rho
    return z * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))


def _nilu_py(z: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rho = _rms(z, eps)
    t = gamma * z / rho
    return z * torch.sigmoid(t)


# ── Modules ────────────────────────────────────────────────────

class _GatedBase(nn.Module):
    """Base for activations with a single learnable scalar gamma.

    Accepts an optional `num_channels` argument for API compatibility
    with the previous per-channel implementation — it is accepted and
    ignored (gamma is always a scalar).
    """

    def __init__(self, num_channels: int = None,
                 eps: float = 1e-6,
                 gamma_init: float = _DEFAULT_GAMMA_INIT):
        super().__init__()
        del num_channels  # unused — gamma is scalar
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init),
                                               dtype=torch.float32))

    def _expand_for_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """The CUDA kernel expects a length-C gamma vector.
        Broadcast the scalar to (C,) via a stride-0 view; autograd
        will sum the resulting (C,) dgamma back to a scalar grad."""
        return self.gamma.expand(z.size(-1))

    def extra_repr(self) -> str:
        return f"eps={self.eps}, gamma={self.gamma.item():.6f}"


class NELU(_GatedBase):
    """NELU with a single learnable scalar gamma per module.

        f(z) = z * Phi(gamma * z / rho)
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NELU_CUDA_FN:
                return _NELU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nelu_py(z, self.gamma, self.eps)


class NiLU(_GatedBase):
    """NiLU with a single learnable scalar gamma per module.

        f(z) = z * sigma(gamma * z / rho)
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NILU_CUDA_FN:
                return _NILU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nilu_py(z, self.gamma, self.eps)


# ── Functional interfaces (gamma=1, no learnable — for testing only) ──

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
