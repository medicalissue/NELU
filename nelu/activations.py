"""RMS-gate-normalized activations with a learnable per-layer scalar γ.

    NELU(z) = z * Phi(gamma * z / rho)
    NiLU(z) = z * sigma(gamma * z / rho)

where rho = rms(z) over the last (channel) dim and gamma is a single
learnable scalar nn.Parameter per activation module, initialized to
1e-6 (near-linear start, matching LayerScale convention). Training grows gamma as the optimization
landscape dictates.

Design choice rationale
-----------------------
On CIFAR-100 MobileNetV2 the ablation (experiments/ablation_gamma_mode.py)
showed per-layer learnable γ and fixed γ=1 with schedule warmup are
statistically tied (72.62 vs 72.67). On ImageNet ConvNeXt-T the
per-layer learnable variant (y61na0ma) showed a clearly faster early
trajectory than per-channel (164tlkm0). Combined with the simpler
optimization story (one learnable scalar per activation, no
additional hyperparameters beyond its init), per-layer learnable is
the chosen final formulation.

The 72% negative γ ConvNeXt observation is a valid "complementary
activation" reparameterization, not a bug: f_{-γ}(z) = z − f_γ(z) is
mathematically absorbable by a per-layer Linear sign flip. Per-layer
sign ambiguity is harmless; per-channel was the problematic case
(which we no longer use).

Channel axis is always the LAST dim. Works for (B, D), (B, L, D),
(B, H, W, D). ConvNeXt Block permutes to NHWC before calling.
"""

import math

import torch
import torch.nn as nn


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_DEFAULT_GAMMA_INIT = 1e-6


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


# ── Python reference ────────────────────────────────────────────

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
    """Base class with a single learnable scalar γ (direct nn.Parameter).

    Accepts optional `num_channels` for API compatibility — ignored
    (γ is per-module scalar regardless).
    """

    def __init__(self, num_channels: int = None,
                 eps: float = 1e-6,
                 gamma_init: float = _DEFAULT_GAMMA_INIT):
        super().__init__()
        del num_channels  # unused
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init),
                                               dtype=torch.float32))

    def _expand_for_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """The CUDA kernel expects a length-C gamma vector.
        Broadcast the scalar to (C,) via a stride-0 view; autograd
        sums the (C,) dgamma back to a scalar grad on self.gamma."""
        return self.gamma.expand(z.size(-1))

    def extra_repr(self) -> str:
        return f"eps={self.eps}, gamma={self.gamma.item():.6f}"


class NELU(_GatedBase):
    """f(z) = z * Phi(gamma * z / rms(z)),  gamma = nn.Parameter(init 1e-6)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NELU_CUDA_FN:
                return _NELU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nelu_py(z, self.gamma, self.eps)


class NiLU(_GatedBase):
    """f(z) = z * sigma(gamma * z / rms(z)),  gamma = nn.Parameter(init 1e-6)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NILU_CUDA_FN:
                return _NILU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nilu_py(z, self.gamma, self.eps)


# ── wandb logging helper ───────────────────────────────────────

def collect_gamma_stats(model: nn.Module):
    """Walk `model` and return a dict mapping scalar names to γ values,
    suitable for `wandb.log(...)`.

    Produces:
      nelu/gamma/layer_{i}   — per-module γ
      nelu/gamma/mean, min, max, std, abs_mean, n_negative — aggregates

    Caller decides when to log (typically epoch end).
    """
    gammas = []
    out = {}
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, _GatedBase):
            g = m.gamma.detach().float().item()
            gammas.append(g)
            out[f"nelu/gamma/layer_{layer_idx}"] = g
            layer_idx += 1
    if not gammas:
        return {}
    n = len(gammas)
    mean = sum(gammas) / n
    var = sum((x - mean) ** 2 for x in gammas) / max(1, n - 1)
    out["nelu/gamma/mean"] = mean
    out["nelu/gamma/min"] = min(gammas)
    out["nelu/gamma/max"] = max(gammas)
    out["nelu/gamma/std"] = var ** 0.5
    out["nelu/gamma/abs_mean"] = sum(abs(x) for x in gammas) / n
    out["nelu/gamma/n_negative"] = float(sum(1 for x in gammas if x < 0))
    out["nelu/gamma/n_modules"] = float(n)
    return out


# ── Functional interfaces (fixed gamma=1, for offline test only) ──

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
