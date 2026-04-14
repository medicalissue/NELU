"""RMS-gate-normalized activations with a softplus-reparameterized
learnable scalar gamma.

    NELU(z) = z * Phi(gamma * z / rho)
    NiLU(z) = z * sigma(gamma * z / rho)

where rho = rms(z) over the last (channel) dim and
    gamma = softplus(raw_gamma),   raw_gamma ∈ nn.Parameter

Why this parameterization
-------------------------
Earlier iterations tried:

  * fixed gamma = 1 → training diverges in early epochs (full-sharpness
    NELU has the same gradient explosion characteristics as
    non-warmed-up ReLU under AdamW + LR=4e-3);
  * unconstrained learnable gamma (scalar) → converges to the
    "complementary activation" local minimum, where 72% of layers end up
    with gamma < 0 because f_{-g}(z) = z - f_g(z) is an algebraic
    reparameterization that the optimizer can slide into by flipping
    pwconv1 weights/bias while leaving pwconv2 mostly unchanged;
  * per-channel learnable gamma → same sign-flip problem plus a 50/50
    split between pos/neg channels, wasting 26k parameters to encode
    ~1 effective degree of freedom per layer.

Softplus reparameterization solves both failure modes at once:

  * gamma is always > 0 → the sign-flip / complementary-activation
    local minimum is unreachable;
  * raw_gamma starts near -9.21 (so softplus(raw_gamma) ≈ 1e-4), which
    gives a near-linear activation at init time — the empirical fix
    for the fixed-γ=1 early-epoch instability;
  * the network still learns its own per-layer sharpness (we observed
    meaningful stage-wise variation 0.7–2.9 in the prior run, which
    shouldn't be thrown away);
  * softplus avoids the kink that abs/squared reparameterizations
    create at zero — a kink acts as an attractor for unstable
    optimizers and is mechanistically what produced the sign-flip
    trap we observed.

References
----------
  * ACON (arxiv:2009.04759) — bounded β (∈(0,1)) outperforms
    unconstrained β by ~1% on ImageNet; authors note unconstrained β
    occasionally goes negative and underperforms. Direct analog of our
    sign-flip observation.
  * Dynamic ReLU (arxiv:2003.10027) — bounded slopes essential;
    unbounded diverges.
  * Real NVP / Glow (arxiv:1605.08803, 1807.03039) — softplus is the
    recommended stable positive-scalar parameterization in the
    normalizing-flow literature.

Channel axis is always the LAST dim. Works for (B, D), (B, L, D),
(B, H, W, D). ConvNeXt Block permutes to NHWC before calling, so it's
still last-dim-feature.

The CUDA kernel takes a length-C gamma vector; we pass
`gamma.expand(C)` — a stride-0 view, materialized to a real tensor
inside the custom_op wrapper. autograd sums the (C,) dgamma back to a
scalar grad on raw_gamma via softplus' backward and expand's backward.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Base class with a softplus-reparameterized positive learnable gamma."""

    def __init__(self, num_channels: int = None,
                 eps: float = 1e-6,
                 gamma_init: float = _DEFAULT_GAMMA_INIT):
        super().__init__()
        del num_channels  # unused — gamma is a single scalar per module
        self.eps = eps
        gi = float(gamma_init)
        if gi <= 0:
            raise ValueError(f"gamma_init must be > 0, got {gi}")
        # raw_gamma chosen so softplus(raw_gamma) == gamma_init.
        # softplus(x) = log(1+exp(x));  inverse: log(expm1(y))
        # For gi=1e-4: raw ≈ -9.2103
        raw = math.log(math.expm1(gi)) if gi > 1e-6 else math.log(gi)
        self.raw_gamma = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    @property
    def gamma(self) -> torch.Tensor:
        """Effective gamma = softplus(raw_gamma), always > 0."""
        return F.softplus(self.raw_gamma)

    def _expand_for_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """Length-C stride-0 view of the scalar gamma for the CUDA kernel.
        autograd sums the (C,) dgamma back to a scalar grad on raw_gamma."""
        return self.gamma.expand(z.size(-1))

    def extra_repr(self) -> str:
        return (f"eps={self.eps}, gamma={self.gamma.item():.6f} "
                f"(raw={self.raw_gamma.item():.3f})")


class NELU(_GatedBase):
    """f(z) = z * Phi(softplus(raw_gamma) * z / rms(z))."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NELU_CUDA_FN:
                return _NELU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nelu_py(z, self.gamma, self.eps)


class NiLU(_GatedBase):
    """f(z) = z * sigma(softplus(raw_gamma) * z / rms(z))."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NILU_CUDA_FN:
                return _NILU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nilu_py(z, self.gamma, self.eps)


# ── wandb logging helper ───────────────────────────────────────

def collect_gamma_stats(model: nn.Module):
    """Walk `model` and return a dict mapping scalar names to effective
    gamma values, suitable for `wandb.log(...)`.

    Produces:
      nelu/gamma/layer_{i}   — per-module effective gamma (softplus(raw))
      nelu/gamma/raw_layer_{i} — raw_gamma (pre-softplus)
      nelu/gamma/mean, min, max, std — aggregates over all modules

    Caller decides when to log (typically epoch end) and whether to
    include an `epoch` / `step` step axis alongside.
    """
    gammas = []
    raws = []
    out = {}
    layer_idx = 0
    for m in model.modules():
        if isinstance(m, _GatedBase):
            g = m.gamma.detach().float().item()
            r = m.raw_gamma.detach().float().item()
            gammas.append(g)
            raws.append(r)
            out[f"nelu/gamma/layer_{layer_idx}"] = g
            out[f"nelu/gamma/raw_layer_{layer_idx}"] = r
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
    out["nelu/gamma/n_modules"] = float(n)
    return out


# ── Functional interfaces (fixed gamma=1, for offline test only) ──

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
