"""RMS-gate-normalized activations with a SCHEDULED (non-learnable) gamma.

    NELU(z) = z * Phi(gamma * z / rho)
    NiLU(z) = z * sigma(gamma * z / rho)

where rho = rms(z) over the last (channel) dim and gamma is a scalar
non-persistent buffer that the training loop updates each epoch via
`set_gamma_all(model, value)` on a cosine warmup schedule matching the
architecture's LR warmup.

Why buffer, not learnable Parameter
-----------------------------------
CIFAR-100 MobileNetV2 ablation (experiments/ablation_gamma_mode.py,
results on 2026-04-14):

    mode                   best   train_loss  gamma behaviour
    nelu_sched (fixed=1)   72.67  0.030       deterministic cosine
    nelu_pl (learnable)    72.62  0.024       γ ∈ [-1.4, 1.3], mean -0.06
    nelu_pc (per-channel)  69.90  0.322       collapsed near 0, 13× higher loss
    nelu_schedlearn_pl     72.12  0.027       γ_l drifted 1.0 → 0.5
    nelu_schedlearn_pc     71.44  0.140       γ_l collapsed

Key findings:
  1. learnable γ (per-layer) provides ZERO benefit over fixed γ=1
     (72.62 vs 72.67, within noise std 0.20). Matches Swish paper's
     finding that learnable β barely beats β=1.
  2. learnable γ (per-channel) CATASTROPHICALLY fails on CIFAR SGD
     recipes — γ stuck near init, model underfits by ~2.7%p.
  3. schedule * learnable hybrids lose the stability of schedule AND
     inherit the optimization risks of learnable, performing worst.
  4. The simplest variant (pure schedule, γ = buffer) wins by a hair
     and is the cleanest paper narrative: NELU adds zero parameters
     over GELU and has one deterministic hyperparameter (γ warmup
     length, which is set equal to the LR warmup length).

Gamma schedule
--------------
  gamma(epoch) = cosine warmup from g_start (default 1e-4) to g_end
                 (default 1.0) over warmup_epochs, then HOLD at g_end.

  warmup_epochs should match each architecture's LR warmup (pass it
  explicitly in the training loop — e.g. 20 for ConvNeXt-T, 5 for DeiT-III).

Training-loop API
-----------------
  from nelu import set_gamma_all, gamma_schedule

  for epoch in range(total_epochs):
      g = gamma_schedule(epoch, warmup_epochs=args.gamma_warmup_epochs,
                         g_start=args.gamma_start, g_end=args.gamma_end,
                         curve=args.gamma_curve)
      set_gamma_all(model, g)
      train_one_epoch(...)

Channel axis is always the LAST dim. Works for (B, D), (B, L, D),
(B, H, W, D). ConvNeXt Block permutes to NHWC before calling.
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
    """Base class with a non-learnable scalar gamma as a non-persistent buffer.

    The training loop calls `set_gamma_all(model, value)` each epoch to
    update `self.gamma` on a schedule. Zero parameters added to the
    model (γ is NOT in state_dict), so resume from a GELU checkpoint
    works without key mismatches.

    Accepts optional `num_channels` / `gamma_init` for API compatibility.
    """

    def __init__(self, num_channels: int = None,
                 eps: float = 1e-6,
                 gamma_init: float = _DEFAULT_GAMMA_INIT):
        super().__init__()
        del num_channels  # unused — γ is a scalar buffer
        self.eps = eps
        self.register_buffer(
            "gamma",
            torch.tensor(float(gamma_init), dtype=torch.float32),
            persistent=False,
        )

    def _expand_for_kernel(self, z: torch.Tensor) -> torch.Tensor:
        """The CUDA kernel expects a length-C gamma vector. Stride-0
        view of the scalar works since all channels share the same γ."""
        return self.gamma.expand(z.size(-1))

    def extra_repr(self) -> str:
        return f"eps={self.eps}, gamma={self.gamma.item():.6f} [scheduled]"


class NELU(_GatedBase):
    """f(z) = z * Phi(gamma * z / rms(z)),  gamma = buffer (set by training loop)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NELU_CUDA_FN:
                return _NELU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nelu_py(z, self.gamma, self.eps)


class NiLU(_GatedBase):
    """f(z) = z * sigma(gamma * z / rms(z)),  gamma = buffer (set by training loop)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.is_cuda:
            _try_load_cuda_backends()
            if _NILU_CUDA_FN:
                return _NILU_CUDA_FN(z, self._expand_for_kernel(z), self.eps)
        return _nilu_py(z, self.gamma, self.eps)


# ── Training-loop helpers ──────────────────────────────────────

def set_gamma_all(model: nn.Module, value: float) -> int:
    """Set the γ buffer on every NELU / NiLU module in `model` to `value`.
    Returns the number of modules updated. Works on DDP/compile-wrapped
    models (the buffer is per-rank, no broadcast needed)."""
    n = 0
    for m in model.modules():
        if isinstance(m, _GatedBase):
            with torch.no_grad():
                m.gamma.fill_(float(value))
            n += 1
    return n


def gamma_schedule(
    epoch: float,
    warmup_epochs: int,
    g_start: float = _DEFAULT_GAMMA_INIT,
    g_end: float = 1.0,
    curve: str = "cosine",
) -> float:
    """γ warmup schedule: rises from g_start to g_end over `warmup_epochs`,
    then HOLDS at g_end.

    Philosophy mirrors LR warmup: γ=1 is the target but starting from
    γ=1 on epoch 0 is unstable (wights are fresh, gradient large), so
    we ramp up over a short early window and then use full γ for the
    bulk of training. Set `warmup_epochs` equal to the architecture's
    LR warmup:
        ConvNeXt-T (300 ep):  20
        DeiT-III B  (800 ep):  5
        EffNet-B2  (450 ep):   3
        Swin       (300 ep):  20
    """
    if epoch >= warmup_epochs:
        return float(g_end)
    t = epoch / max(1, warmup_epochs)
    if curve == "cosine":
        return float(g_start + (g_end - g_start) * 0.5 * (1 - math.cos(math.pi * t)))
    elif curve == "linear":
        return float(g_start + (g_end - g_start) * t)
    elif curve == "exp":
        # log-uniform between g_start and g_end
        if g_start <= 0 or g_end <= 0:
            raise ValueError("exp curve requires g_start, g_end > 0")
        return float(g_start * (g_end / g_start) ** t)
    else:
        raise ValueError(f"unknown curve: {curve!r}")


def current_gamma(model: nn.Module) -> float:
    """Return the current γ value (any NELU/NiLU module; they're all the
    same after a `set_gamma_all` call). Returns nan if no module found."""
    for m in model.modules():
        if isinstance(m, _GatedBase):
            return float(m.gamma.item())
    return float("nan")


# ── Functional interfaces (fixed gamma=1, for offline test only) ──

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
