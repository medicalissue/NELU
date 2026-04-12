"""RMS-gate-normalized activations.

A single one-line fix — divide the gate argument by rms(z) — restores
exact scale invariance for any self-gated activation. We instantiate
three members of the family:

    NELU(z)_i = z_i * Phi(z_i / rho)          GELU variant
    NiLU(z)_i = z_i * sigma(z_i / rho)        SiLU (Swish) variant
    (NiGLU for GLU blocks lives in nelu/glu.py)

where `rho = rms(z)` with gradient flowing through (no stop-grad).

All three satisfy f(alpha z) = alpha f(z) exactly in forward. Backward
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


# ── Stop-Gradient variants ───────────────────────────────────────
#
# rho is detached so the backward has NO cross-term reduction.
# Gradient is purely element-wise: dz_j = g_j * h(t_j).
# More stable with LAMB / high-lr recipes.

class NELU_SG(nn.Module):
    """NELU with stop-gradient on rms — no cross-term in backward."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rho = _rms(z, self.eps).detach()
        return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class NiLU_SG(nn.Module):
    """NiLU with stop-gradient on rms — no cross-term in backward."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rho = _rms(z, self.eps).detach()
        return z * torch.sigmoid(z / rho)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


# ── Learnable-β variants ─────────────────────────────────────────
#
# f(z) = z · g(β · z / ρ)   where β = softplus(raw) > 0, learnable.
#
# β > 1 : sharper gate (→ ReLU-like)
# β < 1 : softer gate (→ linear)
# β = 1 : standard NELU / NiLU
#
# Uses SG on ρ for stability. β gradient flows through autograd
# (one extra scalar per module — negligible overhead).

class NELU_Beta(nn.Module):
    """NELU with learnable gate temperature β. NoSG — gradient flows through rms."""

    def __init__(self, eps: float = 1e-6, init_beta: float = 1.0):
        super().__init__()
        self.eps = eps
        raw = math.log(math.exp(init_beta) - 1.0) if init_beta > 0 else 0.0
        self._raw_beta = nn.Parameter(torch.tensor(raw))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._raw_beta)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rho = _rms(z, self.eps)
        t = self.beta * z / rho
        return z * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))

    def extra_repr(self) -> str:
        return f"eps={self.eps}, beta={self.beta.item():.4f}"


class NiLU_Beta(nn.Module):
    """NiLU with learnable gate temperature β. NoSG — gradient flows through rms."""

    def __init__(self, eps: float = 1e-6, init_beta: float = 1.0):
        super().__init__()
        self.eps = eps
        raw = math.log(math.exp(init_beta) - 1.0) if init_beta > 0 else 0.0
        self._raw_beta = nn.Parameter(torch.tensor(raw))

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._raw_beta)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rho = _rms(z, self.eps)
        t = self.beta * z / rho
        return z * torch.sigmoid(t)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, beta={self.beta.item():.4f}"


# ── Per-channel γ variants (RMSNorm-style affine gate) ───────────
#
# f(z)_i = z_i · g(γ_i · z_i / ρ)
#
# γ is a per-channel (or per-feature) learnable gain, initialized to 1.
# Preserves exact homogeneity: f(αz) = αf(z) because ρ(αz) = αρ(z),
# so γ·αz/(αρ) = γ·z/ρ — the α cancels.
#
# Lazy-initialized: γ is created on first forward() from the input shape.
# A dummy forward must be run BEFORE DDP wrap so all ranks have γ.

class NELU_Gamma(nn.Module):
    """NELU with per-channel learnable gate gain γ.

    Like RMSNorm's affine weight but inside the activation gate.
    NoSG — gradient flows through rms. γ initialized to 1.

    Forces fp32 inside AMP autocast to prevent fp16 overflow in z/rho.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = None  # lazy: materialized on first forward

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.gamma is None:
            dim = z.size(1) if z.dim() == 4 else z.size(-1)
            self.gamma = nn.Parameter(torch.ones(dim, device=z.device))
        rho = _rms(z, self.eps)
        g = self.gamma.view(1, -1, 1, 1) if z.dim() == 4 else self.gamma
        t = g * z / rho
        return z * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))

    def extra_repr(self) -> str:
        n = self.gamma.numel() if self.gamma is not None else '?'
        return f"eps={self.eps}, dim={n}"


class NiLU_Gamma(nn.Module):
    """NiLU with per-channel learnable gate gain γ. NoSG."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = None

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.gamma is None:
            dim = z.size(1) if z.dim() == 4 else z.size(-1)
            self.gamma = nn.Parameter(torch.ones(dim, device=z.device))
        rho = _rms(z, self.eps)
        g = self.gamma.view(1, -1, 1, 1) if z.dim() == 4 else self.gamma
        t = g * z / rho
        return z * torch.sigmoid(t)

    def extra_repr(self) -> str:
        n = self.gamma.numel() if self.gamma is not None else '?'
        return f"eps={self.eps}, dim={n}"

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


# ── Surrogate-backward variants ───────────────────────────────────
#
# Forward: f(z) = z · g(z/ρ)       ← scale-invariant, unchanged
# Backward: dz = g_out · h(z)      ← uses RAW z, not z̃ = z/ρ
#
# h(z) = Φ(z) + z·φ(z)   for NELU
# h(z) = σ(z) + z·σ'(z)  for NiLU
#
# Why: GELU's h(z) saturates at large |z| → {0, 1} → sparse gradient
# → implicit damping. NELU's h(z̃) never saturates because z̃ = z/ρ
# is always O(1). The surrogate re-introduces GELU's damping in
# backward only, keeping forward scale-invariant.
#
# At init (ρ ≈ 1): z ≈ z̃ → no difference.
# When ||W|| grows (ρ >> 1): h(z) saturates → gradient damping.

_INV_SQRT2PI = 0.3989422804014327  # 1/sqrt(2π)


class _NELUSurrFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, z, eps):
        dim = (1, 2, 3) if z.dim() == 4 else -1
        rho = z.pow(2).mean(dim=dim, keepdim=True).add(eps).sqrt()
        gate = 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))
        ctx.save_for_backward(z)  # only z — backward doesn't need ρ
        return z * gate

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        z, = ctx.saved_tensors
        # Surrogate: evaluate h at raw z, not z̃ = z/ρ
        phi_z = _INV_SQRT2PI * torch.exp(-0.5 * z * z)
        cdf_z = 0.5 * (1.0 + torch.erf(z * _INV_SQRT2))
        h_z = cdf_z + z * phi_z
        return grad_out * h_z, None


class NELU_Surr(nn.Module):
    """NELU with surrogate backward: forward is scale-invariant,
    backward uses h(z) instead of h(z/ρ) for GELU-like damping."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return _NELUSurrFn.apply(z, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class _NiLUSurrFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, z, eps):
        dim = (1, 2, 3) if z.dim() == 4 else -1
        rho = z.pow(2).mean(dim=dim, keepdim=True).add(eps).sqrt()
        gate = torch.sigmoid(z / rho)
        ctx.save_for_backward(z)
        return z * gate

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        z, = ctx.saved_tensors
        sig_z = torch.sigmoid(z)
        dsig_z = sig_z * (1.0 - sig_z)
        h_z = sig_z + z * dsig_z
        return grad_out * h_z, None


class NiLU_Surr(nn.Module):
    """NiLU with surrogate backward."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return _NiLUSurrFn.apply(z, self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


# ── Functional interfaces ─────────────────────────────────────────

def nelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * 0.5 * (1.0 + torch.erf((z / rho) * _INV_SQRT2))


def nilu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = _rms(z, eps)
    return z * torch.sigmoid(z / rho)
