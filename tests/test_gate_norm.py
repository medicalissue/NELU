"""Tests for the Gate Normalization core and its concrete instances."""

from __future__ import annotations

import math

import pytest
import torch

from gate_norm import NELU, NELUGLU, NiLU, NiLUGLU, SwiGLU, gate_norm
from gate_norm.core import GateNorm


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _centered(x: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """Reference implementation of (x - μ) / σ used to check formulas."""
    mu = x.mean(dim=axes, keepdim=True)
    var = x.var(dim=axes, keepdim=True, unbiased=False)
    return (x - mu) / var.sqrt()


# ── Forward identities ──────────────────────────────────────────────────


def test_nelu_forward_shape_channel() -> None:
    x = torch.randn(2, 3, 16)
    assert NELU()(x).shape == x.shape


def test_nilu_forward_shape_sample() -> None:
    x = torch.randn(2, 8, 5, 5)
    assert NiLU(norm_axes="sample")(x).shape == x.shape


def test_gamma_zero_recovers_constant_times_x() -> None:
    """At γ = 0 both NELU and NiLU collapse to x · g(0) = x · 0.5."""
    torch.manual_seed(0)
    x = torch.randn(4, 16)

    act_nelu = NELU(gamma_init=0.0, eps=0.0)
    act_nilu = NiLU(gamma_init=0.0, eps=0.0)

    assert torch.allclose(act_nelu(x), x * 0.5, atol=1e-6)
    assert torch.allclose(act_nilu(x), x * 0.5, atol=1e-6)


def test_nelu_matches_explicit_formula() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 8)
    gamma, beta = 0.3, 0.4
    act = NELU(gamma_init=gamma, beta_init=beta, eps=0.0)
    with torch.no_grad():
        u = _centered(x, axes=(-1,))
        t = gamma * u + beta
        expected = x * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))
    assert torch.allclose(act(x), expected, atol=1e-6)


def test_nilu_matches_explicit_formula() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 8)
    gamma, beta = 0.7, -0.2
    act = NiLU(gamma_init=gamma, beta_init=beta, eps=0.0)
    with torch.no_grad():
        u = _centered(x, axes=(-1,))
        expected = x * torch.sigmoid(gamma * u + beta)
    assert torch.allclose(act(x), expected, atol=1e-6)


# ── Shift + scale invariance of the gate input ──────────────────────────


def test_gate_input_invariant_to_affine_reparameterization() -> None:
    """(x - μ)/σ is invariant under x ↦ a·x + b (a > 0)."""
    torch.manual_seed(3)
    x = torch.randn(4, 32)
    a, b = 3.7, -1.9
    act_a = NELU(gamma_init=0.5)
    act_b = NELU(gamma_init=0.5)
    y_a = act_a(x)
    y_b = act_b(a * x + b)

    # The gate is reparameterization-invariant, so the gate value is the
    # same on matched elements. The outer z multiplication differs because
    # z₂ = a·z + b, so we compare gate values directly.
    with torch.no_grad():
        gate_a = y_a / x
        gate_b = y_b / (a * x + b)
    mask = (x.abs() > 1e-3)
    assert torch.allclose(gate_a[mask], gate_b[mask], atol=1e-5)


# ── Backward ────────────────────────────────────────────────────────────


def test_backward_produces_gamma_and_beta_gradients() -> None:
    torch.manual_seed(4)
    x = torch.randn(8, 32, requires_grad=True)
    act = NELU(gamma_init=0.3, beta_init=0.2)
    y = act(x)
    y.sum().backward()
    assert act.gamma.grad is not None and act.gamma.grad.shape == (1,)
    assert act.beta.grad is not None and act.beta.grad.shape == (1,)
    # Both gradients should be non-trivial at a non-zero operating point.
    assert act.gamma.grad.abs().item() > 0
    assert act.beta.grad.abs().item() > 0


def test_beta_zero_matches_betaless_centered_formula() -> None:
    """With β = 0 the module reduces to the β-less centered formula."""
    torch.manual_seed(40)
    x = torch.randn(4, 16)
    gamma = 0.5
    act = NELU(gamma_init=gamma, beta_init=0.0, eps=0.0)
    with torch.no_grad():
        u = _centered(x, axes=(-1,))
        expected = x * 0.5 * (1.0 + torch.erf(gamma * u * _INV_SQRT2))
    assert torch.allclose(act(x), expected, atol=1e-6)


def test_state_dict_back_compat_beta_missing() -> None:
    """Pre-β checkpoints (γ only) load cleanly with β initialized to zero."""
    torch.manual_seed(41)
    old_sd = {"gamma": torch.tensor([0.42])}  # no "beta" key
    m = NELU()
    missing, unexpected = m.load_state_dict(old_sd, strict=True)
    assert missing == [] and unexpected == []
    assert m.gamma.item() == pytest.approx(0.42)
    assert m.beta.item() == pytest.approx(0.0)


def test_state_dict_back_compat_gamma_zero_dim() -> None:
    """0-d γ from much older checkpoints reshapes to (1,)."""
    old_sd = {"gamma": torch.tensor(0.33)}  # 0-dim scalar
    m = NELU()
    m.load_state_dict(old_sd, strict=True)
    assert m.gamma.shape == (1,) and m.gamma.item() == pytest.approx(0.33)


# ── GateNorm subclassing ────────────────────────────────────────────────


class _SquareGate(GateNorm):
    @staticmethod
    def _gate_python(t):
        return t * t


def test_subclassing_gate_function() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 16)
    gn = _SquareGate(gamma_init=1.0, eps=0.0)
    with torch.no_grad():
        u = _centered(x, axes=(-1,))
        expected = x * u * u
    assert torch.allclose(gn(x), expected, atol=1e-6)


# ── Functional API ──────────────────────────────────────────────────────


def test_functional_gate_norm_matches_module() -> None:
    torch.manual_seed(6)
    x = torch.randn(2, 16)
    gamma, beta = 0.25, 0.15
    module = NELU(gamma_init=gamma, beta_init=beta, eps=1e-6)
    functional = gate_norm(
        x,
        gate_fn=lambda t: 0.5 * (1.0 + torch.erf(t * _INV_SQRT2)),
        gamma=gamma,
        beta=beta,
        norm_axes="channel",
        eps=1e-6,
    )
    assert torch.allclose(module(x), functional, atol=1e-6)


# ── GLU variants ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [NELUGLU, NiLUGLU])
def test_glu_variants_match_swiglu_param_count(cls: type) -> None:
    dim = 256
    swiglu = SwiGLU(dim)
    variant = cls(dim)
    n_swiglu = sum(p.numel() for p in swiglu.parameters())
    # The variant adds exactly two scalar parameters: γ and β.
    n_variant = sum(p.numel() for p in variant.parameters())
    assert n_variant == n_swiglu + 2


def test_glu_variants_forward_shape() -> None:
    x = torch.randn(2, 8, 128)
    y = NELUGLU(128)(x)
    assert y.shape == x.shape
