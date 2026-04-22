"""Tests for the Gate Normalization core and its concrete instances."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from gate_norm import NELU, NELUGLU, NiLU, NiLUGLU, SwiGLU, gate_norm
from gate_norm.core import GateNorm


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ── Forward identities ──────────────────────────────────────────────────


def test_nelu_forward_shape_per_token() -> None:
    x = torch.randn(2, 3, 16)
    assert NELU()(x).shape == x.shape


def test_nilu_forward_shape_per_sample() -> None:
    x = torch.randn(2, 8, 5, 5)
    assert NiLU(rms_mode="per_sample")(x).shape == x.shape


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
    gamma = 0.3
    act = NELU(gamma_init=gamma, eps=0.0)
    with torch.no_grad():
        rho = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        t = gamma * x / rho
        expected = x * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))
    assert torch.allclose(act(x), expected, atol=1e-6)


def test_nilu_matches_explicit_formula() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 8)
    gamma = 0.7
    act = NiLU(gamma_init=gamma, eps=0.0)
    with torch.no_grad():
        rho = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        expected = x * torch.sigmoid(gamma * x / rho)
    assert torch.allclose(act(x), expected, atol=1e-6)


# ── Scale invariance ────────────────────────────────────────────────────


def test_scale_invariance_of_gate_input() -> None:
    """γ · x / rms(x) is invariant to positive scalar rescalings of x."""
    torch.manual_seed(3)
    x = torch.randn(4, 32)
    act_a = NELU(gamma_init=0.5)
    act_b = NELU(gamma_init=0.5)
    y_a = act_a(x)
    y_b = act_b(3.7 * x)
    # The gate argument is scale-invariant, so y_b = 3.7 · y_a.
    assert torch.allclose(y_b, 3.7 * y_a, atol=1e-5)


# ── Backward ────────────────────────────────────────────────────────────


def test_backward_produces_gamma_gradient() -> None:
    torch.manual_seed(4)
    x = torch.randn(8, 32, requires_grad=True)
    act = NELU()
    y = act(x)
    y.sum().backward()
    assert act.gamma.grad is not None
    assert act.gamma.grad.shape == (1,)


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
        rho = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        expected = x * (x / rho) ** 2
    assert torch.allclose(gn(x), expected, atol=1e-6)


# ── Functional API ──────────────────────────────────────────────────────


def test_functional_gate_norm_matches_module() -> None:
    torch.manual_seed(6)
    x = torch.randn(2, 16)
    gamma = 0.25
    module = NELU(gamma_init=gamma, eps=1e-6)
    functional = gate_norm(
        x,
        gate_fn=lambda t: 0.5 * (1.0 + torch.erf(t * _INV_SQRT2)),
        gamma=gamma,
        rms_mode="per_token",
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
    # The variant adds exactly one scalar parameter: γ.
    n_variant = sum(p.numel() for p in variant.parameters())
    assert n_variant == n_swiglu + 1


def test_glu_variants_forward_shape() -> None:
    x = torch.randn(2, 8, 128)
    y = NELUGLU(128)(x)
    assert y.shape == x.shape
