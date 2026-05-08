"""Tests for the Gate Normalization core and its concrete instances.

Form under test (legacy RMS-only, scalar-γ NELU)::

    y = x · g(γ · x / rms(x))

with ``γ`` a single learnable scalar and
``rms(x) = sqrt(mean(x²) + eps)``.

The default :class:`gate_norm.NELU` is now LN-normalize + per-channel
``γ_c, β_c``; the legacy RMS form is kept as :class:`NELU_RMS` /
:class:`NiLU_RMS`. These tests target the legacy form (the GateNorm
base class with scalar γ); a separate suite covers the new default.
"""

from __future__ import annotations

import math

import pytest
import torch

# Legacy RMS form is exposed as NELU_RMS / NiLU_RMS in the package; alias
# them locally to keep the rest of the file readable.
from gate_norm import (
    NELU_RMS as NELU,
    NiLU_RMS as NiLU,
    NELUGLU,
    NiLUGLU,
    SwiGLU,
    gate_norm,
)
from gate_norm.core import GateNorm


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _rms(x: torch.Tensor, axes: tuple[int, ...], eps: float = 0.0) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=axes, keepdim=True) + eps)


# ── Forward identities ──────────────────────────────────────────────────


def test_nelu_forward_shape_channel() -> None:
    x = torch.randn(2, 3, 16)
    assert NELU()(x).shape == x.shape


def test_nilu_forward_shape_sample() -> None:
    x = torch.randn(2, 8, 5, 5)
    assert NiLU(norm_axes="sample")(x).shape == x.shape


def test_gamma_init_default_is_one() -> None:
    """The default ``gamma_init=1.0`` makes ``γ_eff = 1.0`` at step 0."""
    for cls in (NELU, NiLU):
        m = cls()
        assert m.gamma.item() == pytest.approx(1.0, abs=1e-6)


def test_nelu_matches_explicit_formula() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 8)
    gamma = 0.6
    act = NELU(gamma_init=gamma, eps=0.0)
    with torch.no_grad():
        rms = _rms(x, axes=(-1,))
        t = gamma * x / rms
        expected = x * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))
    assert torch.allclose(act(x), expected, atol=1e-6)


def test_nilu_matches_explicit_formula() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 8)
    gamma = 0.7
    act = NiLU(gamma_init=gamma, eps=0.0)
    with torch.no_grad():
        rms = _rms(x, axes=(-1,))
        expected = x * torch.sigmoid(gamma * x / rms)
    assert torch.allclose(act(x), expected, atol=1e-6)


# ── Scale invariance of the gate input ─────────────────────────────────


def test_gate_input_invariant_to_positive_scaling() -> None:
    """``x / rms(x)`` is invariant under ``x ↦ a·x`` for any ``a > 0``."""
    torch.manual_seed(3)
    x = torch.randn(4, 32)
    a = 3.7
    act_a = NELU(gamma_init=0.5)
    act_b = NELU(gamma_init=0.5)
    with torch.no_grad():
        y_a = act_a(x)
        y_b = act_b(a * x)
        gate_a = y_a / x
        gate_b = y_b / (a * x)
    mask = x.abs() > 1e-3
    assert torch.allclose(gate_a[mask], gate_b[mask], atol=1e-5)


# ── γ as a learnable scalar ─────────────────────────────────────────────


def test_gamma_is_learnable() -> None:
    """``γ`` is a Parameter with grads enabled."""
    m = NELU(gamma_init=0.5)
    params = dict(m.named_parameters())
    assert "gamma" in params
    assert m.gamma.requires_grad is True
    assert m.gamma.item() == pytest.approx(0.5, abs=1e-6)


def test_backward_updates_gamma() -> None:
    torch.manual_seed(4)
    x = torch.randn(8, 32, requires_grad=True)
    m = NELU(gamma_init=0.3)
    y = m(x)
    y.sum().backward()
    assert m.gamma.grad is not None
    assert m.gamma.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


# ── GateNorm subclassing ────────────────────────────────────────────────


class _SquareGate(GateNorm):
    @staticmethod
    def _gate_python(t):
        return t * t


def test_subclassing_gate_function() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 16)
    gn = _SquareGate(gamma_init=1.0, eps=0.0)
    gamma = gn.gamma.item()
    with torch.no_grad():
        rms = _rms(x, axes=(-1,))
        u = gamma * x / rms
        expected = x * u * u
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
        norm_axes="channel",
        eps=1e-6,
    )
    assert torch.allclose(module(x), functional, atol=1e-6)


# ── GLU variants ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [NELUGLU, NiLUGLU])
def test_glu_variants_have_one_extra_scalar_over_swiglu(cls: type) -> None:
    """The gate-normalized GLUs add exactly a single scalar (γ)
    on top of SwiGLU's three Linear blocks."""
    dim = 256
    swiglu_params = sum(p.numel() for p in SwiGLU(dim).parameters())
    variant_params = sum(p.numel() for p in cls(dim).parameters())
    assert variant_params == swiglu_params + 1


def test_glu_variants_forward_shape() -> None:
    x = torch.randn(2, 8, 128)
    y = NELUGLU(128)(x)
    assert y.shape == x.shape
