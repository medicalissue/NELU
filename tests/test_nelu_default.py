"""Tests for the default NELU/NiLU: LN-normalize + per-channel γ_c, β_c.

Form under test::

    μ_c, var_c = pool over position axis
    z_norm     = (x − μ_c) / sqrt(var_c + ε)
    gate       = g(γ_c · z_norm + β_c)
    y          = x · gate

The position axis is rank-dispatched: spatial ``(2, 3)`` for 4-D
tensors, token ``(1,)`` for 3-D.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from gate_norm import NELU, NiLU


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ── Lazy materialization ────────────────────────────────────────────────


def test_gamma_beta_uninitialized_before_forward() -> None:
    """Per-channel γ, β are UninitializedParameter until first forward."""
    m = NELU()
    assert isinstance(m.gamma, nn.UninitializedParameter)
    assert isinstance(m.beta, nn.UninitializedParameter)


def test_cnn_materializes_to_channel_count() -> None:
    """4-D input materializes γ_c, β_c with shape (C,)."""
    m = NELU()
    x = torch.randn(2, 16, 8, 8)
    _ = m(x)
    assert m.gamma.shape == (16,)
    assert m.beta.shape == (16,)


def test_transformer_materializes_to_last_dim() -> None:
    """3-D input materializes γ_c, β_c with shape (C,) where C = last dim."""
    m = NELU()
    x = torch.randn(2, 64, 192)
    _ = m(x)
    assert m.gamma.shape == (192,)
    assert m.beta.shape == (192,)


# ── Forward shape ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [NELU, NiLU])
def test_cnn_forward_shape(cls: type) -> None:
    x = torch.randn(2, 16, 8, 8)
    assert cls()(x).shape == x.shape


@pytest.mark.parametrize("cls", [NELU, NiLU])
def test_transformer_forward_shape(cls: type) -> None:
    x = torch.randn(2, 64, 192)
    assert cls()(x).shape == x.shape


# ── Math identity (CNN, position pooling = spatial) ──────────────────────


def test_nelu_matches_explicit_formula_cnn() -> None:
    """Forward equals the explicit LN + per-channel γ_c, β_c formula."""
    torch.manual_seed(1)
    x = torch.randn(2, 4, 5, 5)
    m = NELU(gamma_init=0.7, beta_init=-0.2, eps=1e-6)
    _ = m(x)  # materialize

    with torch.no_grad():
        # Spatial mean/var per (B, C)
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        z = (x - mu) / torch.sqrt(var + 1e-6)
        gamma = m.gamma.view(1, -1, 1, 1)
        beta = m.beta.view(1, -1, 1, 1)
        t = gamma * z + beta
        expected = x * 0.5 * (1.0 + torch.erf(t * _INV_SQRT2))

    assert torch.allclose(m(x), expected, atol=1e-5)


def test_nilu_matches_explicit_formula_transformer() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 32, 64)
    m = NiLU(gamma_init=0.5, beta_init=0.1, eps=1e-6)
    _ = m(x)

    with torch.no_grad():
        # Token mean/var per (B, C)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        z = (x - mu) / torch.sqrt(var + 1e-6)
        gamma = m.gamma.view(1, 1, -1)
        beta = m.beta.view(1, 1, -1)
        t = gamma * z + beta
        expected = x * torch.sigmoid(t)

    assert torch.allclose(m(x), expected, atol=1e-5)


# ── Shift invariance (LN subtracts the mean, so x → x + c shouldn't matter) ─


def test_shift_invariant_in_position_axis() -> None:
    """Adding a per-channel constant to the input shouldn't change the gate.

    Mean-subtraction inside the LN-style normalize cancels any constant
    that lives along the position axis. We test this on CNN and Tx.
    """
    torch.manual_seed(3)
    for shape, c_shape in [
        ((2, 8, 5, 5), (1, 8, 1, 1)),  # CNN
        ((2, 32, 16), (1, 1, 16)),     # Tx
    ]:
        x = torch.randn(*shape)
        c = torch.randn(*c_shape)
        m_a = NELU(gamma_init=0.5)
        m_b = NELU(gamma_init=0.5)
        with torch.no_grad():
            y_a = m_a(x)
            y_b = m_b(x + c)
        # The gate should be the same; y differs by x · gate vs (x+c) · gate.
        # Compare gate directly.
        gate_a = y_a / x.clamp_min(1e-3) if False else None  # avoid div issues
        # Simpler: gate values differ by 0; test via reconstruction:
        # gate_a = y_a / x, gate_b = y_b / (x + c) where x and (x+c) ≠ 0.
        mask = (x.abs() > 1e-2) & ((x + c).abs() > 1e-2)
        gate_a = (y_a / x)[mask]
        gate_b = (y_b / (x + c))[mask]
        assert torch.allclose(gate_a, gate_b, atol=1e-4), \
            f"shift-invariance broken for shape {shape}"


# ── Backward / learnability ──────────────────────────────────────────────


def test_per_channel_gamma_beta_are_learnable() -> None:
    m = NELU(gamma_init=1.0, beta_init=0.0)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert m.gamma.grad is not None and m.gamma.grad.shape == (16,)
    assert m.beta.grad is not None and m.beta.grad.shape == (16,)
    assert m.gamma.grad.abs().sum() > 0
    assert m.beta.grad.abs().sum() > 0


# ── Init values ──────────────────────────────────────────────────────────


def test_default_gamma_init_is_one_after_materialization() -> None:
    m = NELU()
    _ = m(torch.randn(2, 8, 4, 4))
    assert torch.allclose(m.gamma, torch.ones_like(m.gamma))
    assert torch.allclose(m.beta, torch.zeros_like(m.beta))


def test_custom_init_propagates() -> None:
    m = NELU(gamma_init=0.3, beta_init=-0.1)
    _ = m(torch.randn(2, 8, 4, 4))
    assert torch.allclose(m.gamma, torch.full_like(m.gamma, 0.3))
    assert torch.allclose(m.beta, torch.full_like(m.beta, -0.1))
