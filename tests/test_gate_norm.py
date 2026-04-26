"""Tests for the Gate Normalization core and its concrete instances.

Form under test::

    y = x · g(γ · x / rms(x))

with γ a non-learnable scalar buffer (driven externally by
:class:`gate_norm.GammaWarmup`) and ``rms(x) = sqrt(mean(x²) + eps)``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from gate_norm import GammaWarmup, NELU, NELUGLU, NiLU, NiLUGLU, SwiGLU, gate_norm
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
    """x / rms(x) is invariant under x ↦ a·x for any a > 0."""
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


# ── γ buffer + scheduler ───────────────────────────────────────────────


def test_gamma_is_frozen_parameter() -> None:
    """γ is a Parameter with requires_grad=False — that way torch.compile /
    inductor doesn't specialize it at trace time, but the optimizer also
    won't touch it. The buffer-equivalent behavior comes from grad=False."""
    act = NELU(gamma_init=0.5)
    assert "gamma" in dict(act.named_parameters())
    assert act.gamma.requires_grad is False


def test_backward_does_not_attach_gradient_to_gamma() -> None:
    torch.manual_seed(4)
    x = torch.randn(8, 32, requires_grad=True)
    act = NELU(gamma_init=0.3)
    y = act(x)
    y.sum().backward()
    # γ has requires_grad=False so its .grad never accumulates.
    assert act.gamma.grad is None
    # x should still receive gradient through the module.
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_gamma_warmup_linearly_ramps_buffer() -> None:
    model = nn.Sequential(NELU(), NELU(), NiLU())
    sched = GammaWarmup(model, warmup_steps=10, init=0.0, final=1.0,
                        schedule="linear")
    assert sched.current_gamma == pytest.approx(0.0)
    sched.step(5)
    assert sched.current_gamma == pytest.approx(0.5)
    sched.step(10)
    assert sched.current_gamma == pytest.approx(1.0)
    sched.step(20)  # past warmup → still at final
    assert sched.current_gamma == pytest.approx(1.0)


def test_gamma_warmup_zero_steps_is_constant() -> None:
    model = nn.Sequential(NELU(), NELU())
    sched = GammaWarmup(model, warmup_steps=0, init=0.0, final=1.0)
    # warmup_steps=0 means "fix at final immediately".
    assert sched.current_gamma == pytest.approx(1.0)
    sched.step(5)
    assert sched.current_gamma == pytest.approx(1.0)


def test_gamma_warmup_state_dict_round_trip() -> None:
    model_a = nn.Sequential(NELU())
    sched_a = GammaWarmup(model_a, warmup_steps=100, init=0.0, final=1.0)
    sched_a.step(60)
    state = sched_a.state_dict()

    model_b = nn.Sequential(NELU())
    sched_b = GammaWarmup(model_b, warmup_steps=100, init=0.0, final=1.0)
    sched_b.load_state_dict(state)
    assert sched_b.current_gamma == pytest.approx(0.6, abs=1e-6)


# ── State-dict back-compat ─────────────────────────────────────────────


def test_state_dict_drops_legacy_beta_key() -> None:
    """Centered+learnable v0.2/v0.3 checkpoints carried β; we silently drop it."""
    legacy_sd = {
        "gamma": torch.tensor([0.42]),
        "beta": torch.tensor([0.13]),  # legacy key, no longer used
    }
    m = NELU()
    missing, unexpected = m.load_state_dict(legacy_sd, strict=True)
    assert missing == [] and unexpected == []
    assert m.gamma.item() == pytest.approx(0.42)


def test_state_dict_back_compat_gamma_zero_dim() -> None:
    """0-d γ from much older checkpoints reshapes to (1,)."""
    old_sd = {"gamma": torch.tensor(0.33)}
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
        rms = _rms(x, axes=(-1,))
        u = x / rms
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
def test_glu_variants_match_swiglu_trainable_param_count(cls: type) -> None:
    dim = 256
    swiglu = SwiGLU(dim)
    variant = cls(dim)
    # γ is a Parameter with requires_grad=False, so it's in
    # module.parameters() but contributes zero trainable params. The
    # variant should match SwiGLU's *trainable* param count exactly.
    def n_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_trainable(variant) == n_trainable(swiglu)


def test_glu_variants_forward_shape() -> None:
    x = torch.randn(2, 8, 128)
    y = NELUGLU(128)(x)
    assert y.shape == x.shape
