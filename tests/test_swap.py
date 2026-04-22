"""Tests for :mod:`train.swap` — activation replacement on a module tree."""

from __future__ import annotations

import torch
import torch.nn as nn

from gate_norm import NELU, NiLU
from train.swap import (
    apply_gate_normalization,
    gelu_to_nelu,
    replace_activation,
    silu_to_nilu,
)


def _make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 16),
        nn.GELU(),
        nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
        ),
        nn.Linear(16, 16),
        nn.SiLU(),
    )


def test_gelu_to_nelu_replaces_two_instances() -> None:
    model = _make_model()
    n = gelu_to_nelu(model)
    assert n == 2
    gelu_count = sum(1 for m in model.modules() if isinstance(m, nn.GELU))
    nelu_count = sum(1 for m in model.modules() if isinstance(m, NELU))
    assert gelu_count == 0
    assert nelu_count == 2


def test_silu_to_nilu_replaces_one_instance() -> None:
    model = _make_model()
    n = silu_to_nilu(model)
    assert n == 1
    assert sum(1 for m in model.modules() if isinstance(m, nn.SiLU)) == 0
    assert sum(1 for m in model.modules() if isinstance(m, NiLU)) == 1


def test_apply_gate_normalization_dispatch() -> None:
    model = _make_model()
    assert apply_gate_normalization(model, "gelu") == 0
    assert apply_gate_normalization(model, "nelu") == 2
    assert apply_gate_normalization(model, "nilu") == 1


def test_replace_activation_preserves_forward_shape() -> None:
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(4, 16)
    pre = model(x)
    apply_gate_normalization(model, "nelu")
    post = model(x)
    assert post.shape == pre.shape


def test_replace_activation_is_in_place_and_counts_correctly() -> None:
    model = _make_model()
    n = replace_activation(model, nn.Linear, lambda: nn.Linear(16, 16))
    # Three linear layers sit at the top of the Sequential plus the nested one.
    assert n == 3
