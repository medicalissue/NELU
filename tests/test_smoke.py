"""Smoke tests — verify forward + backward on a real CIFAR-scale model.

These do not train to accuracy; they only check that the library composes
with a realistic training-style graph (conv + BN + activation swap).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from train.cifar import build_model
from train.swap import apply_gate_normalization, replace_activation


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "nelu", "nilu"])
def test_cifar_resnet_single_step(activation: str) -> None:
    """Build ResNet-20 via the CIFAR factory, swap its activation, do one step."""
    torch.manual_seed(0)
    model = build_model("resnet20", activation=activation, num_classes=100)
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 100, (4,))

    logits = model(x)
    assert logits.shape == (4, 100)

    loss = loss_fn(logits, y)
    loss.backward()
    optim.step()

    assert torch.isfinite(loss)


def test_apply_gate_normalization_on_timm_free_backbone() -> None:
    """``apply_gate_normalization`` works even when the model tree contains
    no timm-specific subclasses. CIFAR ResNet-20 from chenyaofo is pure
    ``nn.Module`` + ``nn.ReLU``, which is the edge case where the
    timm-subclass union reduces to just ``nn.GELU``.
    """
    torch.manual_seed(1)
    model = build_model("resnet20", activation="relu", num_classes=100)

    # Replace every ReLU with a plain nn.GELU to set up the test.
    replace_activation(model, nn.ReLU, lambda: nn.GELU())
    assert all(not isinstance(m, nn.ReLU) for m in model.modules())

    # Now the Gate Normalization swap should find every GELU and replace it.
    n = apply_gate_normalization(model, "nelu")
    assert n > 0

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)
