"""Smoke tests — verify forward + backward on a real CIFAR-scale model.

These do not train to accuracy; they only check that the library composes
with a realistic training-style graph (conv + BN + activation swap + AMP).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from train.cifar import CIFARResNet, build_model
from train.swap import apply_gate_normalization


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "nelu", "nilu"])
def test_cifar_resnet_single_step(activation: str) -> None:
    """Build a CIFARResNet, swap its activation, do one optimizer step."""
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


def test_apply_gate_normalization_on_cifar_resnet() -> None:
    """Swap onto a timm-free CIFAR ResNet — the module tree has no timm
    subclasses, so the union of GELU_TYPES reduces to just ``nn.GELU``."""
    torch.manual_seed(1)
    model = CIFARResNet(20, num_classes=100)

    # Swap ReLU -> GELU first (so apply_gate_normalization has something to
    # rewrite on the second pass).
    n_relu = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
    assert n_relu > 0
    for name, child in list(model.named_modules()):
        pass

    # Replace every ReLU with a plain nn.GELU to set up the test.
    from train.swap import replace_activation
    replace_activation(model, nn.ReLU, lambda: nn.GELU())
    assert all(not isinstance(m, nn.ReLU) for m in model.modules())

    # Now the Gate Normalization swap should find every GELU and replace it.
    n = apply_gate_normalization(model, "nelu")
    assert n > 0

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)
