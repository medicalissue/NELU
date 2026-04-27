"""ResNet for CIFAR-100 with explicit prev-activation gradient flow.

Adapted from chenyaofo/pytorch-cifar-models BasicBlock-based ResNet, but
with two structural changes:

1. Each ``BasicBlock`` has its **own pair of activation modules** (instead
   of sharing a single ``self.relu``), so per-activation learnable state
   (α for ResAct) can attach to its forward site.

2. Both ``BasicBlock.forward`` and ``ResNet.forward`` thread an explicit
   ``prev_act`` argument through every activation site. ``prev_act`` is the
   *output of the previous activation* in module-declaration order, kept
   on the autograd graph so gradient flows backward through the cross-layer
   linear mixing term in :class:`ResActGELU`.

The resulting graph keeps ResNet's identity skip intact and adds a
**second gradient bridge** through the activation stream:

    a_l = act_l(z_l, a_{l-1})
        = GELU(z_l) − ½·z_l + ½·[σ(α)·z_l + (1−σ(α))·a_{l-1}]

    ∂loss/∂a_{l-1} now includes ½(1−σ(α)) · ∂loss/∂a_l   (cross-layer)

For non-ResAct activations (``relu``, ``gelu``, ``silu``, ``nelu``, etc.)
the ``prev_act`` argument is passed but ignored by the wrapper, so the
network reduces exactly to the standard chenyaofo CifarResNet.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from gate_norm import ResActGELU


# ──────────────────────────────────────────────────────────────────────
# Activation wrappers
# ──────────────────────────────────────────────────────────────────────


class _PrevAware(nn.Module):
    """Two-arg wrapper around a stateless activation.

    Lets every block call ``act(z, prev_act)`` regardless of whether the
    underlying activation actually consumes ``prev_act``. Standard
    activations (GELU, ReLU, SiLU, NELU…) ignore the second argument; a
    :class:`ResActGELU` consumes it to mix the cross-layer term.
    """

    def __init__(self, fn: nn.Module, uses_prev: bool = False):
        super().__init__()
        self.fn = fn
        self.uses_prev = uses_prev

    def forward(
        self, x: torch.Tensor, prev_act: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.uses_prev:
            return self.fn(x, prev_act)
        return self.fn(x)


class ResActGELUGraph(nn.Module):
    """ResAct on GELU's odd-even decomposition with **graph-attached prev**.

    Same math as :class:`gate_norm.ResActGELU` but takes ``prev`` as a
    forward argument (not a cached buffer), so gradients flow through the
    cross-layer linear mixing term.

        y = GELU(x) − 0.5·x + 0.5·[σ(α)·x + (1 − σ(α))·prev]

    First-call / shape-mismatch fallback: ``y = GELU(x)``.
    """

    def __init__(self, alpha_init: float = 5.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self, x: torch.Tensor, prev: Optional[torch.Tensor]
    ) -> torch.Tensor:
        gelu_out = nn.functional.gelu(x)
        if prev is None or prev.shape != x.shape:
            return gelu_out
        s = torch.sigmoid(self.alpha)
        return gelu_out - 0.5 * x + 0.5 * (s * x + (1.0 - s) * prev)

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = torch.sigmoid(self.alpha).item()
        return f"alpha={self.alpha.item():.3f}, sigmoid(alpha)={s:.4f}"


def _make_act(name: str) -> _PrevAware:
    """Construct a per-site activation module, prev-aware where applicable."""
    if name == "relu":
        return _PrevAware(nn.ReLU(inplace=False), uses_prev=False)
    if name == "gelu":
        return _PrevAware(nn.GELU(), uses_prev=False)
    if name == "silu":
        return _PrevAware(nn.SiLU(inplace=False), uses_prev=False)
    if name == "resact_gelu_a5":
        return _PrevAware(ResActGELUGraph(alpha_init=5.0), uses_prev=True)
    if name == "resact_gelu_a0":
        return _PrevAware(ResActGELUGraph(alpha_init=0.0), uses_prev=True)
    raise ValueError(f"Unsupported activation for ResNet-ResAct: {name!r}")


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet basic block with two prev-aware activations.

    forward(x, prev_act) → (out, last_act_out)
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        *,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = _make_act(activation)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = _make_act(activation)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self, x: torch.Tensor, prev_act: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out, prev_act)
        # ``out`` is now this block's first activation output. Pass it as
        # prev_act for the second activation site.
        first_act_out = out

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.act2(out, first_act_out)

        return out, out


class CifarResNet(nn.Module):
    """ResNet for CIFAR with explicit prev-activation propagation.

    The standard chenyaofo recipe (depth = 6n+2, widths 16/32/64) with
    activation modules unique per site so ResAct's ``α`` is per-layer and
    ``prev_act`` is threaded through ``forward``.
    """

    def __init__(
        self, layers: list[int], num_classes: int = 100, *,
        activation: str = "relu",
    ):
        super().__init__()
        self.activation = activation
        self.inplanes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.act_stem = _make_act(activation)

        self.layer1 = self._make_layer(16, layers[0], stride=1)
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.ModuleList:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        modules = [
            BasicBlock(self.inplanes, planes, stride, downsample,
                       activation=self.activation),
        ]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            modules.append(BasicBlock(self.inplanes, planes,
                                      activation=self.activation))
        # ModuleList lets us iterate explicitly in forward and pass
        # ``prev_act`` between blocks.
        return nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_stem(out, None)        # first activation: no prev
        prev_act = out

        for block in self.layer1:
            out, prev_act = block(out, prev_act)
        for block in self.layer2:
            out, prev_act = block(out, prev_act)
        for block in self.layer3:
            out, prev_act = block(out, prev_act)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ──────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────


_RESNET_LAYERS: dict[str, list[int]] = {
    "resnet20":  [3, 3, 3],
    "resnet32":  [5, 5, 5],
    "resnet44":  [7, 7, 7],
    "resnet56":  [9, 9, 9],
    "resnet110": [18, 18, 18],
}


def build_resnet_resact(
    name: str = "resnet56",
    activation: str = "resact_gelu_a5",
    num_classes: int = 100,
) -> CifarResNet:
    """Construct a CIFAR ResNet with prev-aware activations."""
    if name not in _RESNET_LAYERS:
        raise ValueError(
            f"Unknown resnet variant {name!r}. "
            f"Choices: {sorted(_RESNET_LAYERS)}"
        )
    return CifarResNet(_RESNET_LAYERS[name], num_classes=num_classes,
                       activation=activation)


def collect_resact_graph_stats(
    model: nn.Module, prefix: str = "resact",
) -> dict[str, float]:
    """Per-layer α / σ(α) collector for :class:`ResActGELUGraph`."""
    alphas: list[float] = []
    sigmas: list[float] = []
    out: dict[str, float] = {}
    for m in model.modules():
        if not isinstance(m, ResActGELUGraph):
            continue
        a = m.alpha.detach().float().item()
        s = torch.sigmoid(m.alpha.detach().float()).item()
        out[f"{prefix}/alpha/layer_{len(alphas)}"] = a
        out[f"{prefix}/sigmoid_alpha/layer_{len(sigmas)}"] = s
        alphas.append(a)
        sigmas.append(s)
    if alphas:
        for vals, key in [(alphas, "alpha"), (sigmas, "sigmoid_alpha")]:
            n = len(vals)
            mean = sum(vals) / n
            var = sum((x - mean) ** 2 for x in vals) / max(1, n - 1)
            out[f"{prefix}/{key}/mean"] = mean
            out[f"{prefix}/{key}/min"] = min(vals)
            out[f"{prefix}/{key}/max"] = max(vals)
            out[f"{prefix}/{key}/std"] = var ** 0.5
            out[f"{prefix}/{key}/n_modules"] = float(n)
    return out
