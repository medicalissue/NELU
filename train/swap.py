"""Swap activation modules throughout a model tree.

The core experiment of this repository: take a model that ships with
:class:`torch.nn.GELU` or :class:`torch.nn.SiLU` (or a library-specific
subclass thereof) and replace every instance with a Gate Normalization
variant. The swap operates on the module tree, so it is architecture- and
library-agnostic: timm, torchvision, HuggingFace models all work unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn

from gate_norm import NELU, NiLU
from gate_norm.core import GateNorm
from gate_norm.reduction import DimsLike, RmsMode


# ── Baseline-class discovery ────────────────────────────────────────────
#
# timm ships its own GELU / SiLU subclasses (timm.layers.activations.GELU,
# FastGELU, SiLU, …) that do not inherit from the torch built-ins. We union
# the known torch classes with anything importable from timm so an
# ``isinstance`` check works everywhere.

def _timm_subclasses(names: Iterable[str]) -> tuple[type, ...]:
    out: list[type] = []
    try:
        import timm.layers.activations as _ta  # type: ignore[import-untyped]
    except ImportError:
        return tuple(out)
    for n in names:
        cls = getattr(_ta, n, None)
        if cls is not None:
            out.append(cls)
    return tuple(out)


GELU_TYPES: tuple[type, ...] = (nn.GELU,) + _timm_subclasses(
    ("GELU", "FastGELU", "QuickGELU", "ApproxGELU")
)
SILU_TYPES: tuple[type, ...] = (nn.SiLU,) + _timm_subclasses(("SiLU", "Swish"))


# ── Generic replacer ─────────────────────────────────────────────────────


def replace_activation(
    model: nn.Module,
    source_types: type | tuple[type, ...],
    factory,
) -> int:
    """Recursively replace every child of type ``source_types`` by ``factory()``.

    Returns the number of modules replaced. The model is modified in place.
    """
    if isinstance(source_types, type):
        source_types = (source_types,)

    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, source_types):
            setattr(model, name, factory())
            count += 1
        else:
            count += replace_activation(child, source_types, factory)
    return count


# ── Convenience wrappers ─────────────────────────────────────────────────


def _factory(cls: type[GateNorm], **kwargs: Any):
    def make():
        return cls(**kwargs)
    return make


def gelu_to_nelu(
    model: nn.Module,
    *,
    rms_mode: RmsMode | DimsLike = "per_token",
    eps: float = 1e-6,
    gamma_init: float = 1e-6,
) -> int:
    """Swap every GELU instance for :class:`gate_norm.NELU`."""
    return replace_activation(
        model,
        GELU_TYPES,
        _factory(NELU, rms_mode=rms_mode, eps=eps, gamma_init=gamma_init),
    )


def silu_to_nilu(
    model: nn.Module,
    *,
    rms_mode: RmsMode | DimsLike = "per_token",
    eps: float = 1e-6,
    gamma_init: float = 1e-6,
) -> int:
    """Swap every SiLU instance for :class:`gate_norm.NiLU`."""
    return replace_activation(
        model,
        SILU_TYPES,
        _factory(NiLU, rms_mode=rms_mode, eps=eps, gamma_init=gamma_init),
    )


def apply_gate_normalization(
    model: nn.Module,
    activation: str,
    *,
    rms_mode: RmsMode | DimsLike | None = None,
    eps: float = 1e-6,
    gamma_init: float = 1e-6,
) -> int:
    """Dispatch based on a string activation name.

    ``activation`` may be one of ``"nelu"``, ``"nilu"``, or a baseline name
    (``"gelu"``, ``"silu"``) in which case no swap is performed.

    ``rms_mode`` defaults to ``"per_token"`` for ``nelu`` (typically used in
    transformers / channels-last CNNs) and ``"per_sample"`` for ``nilu``
    (typically used in NCHW EfficientNet-style CNNs). Either default can be
    overridden.

    After the generic swap, architecture-aware rewiring runs to assign the
    rms axes that match the mixing axes of the preceding linear operation —
    most notably, in EfficientNet's InvertedResidual block the two
    activations sit in different contexts (channel-mixing pointwise vs.
    spatial-mixing depthwise) and therefore need different axes.
    """
    act = activation.lower()
    if act in {"gelu", "silu", "relu"}:
        return 0
    if act == "nelu":
        mode = rms_mode if rms_mode is not None else "per_token"
        n = gelu_to_nelu(model, rms_mode=mode, eps=eps, gamma_init=gamma_init)
    elif act == "nilu":
        mode = rms_mode if rms_mode is not None else "per_sample"
        n = silu_to_nilu(model, rms_mode=mode, eps=eps, gamma_init=gamma_init)
    else:
        raise ValueError(f"Unknown activation {activation!r}")

    # Post-swap: EfficientNet MBConv two-activation split.
    rewire_efficientnet_mbconv(model)
    return n


def rewire_efficientnet_mbconv(model: nn.Module) -> int:
    """Assign per-location rms axes to GateNorm activations inside timm's
    ``InvertedResidual`` (MBConv) blocks.

    MBConv contains two main activation sites in NCHW (N, C, H, W) layout:

    * ``bn1`` — follows the 1×1 pointwise expansion ``conv_pw``. The
      preceding linear op mixes only the channel axis, so the matching
      RMS axis is ``(1,)``.
    * ``bn2`` — follows the k×k depthwise convolution ``conv_dw``, which
      mixes only the spatial axes (channels remain independent), so the
      matching RMS axes are ``(2, 3)``.

    In addition, each MBConv's Squeeze-and-Excite block contains its own
    activation (``se.act1``) on a tensor with H=W=1 following ``conv_reduce``
    (1×1 channel mix). The matching axis there is also ``(1,)``; for tensors
    with H=W=1 this coincides numerically with ``(1, 2, 3)``, but the
    explicit ``(1,)`` keeps the model consistent with the mixing-axes
    principle.

    ``DepthwiseSeparableConv`` and the fused ``EdgeResidual`` variant are
    handled analogously when they appear.

    timm's ``norm_act_layer`` packs a BatchNorm and an activation into a
    single module; the activation lives in an attribute conventionally named
    ``act`` or ``act1``. We locate the GateNorm module inside the stored
    ``bn1``/``bn2``/``se`` subtree rather than relying on a specific
    attribute path.

    Returns the number of GateNorm modules whose ``rms_mode`` was updated.
    """
    updated = 0

    def _set_mode(submodule: nn.Module, axes: tuple[int, ...]) -> int:
        count = 0
        for m in submodule.modules():
            if isinstance(m, GateNorm):
                m.rms_mode = axes
                count += 1
        return count

    try:
        from timm.models._efficientnet_blocks import (  # type: ignore[import-untyped]
            InvertedResidual,
            DepthwiseSeparableConv,
            EdgeResidual,
        )
    except ImportError:
        return 0

    for module in model.modules():
        if isinstance(module, InvertedResidual):
            # conv_pw → bn1 (channel mixing); conv_dw → bn2 (spatial mixing);
            # se.conv_reduce → se.act1 (channel mixing on (N, rd_chs, 1, 1)).
            if hasattr(module, "bn1"):
                updated += _set_mode(module.bn1, (1,))
            if hasattr(module, "bn2"):
                updated += _set_mode(module.bn2, (2, 3))
            if hasattr(module, "se"):
                updated += _set_mode(module.se, (1,))
        elif isinstance(module, DepthwiseSeparableConv):
            # conv_dw (spatial) → bn1; conv_pw (channel) → bn2;
            # se.conv_reduce → se.act1 (channel mixing).
            if hasattr(module, "bn1"):
                updated += _set_mode(module.bn1, (2, 3))
            if hasattr(module, "bn2"):
                updated += _set_mode(module.bn2, (1,))
            if hasattr(module, "se"):
                updated += _set_mode(module.se, (1,))
        elif isinstance(module, EdgeResidual):
            # conv_exp (k×k, channel+spatial) → bn1; conv_pwl is
            # apply_act=False so bn2 contains no activation;
            # se.conv_reduce → se.act1 (channel mixing).
            if hasattr(module, "bn1"):
                updated += _set_mode(module.bn1, (1, 2, 3))
            if hasattr(module, "bn2"):
                updated += _set_mode(module.bn2, (1,))
            if hasattr(module, "se"):
                updated += _set_mode(module.se, (1,))
    return updated
