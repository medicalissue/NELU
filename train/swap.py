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
    """
    act = activation.lower()
    if act in {"gelu", "silu", "relu"}:
        return 0
    if act == "nelu":
        mode = rms_mode if rms_mode is not None else "per_token"
        return gelu_to_nelu(model, rms_mode=mode, eps=eps, gamma_init=gamma_init)
    if act == "nilu":
        mode = rms_mode if rms_mode is not None else "per_sample"
        return silu_to_nilu(model, rms_mode=mode, eps=eps, gamma_init=gamma_init)
    raise ValueError(f"Unknown activation {activation!r}")
