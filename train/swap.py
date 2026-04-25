"""Swap activation modules throughout a model tree.

The core experiment of this repository: take a model that ships with
:class:`torch.nn.GELU` or :class:`torch.nn.SiLU` (or a library-specific
subclass thereof) and replace every instance with a Gate Normalization
variant. The swap is architecture-aware: the activation's normalization
axes are chosen to match the mixing axes of the preceding linear op, so
the gate "sees" the same statistical granularity the upstream linear
created.

Entry point: :func:`apply_gate_normalization`. It runs a single recursive
pass that, at each candidate site, consults an architecture policy to
pick ``norm_axes``. Having one pass (rather than the older "set a default
then rewire later" two-phase flow) makes the policy traceable to one
file.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch.nn as nn

from gate_norm import NELU, NiLU
from gate_norm.core import GateNorm
from gate_norm.layout import DimsLike, NormAxes


# ── Baseline-class discovery ────────────────────────────────────────────

def _timm_subclasses(names: Iterable[str]) -> tuple[type, ...]:
    """Collect timm's GELU/SiLU variants if they're importable."""
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


# ── Architecture policy ──────────────────────────────────────────────────
#
# ``norm_axes`` for a GateNorm activation should match the mixing axes of
# the *immediately preceding* linear operation. That relationship is
# determined by where the activation sits inside a block, not by what
# activation class it is, so we encode the rule as a policy keyed by the
# enclosing block's type. :func:`_axes_for_site` walks ancestors to find
# the matching block.


def _timm_blocks() -> dict[str, type]:
    """Resolve timm's MBConv-family block classes lazily."""
    try:
        from timm.models._efficientnet_blocks import (  # type: ignore[import-untyped]
            InvertedResidual, DepthwiseSeparableConv, EdgeResidual,
        )
    except ImportError:
        return {}
    return {
        "InvertedResidual":       InvertedResidual,
        "DepthwiseSeparableConv": DepthwiseSeparableConv,
        "EdgeResidual":           EdgeResidual,
    }


# Block-level policy: for each known block class, map the attribute-name
# through which the activation is reached to the norm_axes that match the
# upstream linear op.
#
#   * 1×1 pointwise  → channel mixing only        → (1,)
#   * k×k depthwise  → spatial mixing only        → (2, 3)
#   * k×k full/fused → channel + spatial mixing   → (1, 2, 3)
#   * SE reduce/expand (1×1 conv on (N, rd, 1, 1))
#                    → channel mix                 → (1,)
_MBCONV_POLICY: dict[str, dict[str, tuple[int, ...]]] = {
    "InvertedResidual": {
        "bn1": (1,),       # after 1×1 pointwise expansion
        "bn2": (2, 3),     # after k×k depthwise
        "se":  (1,),
    },
    "DepthwiseSeparableConv": {
        "bn1": (2, 3),     # after k×k depthwise
        "bn2": (1,),       # after 1×1 pointwise
        "se":  (1,),
    },
    "EdgeResidual": {
        "bn1": (1, 2, 3),  # after k×k fused conv mixing both
        "bn2": (1,),       # after 1×1 pointwise
        "se":  (1,),
    },
}


def _axes_from_ancestors(
    ancestors: list[tuple[str, nn.Module]],
    default: NormAxes | DimsLike | None = None,
):
    """Pick ``norm_axes`` by walking up ``ancestors`` for an MBConv match.

    ``ancestors`` is ordered *bottom-up*: ``ancestors[0]`` is the
    activation's direct parent together with the attribute the parent
    uses to reach it, ``ancestors[1]`` is the grandparent, and so on.

    Returns the policy-dictated axes if any ancestor is a known MBConv
    block type and its downward attribute hits :data:`_MBCONV_POLICY`.
    Returns ``default`` otherwise; callers pass ``None`` to distinguish
    "policy had nothing to say" from "policy said ``sample``" and hand
    resolution off to the conv-introspection fallback.
    """
    blocks = _timm_blocks()
    if not blocks:
        return default

    for i, (attr, mod) in enumerate(ancestors):
        for block_name, cls in blocks.items():
            if isinstance(mod, cls) and attr in _MBCONV_POLICY[block_name]:
                return _MBCONV_POLICY[block_name][attr]
    return default


# ── Default norm_axes per activation + arch ─────────────────────────────


def default_norm_axes(activation: str, model_name: str) -> NormAxes:
    """Sensible ``norm_axes`` default for a ``(activation, model)`` pair.

    Most ImageNet models want ``"channel"`` — the activation sits on the
    output of a pointwise (or equivalently channel-mixing) linear. The
    EfficientNet family used ``"sample"`` historically; that remains the
    default there but per-site overrides in :data:`_MBCONV_POLICY`
    dominate at the MBConv activation sites, which is where they live.
    """
    name = model_name.lower()
    if "efficientnet" in name:
        return "sample"
    return "channel"


# ── Core replacer ────────────────────────────────────────────────────────


def _replace_with_policy(
    model: nn.Module,
    baseline_types: tuple[type, ...],
    gate_cls: type[GateNorm],
    default_axes: NormAxes | DimsLike,
    *,
    eps: float,
    gamma_init: float,
    beta_init: float,
) -> int:
    """Walk the module tree; substitute baseline activations for GateNorm.

    Axis resolution order, first hit wins:
      1. MBConv / DepthwiseSeparableConv / EdgeResidual policy
         (:func:`_axes_from_ancestors`).
      2. Nearest preceding ``nn.Conv2d`` / ``nn.Linear`` in declaration
         order, classified by :func:`_linear_mixing_axes`. This is the
         generic "post-activation auto-axes" used by CIFAR's hub-backed
         models; it also catches ImageNet activations that sit outside
         any MBConv-family block (e.g. EfficientNet's ``bn2`` head).
      3. ``default_axes`` (last-resort fallback for sites that have no
         upstream linear op at all — rare, typically the first activation
         in a stem).

    Walking the tree in two passes is necessary so pass 2 can see the
    complete ordered list of linear ops for step (2) above. Pass 1 just
    records every ``(parent, child_name, ancestors, linear_index)``
    tuple; pass 2 writes the swaps back.
    """
    linear_ops: list[nn.Module] = []
    sites: list[tuple[nn.Module, str, list[tuple[str, nn.Module]], int]] = []

    def sweep(
        root: nn.Module,
        ancestors: list[tuple[str, nn.Module]],
    ) -> None:
        for name, child in root.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                linear_ops.append(child)
            if isinstance(child, baseline_types):
                chain = [(name, root)] + ancestors
                sites.append((root, name, chain, len(linear_ops)))
            else:
                sweep(child, [(name, root)] + ancestors)

    sweep(model, [])

    for parent, name, chain, idx in sites:
        # (1) MBConv-family policy.
        axes = _axes_from_ancestors(chain, default=None)
        # (2) Conv-introspection fallback.
        if axes is None:
            if idx > 0:
                axes = _linear_mixing_axes(linear_ops[idx - 1])
            else:
                axes = default_axes
        setattr(parent, name, gate_cls(
            norm_axes=axes, eps=eps,
            gamma_init=gamma_init, beta_init=beta_init,
        ))
    return len(sites)


# ── Thin, type-specific wrappers ────────────────────────────────────────


def gelu_to_nelu(
    model: nn.Module,
    *,
    norm_axes: NormAxes | DimsLike = "channel",
    eps: float = 1e-6,
    gamma_init: float = 0.0,
    beta_init: float = 0.0,
) -> int:
    """Swap every GELU instance for :class:`gate_norm.NELU`."""
    return _replace_with_policy(
        model, GELU_TYPES, NELU, norm_axes,
        eps=eps, gamma_init=gamma_init, beta_init=beta_init,
    )


def silu_to_nilu(
    model: nn.Module,
    *,
    norm_axes: NormAxes | DimsLike = "channel",
    eps: float = 1e-6,
    gamma_init: float = 0.0,
    beta_init: float = 0.0,
) -> int:
    """Swap every SiLU instance for :class:`gate_norm.NiLU`."""
    return _replace_with_policy(
        model, SILU_TYPES, NiLU, norm_axes,
        eps=eps, gamma_init=gamma_init, beta_init=beta_init,
    )


def apply_gate_normalization(
    model: nn.Module,
    activation: str,
    *,
    norm_axes: NormAxes | DimsLike | None = None,
    eps: float = 1e-6,
    gamma_init: float = 0.0,
    beta_init: float = 0.0,
    model_name: str = "",
) -> int:
    """Dispatch based on a string activation name.

    ``activation`` may be ``"nelu"``, ``"nilu"``, or a baseline
    (``"gelu"``, ``"silu"``, ``"relu"``) in which case no swap is
    performed.

    If ``norm_axes`` is ``None``, :func:`default_norm_axes` chooses a
    default from the activation + model name. For EfficientNet-family
    CNNs the per-site MBConv policy in :data:`_MBCONV_POLICY` overrides
    the default wherever it applies.
    """
    act = activation.lower()
    if act in {"gelu", "silu", "relu"}:
        return 0

    default_axes = norm_axes or default_norm_axes(act, model_name)

    if act == "nelu":
        return gelu_to_nelu(
            model, norm_axes=default_axes, eps=eps,
            gamma_init=gamma_init, beta_init=beta_init,
        )
    if act == "nilu":
        return silu_to_nilu(
            model, norm_axes=default_axes, eps=eps,
            gamma_init=gamma_init, beta_init=beta_init,
        )
    raise ValueError(f"unknown activation {activation!r}")


# ── Generic unconditional replacer (used by tests / ad-hoc scripts) ─────


def replace_activation(
    model: nn.Module,
    source_types: type | tuple[type, ...],
    factory: Callable[[], nn.Module],
) -> int:
    """Recursively replace every child of type ``source_types`` with ``factory()``.

    Kept separate from the architecture-aware path because some callers
    want to install an arbitrary module without threading an axis policy
    through.
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


# ── Linear-/Conv-introspection axis policy ────────────────────────────
#
# When no timm block class drives the policy (torchvision MobileNetV2,
# chenyaofo VGG, custom CIFAR residual blocks) we fall back to inspecting
# the nearest preceding linear op in a single module-tree sweep. ``norm_axes``
# is chosen from that op's shape:
#
#   Conv2d, kernel = 1×1                        → (1,)   channel-mix
#   Conv2d, kernel = k×k, groups=in=out         → (2, 3) depthwise spatial
#   Conv2d, kernel = k×k, otherwise             → sample channel + spatial
#   Linear                                       → channel (input is 2-D)
#
# This is architecture-agnostic: the walker matches every activation to its
# upstream linear by module-declaration order — the same order PyTorch
# reports in ``named_modules`` — so it handles classifier heads (Linear →
# ReLU → Linear) as well as conv bodies without per-architecture tables.


def _linear_mixing_axes(mod: nn.Module) -> tuple[int, ...] | str:
    """Axes over which the gate should normalize given the preceding linear op.

    Returns a mixing-axes spec compatible with ``resolve_axes``. A caller
    that receives ``"sample"`` must only apply it to ≥3-D tensors; the
    sweep below accordingly routes ``nn.Linear`` through the 2-D-safe
    ``"channel"`` alias.
    """
    if isinstance(mod, nn.Conv2d):
        k = mod.kernel_size
        if k == (1, 1):
            return (1,)
        is_depthwise = (
            mod.groups == mod.in_channels == mod.out_channels
            and mod.groups > 1
        )
        if is_depthwise:
            return (2, 3)
        return "sample"
    if isinstance(mod, nn.Linear):
        return "channel"
    raise TypeError(f"unsupported linear op type: {type(mod).__name__}")


def replace_activation_auto_axes(
    model: nn.Module,
    baseline_types: type | tuple[type, ...],
    gate_cls: type[GateNorm],
    *,
    activation_order: str = "post",
    default_axes: NormAxes | DimsLike = "sample",
    eps: float = 1e-6,
    gamma_init: float = 0.0,
    beta_init: float = 0.0,
) -> int:
    """Replace baseline activations with ``gate_cls``, picking axes from a
    nearby :class:`nn.Conv2d` in the module tree.

    Parameters
    ----------
    activation_order : ``"post"`` or ``"pre"``
        * ``"post"`` — activation sits *after* the conv it reads from
          (``Conv → BN → ReLU``, typical of ResNet/MobileNetV2). We pair
          each activation with the **nearest preceding** ``Conv2d`` in
          module-declaration order.
        * ``"pre"`` — activation sits *before* the conv that consumes its
          output (``BN → ReLU → Conv``, pre-activation WideResNet and
          DenseNet). We pair each activation with the **next following**
          ``Conv2d`` instead.

    ``default_axes`` applies to activations that have no matching Conv2d
    on the chosen side — typically the very first or very last site.

    Returns the number of activations replaced.
    """
    if isinstance(baseline_types, type):
        baseline_types = (baseline_types,)
    if activation_order not in ("pre", "post"):
        raise ValueError(f"activation_order must be 'pre' or 'post'")

    # First pass: collect ordered references to every linear op (Conv2d
    # or Linear) and every target activation in module-declaration order.
    linear_ops: list[nn.Module] = []
    # Each activation site is (parent, child_name, linear_index_at_sighting)
    # where the index points into ``linear_ops``. For "post" we pair with
    # the last linear op seen before the activation; for "pre" we pair
    # with the first one seen after.
    sites: list[tuple[nn.Module, str, int]] = []

    def sweep(root: nn.Module) -> None:
        for name, child in root.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                linear_ops.append(child)
            if isinstance(child, baseline_types):
                sites.append((root, name, len(linear_ops)))
            else:
                sweep(child)

    sweep(model)

    # Second pass: resolve axes for each site and install the gate.
    for parent, name, idx in sites:
        if activation_order == "post":
            # nearest preceding linear op is at ``idx - 1``
            if idx == 0:
                axes: NormAxes | DimsLike = default_axes
            else:
                axes = _linear_mixing_axes(linear_ops[idx - 1])
        else:  # "pre"
            # nearest following linear op is at ``idx``
            if idx >= len(linear_ops):
                axes = default_axes
            else:
                axes = _linear_mixing_axes(linear_ops[idx])

        setattr(parent, name, gate_cls(
            norm_axes=axes, eps=eps,
            gamma_init=gamma_init, beta_init=beta_init,
        ))
    return len(sites)
