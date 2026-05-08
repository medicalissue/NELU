"""Swap activation modules throughout a model tree.

The core experiment of this repository: take a model that ships with
:class:`torch.nn.GELU` or :class:`torch.nn.SiLU` (or a library-specific
subclass thereof) and replace every instance with a Gate Normalization
variant.

Axis policy is unified across architectures: every swapped activation
gets ``norm_axes="position"``, which is rank-dispatched at runtime by
:func:`gate_norm.layout.resolve_axes`:

  * 4-D ``(B, C, H, W)``  → spatial axes ``(2, 3)``  — CNN
  * 3-D ``(B, T, C)``     → token  axis ``(1,)``    — Transformer

Pool over the position axis to estimate per-channel scale, normalize
the gate input by it, and broadcast the resulting gate back across
positions. Same primitive in CNN and Transformer; only the axis index
differs.

Entry point: :func:`apply_gate_normalization`.
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


# ── Default norm_axes per activation + arch ─────────────────────────────


def default_norm_axes(activation: str, model_name: str) -> NormAxes:
    """Unified ``norm_axes`` default: pool over the position axis.

    The position axis is dispatched at runtime by tensor rank
    (see :data:`gate_norm.layout._RANK_ALIASES`):

      * 4-D ``(B, C, H, W)``  → spatial ``(2, 3)``  — CNN
      * 3-D ``(B, T, C)``     → token   ``(1,)``    — Transformer

    The same alias works for every architecture; per-block axis policies
    are no longer needed because the rank fully determines the position
    axis. The signature is kept for API stability.
    """
    return "position"


# ── Core replacer ────────────────────────────────────────────────────────


def _channels_of(mod: nn.Module) -> int | None:
    """Return the channel-count produced by a Linear/Conv module, or None.

    The number of feature channels each activation sees equals the
    out-features of the nearest preceding Linear/Conv. We use that to
    eagerly materialize per-channel γ_c, β_c at swap time so the NELU
    module behaves like a regular ``nn.Module`` from installation
    onwards (no LazyModule gotchas around ``.numel()`` / ``state_dict``
    / DDP).
    """
    if isinstance(mod, nn.Conv2d):
        return mod.out_channels
    if isinstance(mod, nn.Linear):
        return mod.out_features
    return None


def _replace_with_policy(
    model: nn.Module,
    baseline_types: tuple[type, ...],
    gate_cls: type[GateNorm],
    default_axes: NormAxes | DimsLike,
    *,
    eps: float,
    gamma_init: float,
) -> int:
    """Walk the module tree; substitute baseline activations for ``gate_cls``.

    Every site receives the same ``default_axes`` (typically the
    ``"position"`` alias). The runtime rank-dispatch in
    :func:`gate_norm.layout.resolve_axes` then picks the actual axes per
    forward call: spatial for 4-D tensors, token for 3-D tensors. This
    replaces the previous per-block / per-Conv2d policy which has been
    subsumed by the unified position-axis alias.

    Two-pass walk:
      Pass 1 collects every Linear/Conv2d in module-declaration order so
      we can read the nearest-preceding linear op for each activation
      site (its ``out_*`` is the channel count the NELU will see).
      Pass 2 installs each gate, threading ``num_channels`` so per-
      channel γ_c, β_c are materialized eagerly.
    """
    linear_ops: list[nn.Module] = []
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

    n = 0
    for parent, child_name, idx in sites:
        nc: int | None = None
        if idx > 0:
            nc = _channels_of(linear_ops[idx - 1])
        try:
            new_mod = gate_cls(
                norm_axes=default_axes, eps=eps,
                gamma_init=gamma_init,
                num_channels=nc,
            )
        except TypeError:
            # Subclass doesn't accept ``num_channels`` (legacy scalar-γ
            # path: NELU_RMS, NiLU_RMS). Fall back to the original
            # signature; those modules don't need eager materialization.
            new_mod = gate_cls(
                norm_axes=default_axes, eps=eps,
                gamma_init=gamma_init,
            )
        setattr(parent, child_name, new_mod)
        n += 1
    return n


# ── Thin, type-specific wrappers ────────────────────────────────────────


def gelu_to_nelu(
    model: nn.Module,
    *,
    norm_axes: NormAxes | DimsLike = "position",
    eps: float = 1e-6,
    gamma_init: float = 1.0,
) -> int:
    """Swap every GELU instance for :class:`gate_norm.NELU`.

    Default ``norm_axes="position"`` pools over spatial axes for 4-D
    inputs (CNN) and over the token axis for 3-D inputs (Transformer).
    """
    return _replace_with_policy(
        model, GELU_TYPES, NELU, norm_axes,
        eps=eps, gamma_init=gamma_init,
    )


def silu_to_nilu(
    model: nn.Module,
    *,
    norm_axes: NormAxes | DimsLike = "position",
    eps: float = 1e-6,
    gamma_init: float = 1.0,
) -> int:
    """Swap every SiLU instance for :class:`gate_norm.NiLU`.

    Default ``norm_axes="position"`` pools over spatial axes for 4-D
    inputs (CNN) and over the token axis for 3-D inputs (Transformer).
    """
    return _replace_with_policy(
        model, SILU_TYPES, NiLU, norm_axes,
        eps=eps, gamma_init=gamma_init,
    )


def apply_gate_normalization(
    model: nn.Module,
    activation: str,
    *,
    norm_axes: NormAxes | DimsLike | None = None,
    eps: float = 1e-6,
    gamma_init: float = 1.0,
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
            gamma_init=gamma_init,
        )
    if act == "nilu":
        return silu_to_nilu(
            model, norm_axes=default_axes, eps=eps,
            gamma_init=gamma_init,
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


def replace_activation_auto_axes(
    model: nn.Module,
    baseline_types: type | tuple[type, ...],
    gate_cls: type[GateNorm],
    *,
    activation_order: str = "post",
    default_axes: NormAxes | DimsLike = "position",
    eps: float = 1e-6,
    gamma_init: float = 1.0,
) -> int:
    """Replace every baseline activation with ``gate_cls``, unified axes.

    Every activation site receives the same ``norm_axes`` (default
    ``"position"`` — pool over the position axis: spatial in CNN, token
    in Transformer; rank-dispatched at runtime by
    :func:`gate_norm.layout.resolve_axes`).

    The ``activation_order`` argument and the conv-shape introspection
    that selected axes per-site are no longer needed: the position-axis
    alias makes the choice automatic from tensor rank. Both kept for
    API stability — callers can still pass an explicit ``default_axes``
    tuple to override.
    """
    if isinstance(baseline_types, type):
        baseline_types = (baseline_types,)
    if activation_order not in ("pre", "post"):
        raise ValueError(f"activation_order must be 'pre' or 'post'")

    # Two-pass: collect Linear/Conv ops in declaration order so we can
    # read the nearest-preceding (post-act) or nearest-following (pre-act)
    # linear op's ``out_*`` channel count for each activation site.
    linear_ops: list[nn.Module] = []
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

    n = 0
    for parent, name, idx in sites:
        nc: int | None = None
        if activation_order == "post":
            if idx > 0:
                nc = _channels_of(linear_ops[idx - 1])
        else:  # "pre"
            if idx < len(linear_ops):
                nc = _channels_of(linear_ops[idx])
        try:
            new_mod = gate_cls(
                norm_axes=default_axes, eps=eps,
                gamma_init=gamma_init,
                num_channels=nc,
            )
        except TypeError:
            # Legacy scalar-γ subclass that doesn't accept num_channels.
            new_mod = gate_cls(
                norm_axes=default_axes, eps=eps,
                gamma_init=gamma_init,
            )
        setattr(parent, name, new_mod)
        n += 1
    return n
