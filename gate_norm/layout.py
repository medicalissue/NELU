"""Normalization-axis canonicalization and permute/flatten utilities.

Fused kernels reduce over the trailing dimension only. This module does the
bookkeeping that lets us present an architecture-friendly API (``"channel"``,
``"sample"``, or any tuple of axes) on top of that kernel contract.

Two services:

1. :func:`resolve_axes` — translate a human-facing spec (string alias or
   explicit tuple) into a canonical, sorted tuple of non-negative axes.
2. :func:`flatten_for_reduction` / :func:`restore` — permute and reshape so
   the chosen reduction axes collapse into a single trailing axis, with an
   inverse that recovers the original layout.

Aliases:

* ``"position"`` — **the position axis** of the tensor, dispatched by rank:
    - 4-D ``(B, C, H, W)``  → ``(2, 3)`` (spatial)        — CNN
    - 3-D ``(B, T, C)``     → ``(1,)``  (token)           — Transformer
    - 2-D ``(B, C)``        → ``(1,)``  (degenerate fallback)
  Unified policy: pool over the position axis to get a per-channel
  statistic, then broadcast it back across positions. Same primitive in
  CNN and Transformer; only the axis index differs.
* ``"channel"`` — last axis only. Matches channel-mixing linear operations
  (transformer FFNs, ConvNeXt depthwise→pointwise, channels-last activations
  at ``(B, D)`` / ``(B, L, D)`` / ``(B, H, W, D)``).
* ``"sample"``  — last three axes. Matches operations that mix across both
  channel and spatial axes before the activation; valid for NCHW tensors
  ``(B, C, H, W)``.

For anything in between (e.g. ``(2, 3)`` after a depthwise convolution that
mixes only the spatial axes) pass the axis tuple directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch


DimsLike = int | Iterable[int]
NormAxes = Literal["position", "channel", "sample"]

# Static aliases: tuple of axes (negative for "from end").
_ALIASES: dict[str, tuple[int, ...]] = {
    "channel": (-1,),
    "sample":  (-3, -2, -1),
}

# Rank-dispatched aliases: ndim → axis tuple. Resolved at call time.
# "position": pool over the position axis (spatial in CNN, token in
# Transformer), keeping per-channel statistics.
_RANK_ALIASES: dict[str, dict[int, tuple[int, ...]]] = {
    "position": {
        2: (1,),         # (B, C)        — degenerate fallback
        3: (1,),         # (B, T, C)     — Transformer: pool over tokens
        4: (2, 3),       # (B, C, H, W)  — CNN: pool over spatial
    },
}


def resolve_axes(ndim: int, spec: NormAxes | DimsLike) -> tuple[int, ...]:
    """Canonicalize an axis spec for a tensor of rank ``ndim``.

    Returns a sorted tuple of non-negative axes. Raises on empty, duplicate,
    or out-of-range axes and on string aliases incompatible with ``ndim``.
    """
    if isinstance(spec, str):
        if spec in _RANK_ALIASES:
            table = _RANK_ALIASES[spec]
            if ndim not in table:
                raise ValueError(
                    f"alias {spec!r} only supports ndim in {sorted(table)}, "
                    f"got ndim={ndim}"
                )
            raw = table[ndim]
        elif spec in _ALIASES:
            raw = _ALIASES[spec]
            if ndim < len(raw):
                raise ValueError(
                    f"alias {spec!r} needs ndim >= {len(raw)}, got ndim={ndim}"
                )
        else:
            valid = sorted(set(_ALIASES) | set(_RANK_ALIASES))
            raise ValueError(
                f"unknown norm-axes alias {spec!r}; valid: {valid} "
                "or an explicit axis tuple"
            )
    elif isinstance(spec, int):
        raw = (spec,)
    else:
        raw = tuple(int(a) for a in spec)

    if not raw:
        raise ValueError("reduction axes must be non-empty")

    canonical = []
    for a in raw:
        if a < 0:
            a += ndim
        if a < 0 or a >= ndim:
            raise IndexError(f"axis {a} out of range for ndim={ndim}")
        canonical.append(a)

    if len(set(canonical)) != len(canonical):
        raise ValueError(f"reduction axes must be unique, got {raw}")

    return tuple(sorted(canonical))


def reduction_numel(shape: tuple[int, ...], axes: tuple[int, ...]) -> int:
    return math.prod(shape[a] for a in axes)


@dataclass(frozen=True)
class ReductionLayout:
    """Bookkeeping for a permute + flatten pair."""

    permute: tuple[int, ...]
    inverse_permute: tuple[int, ...]
    permuted_shape: tuple[int, ...]


def flatten_for_reduction(
    z: torch.Tensor, axes: tuple[int, ...]
) -> tuple[torch.Tensor, ReductionLayout]:
    """Permute ``axes`` to the end and flatten them into a single trailing axis."""
    keep = tuple(a for a in range(z.ndim) if a not in axes)
    permute = keep + axes

    inverse = [0] * z.ndim
    for new, old in enumerate(permute):
        inverse[old] = new

    z_perm = z if permute == tuple(range(z.ndim)) else z.permute(permute)
    flat_shape = tuple(z.size(a) for a in keep) + (
        reduction_numel(tuple(z.shape), axes),
    )

    return (
        z_perm.reshape(flat_shape),
        ReductionLayout(
            permute=permute,
            inverse_permute=tuple(inverse),
            permuted_shape=tuple(z_perm.shape),
        ),
    )


def restore(z_flat: torch.Tensor, layout: ReductionLayout) -> torch.Tensor:
    """Inverse of :func:`flatten_for_reduction`."""
    z_perm = z_flat.reshape(layout.permuted_shape)
    if layout.permute == tuple(range(len(layout.permute))):
        return z_perm
    return z_perm.permute(layout.inverse_permute)
