"""Reduction-axis utilities for Gate Normalization.

The fused CUDA kernels reduce over the trailing dimension only. This module
provides two services:

1. `rms_axes` — translate a human-facing spec (string alias or explicit tuple)
   into a canonical, sorted tuple of non-negative axes for a given tensor rank.
2. `flatten_for_reduction` / `restore` — permute and reshape a tensor so that
   the chosen reduction axes collapse into a single trailing axis, with an
   inverse op to restore the original layout.

Two string aliases cover every architecture we care about:

* `"per_token"`  → last axis only. Valid for `(B, D)`, `(B, L, D)`,
  channels-last `(B, H, W, D)`.
* `"per_sample"` → last three axes. Valid for NCHW convolutions
  `(B, C, H, W)`.

Any tuple of axes may also be passed explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch


DimsLike = int | Iterable[int]
RmsMode = Literal["per_token", "per_sample"]

_RMS_ALIASES: dict[str, tuple[int, ...]] = {
    "per_token": (-1,),
    "per_sample": (-3, -2, -1),
}


def rms_axes(ndim: int, spec: RmsMode | DimsLike) -> tuple[int, ...]:
    """Canonicalize a reduction-axis spec for a tensor of rank `ndim`.

    Returns a sorted tuple of non-negative axes. Raises on empty, duplicate,
    or out-of-range axes and when a string alias is incompatible with `ndim`.
    """
    if isinstance(spec, str):
        if spec not in _RMS_ALIASES:
            raise ValueError(
                f"Unknown rms mode {spec!r}. Valid: {sorted(_RMS_ALIASES)} "
                "or an explicit axis tuple."
            )
        raw = _RMS_ALIASES[spec]
        if ndim < len(raw):
            raise ValueError(
                f"rms mode {spec!r} needs ndim >= {len(raw)}, got ndim={ndim}"
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
    """Bookkeeping for a permute + flatten pair.

    `permute`/`inverse_permute` are the axis orderings used to move the
    reduction axes to the trailing position; `permuted_shape` is the shape of
    the permuted tensor before flattening.
    """

    permute: tuple[int, ...]
    inverse_permute: tuple[int, ...]
    permuted_shape: tuple[int, ...]


def flatten_for_reduction(
    z: torch.Tensor, axes: tuple[int, ...]
) -> tuple[torch.Tensor, ReductionLayout]:
    """Permute `axes` to the end and flatten them into a single trailing axis."""
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
    """Inverse of `flatten_for_reduction`."""
    z_perm = z_flat.reshape(layout.permuted_shape)
    if layout.permute == tuple(range(len(layout.permute))):
        return z_perm
    return z_perm.permute(layout.inverse_permute)


def rms(
    z: torch.Tensor, axes: tuple[int, ...], eps: float = 1e-6
) -> torch.Tensor:
    """Root-mean-square over `axes`, keeping dims for broadcast."""
    return z.pow(2).mean(dim=axes, keepdim=True).add(eps).sqrt()
