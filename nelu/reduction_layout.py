"""Utilities for flattening arbitrary reduction dims into a kernel-friendly suffix.

The fused CUDA kernels operate on tensors whose normalized axis is the last
dimension. This helper lets higher-level code expose a more general `dims=...`
interface by reordering / flattening the requested reduction dims into a single
trailing axis and then restoring the original layout afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


DimsLike = int | Iterable[int]


@dataclass(frozen=True)
class ReductionLayout:
    permute: tuple[int, ...]
    inverse_permute: tuple[int, ...]
    permuted_shape: tuple[int, ...]


def canonicalize_reduce_dims(ndim: int, dims: DimsLike) -> tuple[int, ...]:
    if isinstance(dims, int):
        dims = (dims,)
    else:
        dims = tuple(int(dim) for dim in dims)

    if not dims:
        raise ValueError("dims must contain at least one axis")

    canonical = []
    for dim in dims:
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise IndexError(f"dim {dim} out of range for ndim={ndim}")
        canonical.append(dim)

    if len(set(canonical)) != len(canonical):
        raise ValueError(f"dims must be unique, got {dims}")

    return tuple(sorted(canonical))


def reduction_size(shape: tuple[int, ...], dims: DimsLike) -> int:
    dims = canonicalize_reduce_dims(len(shape), dims)
    return math.prod(shape[dim] for dim in dims)


def flatten_reduction_dims(
    z: torch.Tensor,
    dims: DimsLike,
) -> tuple[torch.Tensor, ReductionLayout]:
    dims = canonicalize_reduce_dims(z.ndim, dims)
    keep_dims = tuple(dim for dim in range(z.ndim) if dim not in dims)
    permute = keep_dims + dims

    inverse = [0] * len(permute)
    for new_idx, old_idx in enumerate(permute):
        inverse[old_idx] = new_idx
    inverse_permute = tuple(inverse)

    if permute == tuple(range(z.ndim)):
        z_perm = z
    else:
        z_perm = z.permute(permute)

    reduced = reduction_size(tuple(z.shape), dims)
    flat_shape = tuple(z.size(dim) for dim in keep_dims) + (reduced,)
    z_flat = z_perm.reshape(flat_shape)

    layout = ReductionLayout(
        permute=permute,
        inverse_permute=inverse_permute,
        permuted_shape=tuple(z_perm.shape),
    )
    return z_flat, layout


def restore_reduction_dims(z_flat: torch.Tensor, layout: ReductionLayout) -> torch.Tensor:
    z_perm = z_flat.reshape(layout.permuted_shape)
    if layout.permute == tuple(range(len(layout.permute))):
        return z_perm
    return z_perm.permute(layout.inverse_permute)
