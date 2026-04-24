"""Tests for :mod:`gate_norm.layout`."""

from __future__ import annotations

import pytest
import torch

from gate_norm.layout import (
    flatten_for_reduction,
    reduction_numel,
    resolve_axes,
    restore,
)


def test_resolve_axes_channel_alias() -> None:
    assert resolve_axes(2, "channel") == (1,)
    assert resolve_axes(3, "channel") == (2,)
    assert resolve_axes(4, "channel") == (3,)


def test_resolve_axes_sample_alias_requires_ndim() -> None:
    assert resolve_axes(4, "sample") == (1, 2, 3)
    with pytest.raises(ValueError):
        resolve_axes(2, "sample")


def test_resolve_axes_explicit_tuple_is_canonicalized() -> None:
    assert resolve_axes(4, (-1, -2)) == (2, 3)
    assert resolve_axes(3, 1) == (1,)


def test_resolve_axes_rejects_duplicates_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        resolve_axes(3, (1, 1))
    with pytest.raises(IndexError):
        resolve_axes(3, 5)
    with pytest.raises(ValueError):
        resolve_axes(3, ())


def test_resolve_axes_rejects_unknown_alias() -> None:
    # "per_token"/"per_sample" are legacy aliases; they must no longer
    # resolve, so callers that missed the rename fail loudly.
    with pytest.raises(ValueError):
        resolve_axes(4, "per_token")


def test_flatten_restore_roundtrip() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 8)
    axes = resolve_axes(x.ndim, "sample")
    flat, layout = flatten_for_reduction(x, axes)
    assert flat.shape == (2, 4 * 6 * 8)
    recovered = restore(flat, layout)
    assert recovered.shape == x.shape
    assert torch.equal(recovered, x)


def test_flatten_restore_with_interior_axis() -> None:
    x = torch.randn(3, 5, 7)
    axes = resolve_axes(x.ndim, (1,))
    flat, layout = flatten_for_reduction(x, axes)
    assert flat.shape == (3, 7, 5)
    recovered = restore(flat, layout)
    assert torch.equal(recovered, x)


def test_reduction_numel() -> None:
    assert reduction_numel((2, 3, 4), (1, 2)) == 12
    assert reduction_numel((5, 5, 5), (0,)) == 5
