"""Tests for :mod:`gate_norm.reduction`."""

from __future__ import annotations

import pytest
import torch

from gate_norm.reduction import (
    flatten_for_reduction,
    reduction_numel,
    restore,
    rms,
    rms_axes,
)


def test_rms_axes_per_token() -> None:
    assert rms_axes(2, "per_token") == (1,)
    assert rms_axes(3, "per_token") == (2,)
    assert rms_axes(4, "per_token") == (3,)


def test_rms_axes_per_sample_requires_ndim() -> None:
    assert rms_axes(4, "per_sample") == (1, 2, 3)
    with pytest.raises(ValueError):
        rms_axes(2, "per_sample")


def test_rms_axes_explicit_tuple_is_canonicalized() -> None:
    assert rms_axes(4, (-1, -2)) == (2, 3)
    assert rms_axes(3, 1) == (1,)


def test_rms_axes_rejects_duplicates_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        rms_axes(3, (1, 1))
    with pytest.raises(IndexError):
        rms_axes(3, 5)
    with pytest.raises(ValueError):
        rms_axes(3, ())


def test_flatten_restore_roundtrip() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 8)
    axes = rms_axes(x.ndim, "per_sample")
    flat, layout = flatten_for_reduction(x, axes)
    assert flat.shape == (2, 4 * 6 * 8)
    recovered = restore(flat, layout)
    assert recovered.shape == x.shape
    # The full permute/reshape pair should be the identity.
    assert torch.equal(recovered, x)


def test_flatten_restore_with_interior_axis() -> None:
    x = torch.randn(3, 5, 7)
    axes = rms_axes(x.ndim, (1,))
    flat, layout = flatten_for_reduction(x, axes)
    assert flat.shape == (3, 7, 5)
    recovered = restore(flat, layout)
    assert torch.equal(recovered, x)


def test_reduction_numel() -> None:
    assert reduction_numel((2, 3, 4), (1, 2)) == 12
    assert reduction_numel((5, 5, 5), (0,)) == 5


def test_rms_matches_manual() -> None:
    x = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    r = rms(x, (1,), eps=0.0)
    expected = torch.tensor([[5.0 / (3 ** 0.5)], [6.0 / (3 ** 0.5)]])
    assert torch.allclose(r, expected, atol=1e-6)
