import torch

from nelu.activations import nelu, nilu
from nelu.reduction_layout import (
    canonicalize_reduce_dims,
    flatten_reduction_dims,
    restore_reduction_dims,
)


def test_canonicalize_reduce_dims():
    assert canonicalize_reduce_dims(4, -1) == (3,)
    assert canonicalize_reduce_dims(4, (-3, -2, -1)) == (1, 2, 3)
    assert canonicalize_reduce_dims(5, (3, 1)) == (1, 3)


def test_flatten_restore_suffix_dims_roundtrip():
    z = torch.randn(2, 3, 4, 5)
    z_flat, layout = flatten_reduction_dims(z, (-3, -2, -1))
    assert z_flat.shape == (2, 60)
    z_restored = restore_reduction_dims(z_flat, layout)
    assert torch.equal(z_restored, z)


def test_flatten_restore_nonsuffix_dims_roundtrip():
    z = torch.randn(2, 3, 4, 5)
    z_flat, layout = flatten_reduction_dims(z, (1, 3))
    assert z_flat.shape == (2, 4, 15)
    z_restored = restore_reduction_dims(z_flat, layout)
    assert torch.equal(z_restored, z)


def test_nelu_functional_dims_matches_manual():
    z = torch.randn(2, 3, 4, 5)
    rho = z.pow(2).mean(dim=(-3, -2, -1), keepdim=True).add(1e-6).sqrt()
    expected = z * 0.5 * (1.0 + torch.erf((z / rho) * (1.0 / torch.sqrt(torch.tensor(2.0)))))
    actual = nelu(z, dims=(-3, -2, -1))
    torch.testing.assert_close(actual, expected)


def test_nilu_functional_dims_matches_manual():
    z = torch.randn(2, 7, 11)
    rho = z.pow(2).mean(dim=(-1,), keepdim=True).add(1e-6).sqrt()
    expected = z * torch.sigmoid(z / rho)
    actual = nilu(z, dims=-1)
    torch.testing.assert_close(actual, expected)
