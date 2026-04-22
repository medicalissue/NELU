"""Tests for :mod:`eval.cifar_robustness`.

We fabricate a stand-in CIFAR-100-C directory with the authentic shape
layout (``(50000, 32, 32, 3)`` plus ``labels.npy``) and verify that
severity slicing, dtype normalization, and the constants the evaluation
loop depends on are all correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eval.cifar_robustness import CIFAR100C, CORRUPTIONS, SEVERITIES


_N_PER_SEVERITY = 10_000


@pytest.fixture()
def fake_cifar_c(tmp_path: Path) -> Path:
    """Create a stand-in CIFAR-100-C folder with one .npy + labels.npy."""
    n = _N_PER_SEVERITY * len(SEVERITIES)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 100, size=(n,), dtype=np.int64)
    root = tmp_path / "CIFAR-100-C"
    root.mkdir()
    np.save(root / "gaussian_noise.npy", images)
    np.save(root / "labels.npy", labels)
    return root


def test_cifar_c_dataset_length_per_severity(fake_cifar_c: Path) -> None:
    ds = CIFAR100C(str(fake_cifar_c), "gaussian_noise", severity=3)
    assert len(ds) == _N_PER_SEVERITY


def test_cifar_c_dataset_tensor_shape(fake_cifar_c: Path) -> None:
    ds = CIFAR100C(str(fake_cifar_c), "gaussian_noise", severity=1)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert 0 <= int(label) < 100


def test_cifar_c_rejects_invalid_severity(fake_cifar_c: Path) -> None:
    with pytest.raises(ValueError):
        CIFAR100C(str(fake_cifar_c), "gaussian_noise", severity=0)
    with pytest.raises(ValueError):
        CIFAR100C(str(fake_cifar_c), "gaussian_noise", severity=6)


def test_cifar_c_rejects_unknown_corruption(fake_cifar_c: Path) -> None:
    with pytest.raises(ValueError):
        CIFAR100C(str(fake_cifar_c), "nonexistent_corruption", severity=1)


def test_corruption_list_has_19_entries() -> None:
    assert len(CORRUPTIONS) == 19
    assert len(set(CORRUPTIONS)) == 19
