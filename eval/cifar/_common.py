"""Shared loaders + feature extraction for the CIFAR repr-quality probes.

Every probe follows the same skeleton: build the same model architecture
the checkpoint was trained with, restore weights, optionally turn the
classifier head off and harvest the penultimate feature instead. The
feature extractor handles the seven CIFAR architectures uniformly by
swapping the final ``Linear`` for ``nn.Identity``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from eval.cifar_robustness import (
    CIFAR100C, CORRUPTIONS, SEVERITIES, load_checkpoint,
)
from train.cifar import build_model, CIFAR100_MEAN, CIFAR100_STD


__all__ = [
    "ACTIVATIONS", "MODELS", "build_eval_model", "build_feature_extractor",
    "build_loader", "build_corruption_loader", "load_checkpoint",
    "CIFAR100C", "CORRUPTIONS", "SEVERITIES",
    "add_common_args", "device_from_args",
]


ACTIVATIONS = ("relu", "gelu", "silu", "nelu", "nilu",
               "nelu_ln", "nilu_ln",
               "nelu_aff", "nilu_aff",
               "nelu_affcw", "nilu_affcw")
MODELS = (
    "resnet20", "resnet56", "resnet110", "vgg16_bn",
    "shufflenetv2", "mobilenetv2", "densenet_bc_100_12",
)


# ── Model construction ─────────────────────────────────────────────────

def build_eval_model(
    name: str, activation: str, checkpoint: str, device: str,
) -> nn.Module:
    """Build the model, load weights, move to device, set eval()."""
    model = build_model(name, activation=activation, num_classes=100)
    # Channel-wise affine variants need a dummy forward to materialize
    # γ_c, β_c before load_state_dict can map the saved tensors back.
    if activation in ("nelu_affcw", "nilu_affcw"):
        with torch.no_grad():
            _ = model(torch.zeros(2, 3, 32, 32))
    load_checkpoint(model, checkpoint)
    return model.to(device).eval()


def _replace_head(model: nn.Module) -> int:
    """Replace the final ``Linear`` layer with ``nn.Identity`` in-place.

    Returns the in-features of the replaced head so callers know the
    feature dimensionality. The seven CIFAR architectures all expose
    the head under one of these attribute names; we walk them in turn.
    """
    for attr in ("fc", "classifier", "linear", "head"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Linear):
            in_features = head.in_features
            setattr(model, attr, nn.Identity())
            return in_features
        if isinstance(head, nn.Sequential):
            # mobilenetv2 / vgg style: classifier is a Sequential ending
            # in a Linear. Replace the last Linear with Identity.
            for i in range(len(head) - 1, -1, -1):
                if isinstance(head[i], nn.Linear):
                    in_features = head[i].in_features
                    head[i] = nn.Identity()
                    return in_features
    raise RuntimeError(
        "could not find a Linear classifier head on the model; "
        "extend _replace_head for this architecture"
    )


def build_feature_extractor(
    name: str, activation: str, checkpoint: str, device: str,
) -> tuple[nn.Module, int]:
    """Return ``(model, feature_dim)``. Forward yields penultimate features."""
    model = build_eval_model(name, activation, checkpoint, device)
    feat_dim = _replace_head(model)
    return model, feat_dim


# ── Data loaders ───────────────────────────────────────────────────────

def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _train_eval_transform() -> transforms.Compose:
    """Deterministic eval transform applied to the *train* split.

    Linear probes / kNN want to see the train-split images without
    augmentation jitter so the probe sees the same sample distribution
    the model was scored on at validation time.
    """
    return _eval_transform()


def build_loader(
    data_dir: str, *, train: bool, batch_size: int = 256, workers: int = 4,
    augment: bool = False,
) -> DataLoader:
    if augment and train:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        tf = _eval_transform()
    ds = datasets.CIFAR100(data_dir, train=train, download=False, transform=tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )


def build_corruption_loader(
    data_root: str, corruption: str, severity: int, *,
    batch_size: int = 256, workers: int = 4,
) -> DataLoader:
    root = str(Path(data_root) / "CIFAR-100-C")
    ds = CIFAR100C(root, corruption, severity)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )


# ── Argument parsing ───────────────────────────────────────────────────

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--activation", required=True, choices=ACTIVATIONS)
    p.add_argument("--checkpoint", required=True,
                   help="path to a CIFAR-100 checkpoint (.pt)")
    p.add_argument("--data-root", default="/data",
                   help="parent directory with cifar-100-python/ and CIFAR-100-C/")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)


def device_from_args(args: argparse.Namespace) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Feature dump helper used by knn / geometry / cka / linear_probe ────

@torch.no_grad()
def dump_features(
    model: nn.Module, loader: DataLoader, device: str,
    *, max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` over ``loader`` and stack penultimate features + labels."""
    model.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model(x)
        if f.dim() > 2:
            f = torch.flatten(f, 1)
        feats.append(f.cpu())
        labels.append(y)
        seen += y.size(0)
        if max_samples is not None and seen >= max_samples:
            break
    F = torch.cat(feats, dim=0)
    L = torch.cat(labels, dim=0)
    if max_samples is not None:
        F = F[:max_samples]
        L = L[:max_samples]
    return F, L
