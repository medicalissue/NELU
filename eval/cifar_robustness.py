"""CIFAR-100-C corruption evaluation.

Hendrycks & Dietterich release CIFAR-100-C as 19 corruption types, each
stored as a single ``.npy`` file of shape ``(50000, 32, 32, 3)`` uint8.
The 50 000 examples are the CIFAR-100 validation set rendered at all five
severity levels and concatenated — severity *s* occupies indices
``(s-1)·10000 : s·10000``. A shared ``labels.npy`` holds the ground-truth
labels (identical 10 000 labels repeated five times).

The dataset root is expected to contain the unpacked tarball::

    data-root/CIFAR-100-C/
        brightness.npy  contrast.npy  ... (19 corruption .npy files)
        labels.npy

Only the validation set is evaluated; no training is performed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train.cifar import build_model
from train.cifar import CIFAR100_MEAN, CIFAR100_STD


# The 19 CIFAR-100-C corruptions in the canonical order used by Hendrycks.
CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression", "speckle_noise", "gaussian_blur",
    "spatter", "saturate",
)

SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)
_PER_SEVERITY = 10_000


class CIFAR100C(Dataset):
    """CIFAR-100-C as a PyTorch Dataset, sliced to one severity.

    Images in the .npy files are uint8 HWC; we convert to CHW float tensors
    and apply the same normalization used for CIFAR-100 training.
    """

    def __init__(self, root: str, corruption: str, severity: int):
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption {corruption!r}")
        if severity not in SEVERITIES:
            raise ValueError(f"Severity must be 1..5, got {severity}")
        root_path = Path(root)
        images = np.load(root_path / f"{corruption}.npy", mmap_mode="r")
        labels = np.load(root_path / "labels.npy", mmap_mode="r")
        start = (severity - 1) * _PER_SEVERITY
        stop = severity * _PER_SEVERITY
        self.images = images[start:stop]
        self.labels = labels[start:stop].astype(np.int64)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # ``np.asarray`` with ``mmap_mode`` returns a read-only view; copy
        # so the downstream torchvision transform can hand us a writable
        # tensor without triggering a warning.
        img = np.asarray(self.images[idx]).copy()
        return self.transform(img), int(self.labels[idx])


# ── Checkpoint loading ──────────────────────────────────────────────────


def load_checkpoint(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] warning: missing keys: {missing[:4]}...")
    if unexpected:
        print(f"[eval] warning: unexpected keys: {unexpected[:4]}...")
    epoch = ckpt.get("epoch", "?")
    print(f"[eval] loaded checkpoint {path} (epoch={epoch})")


# ── Evaluation loop ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_one(
    model: nn.Module, root: str, corruption: str, severity: int,
    batch_size: int, workers: int, device: str,
) -> float:
    dataset = CIFAR100C(root, corruption, severity)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_cifar_c(
    model: nn.Module, root: str, batch_size: int, workers: int, device: str,
) -> dict:
    out: dict = {}
    for corruption in CORRUPTIONS:
        per_severity: list[float] = []
        for s in SEVERITIES:
            acc = evaluate_one(
                model, root, corruption, s, batch_size, workers, device,
            )
            per_severity.append(acc)
        mean = sum(per_severity) / len(per_severity)
        out[corruption] = {"per_severity": per_severity, "mean": mean}
        print(f"  {corruption:<25s} {mean:6.2f}%")
    means = [v["mean"] for v in out.values()]
    out["_mean"] = sum(means) / len(means)
    print(f"  {'MEAN':<25s} {out['_mean']:6.2f}%")
    return out


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-100-C robustness evaluation")
    p.add_argument("--model", default="resnet20",
                   help="One of the CIFAR models registered in train.cifar.")
    p.add_argument("--activation", default="relu",
                   choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="/data",
                   help="Parent directory of CIFAR-100-C/.")
    p.add_argument("--output", default="results/cifar_c.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args.model, activation=args.activation, num_classes=100)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)

    root = Path(args.data_root) / "CIFAR-100-C"
    if not root.is_dir():
        raise FileNotFoundError(f"CIFAR-100-C directory not found at {root}")

    print(f"[eval] model={args.model}, activation={args.activation}")
    print(f"[eval] CIFAR-100-C root={root}\n── CIFAR-100-C ──")
    t0 = time.time()
    c = evaluate_cifar_c(
        model, str(root), args.batch_size, args.workers, device,
    )

    results = {
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "cifar_100_c": c,
        "seconds": time.time() - t0,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[eval] results saved to {out}")


if __name__ == "__main__":
    main()
