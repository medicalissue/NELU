#!/usr/bin/env python3
"""Validate or reorganize ImageNet-1k validation images into ImageFolder layout.

Expected final layout:
  imagenet/
    train/<synset>/*.JPEG
    val/<synset>/*.JPEG

If `val/` is flat (50,000 files at the root named `ILSVRC2012_val_*.JPEG`),
this script can reorganize it using a 50,000-line synset label file where each
line contains the target synset for the corresponding validation image.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


DEFAULT_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="ImageNet train directory")
    parser.add_argument("--val-dir", required=True, help="ImageNet val directory")
    parser.add_argument(
        "--synset-labels",
        default="",
        help="Path to a 50,000-line validation synset labels file",
    )
    parser.add_argument(
        "--labels-url",
        default=DEFAULT_LABELS_URL,
        help="Fallback URL to download validation synset labels from",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/nelu_imagenet",
        help="Directory to cache downloaded label files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate layout only; do not modify val/",
    )
    return parser.parse_args()


def list_dirs(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if p.is_dir())


def list_root_images(path: Path) -> list[Path]:
    exts = {".jpeg", ".jpg", ".JPEG", ".JPG"}
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix in exts
    )


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"Validation labels file is empty: {path}")
    return labels


def resolve_labels_file(args: argparse.Namespace) -> Path:
    if args.synset_labels:
        path = Path(args.synset_labels)
        if not path.is_file():
            raise RuntimeError(f"Validation labels file not found: {path}")
        return path

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "imagenet_2012_validation_synset_labels.txt"
    if not cached.is_file():
        if not args.labels_url:
            raise RuntimeError(
                "Validation labels file is required to reorganize flat val/, "
                "but neither --synset-labels nor --labels-url was provided."
            )
        print(f"Downloading validation synset labels from {args.labels_url}", flush=True)
        with urllib.request.urlopen(args.labels_url, timeout=60) as response:
            cached.write_bytes(response.read())
    return cached


def validate_layout(train_dir: Path, val_dir: Path) -> None:
    train_dirs = list_dirs(train_dir)
    val_dirs = list_dirs(val_dir)
    val_root_images = list_root_images(val_dir)

    if not train_dirs:
        raise RuntimeError(f"Train directory has no class subdirectories: {train_dir}")
    if val_root_images:
        raise RuntimeError(
            f"Validation directory is still flat: found {len(val_root_images)} root images in {val_dir}"
        )
    if not val_dirs:
        raise RuntimeError(f"Validation directory has no class subdirectories: {val_dir}")

    train_set = set(train_dirs)
    val_set = set(val_dirs)
    if train_set != val_set:
        only_train = sorted(train_set - val_set)[:10]
        only_val = sorted(val_set - train_set)[:10]
        raise RuntimeError(
            "Train/val class directories do not match.\n"
            f"  train-only sample: {only_train}\n"
            f"  val-only sample: {only_val}"
        )

    val_total = sum(sum(1 for p in (val_dir / d).iterdir() if p.is_file()) for d in val_dirs)
    if val_total != 50000:
        raise RuntimeError(
            f"Validation directory should contain 50000 images, found {val_total} in {val_dir}"
        )


def reorganize_flat_val(train_dir: Path, val_dir: Path, labels: list[str]) -> None:
    root_images = list_root_images(val_dir)
    if not root_images:
        raise RuntimeError(f"No flat validation images found in {val_dir}")
    if len(root_images) != len(labels):
        raise RuntimeError(
            f"Validation image count ({len(root_images)}) does not match labels count ({len(labels)})"
        )

    train_dirs = list_dirs(train_dir)
    train_set = set(train_dirs)
    label_set = set(labels)
    if train_set != label_set:
        only_train = sorted(train_set - label_set)[:10]
        only_labels = sorted(label_set - train_set)[:10]
        raise RuntimeError(
            "Validation synset labels do not match train classes.\n"
            f"  train-only sample: {only_train}\n"
            f"  labels-only sample: {only_labels}"
        )

    for synset in sorted(label_set):
        (val_dir / synset).mkdir(parents=True, exist_ok=True)

    for image_path, synset in zip(root_images, labels):
        shutil.move(str(image_path), str(val_dir / synset / image_path.name))


def main() -> int:
    args = parse_args()
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)

    if not train_dir.is_dir():
        raise RuntimeError(f"Train directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise RuntimeError(f"Val directory not found: {val_dir}")

    if args.check_only:
        validate_layout(train_dir, val_dir)
        print("ImageNet val layout OK")
        return 0

    val_root_images = list_root_images(val_dir)
    if val_root_images:
        labels_file = resolve_labels_file(args)
        labels = read_labels(labels_file)
        print(
            f"Reorganizing flat ImageNet val directory using labels from {labels_file}",
            flush=True,
        )
        reorganize_flat_val(train_dir, val_dir, labels)

    validate_layout(train_dir, val_dir)
    print("ImageNet val layout ready")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
