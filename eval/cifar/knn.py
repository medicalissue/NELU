"""k-NN probe on penultimate features.

For each ``k`` in ``--ks``, fit a cosine-similarity kNN classifier on the
CIFAR-100 train split's L2-normalized features and report top-1 accuracy
on the test split. Pure inference, no training. The standard SSL eval
metric (DINO / MoCo) for representation quality.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from eval.cifar._common import (
    add_common_args, build_feature_extractor, build_loader,
    device_from_args, dump_features,
)


@torch.no_grad()
def knn_classify(
    train_feats: torch.Tensor, train_labels: torch.Tensor,
    test_feats: torch.Tensor, test_labels: torch.Tensor,
    *, k: int, num_classes: int = 100, temperature: float = 0.07,
    chunk: int = 512, device: str = "cuda",
) -> float:
    """Cosine-similarity kNN with softmax-weighted voting.

    Mirrors DINO's eval recipe — temperature-softmaxed similarity over the
    top-k neighbours, then sum-by-class. Returns top-1 accuracy in %.
    """
    train_feats = F.normalize(train_feats.to(device), dim=1)
    test_feats = F.normalize(test_feats.to(device), dim=1)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    correct = 0
    total = test_feats.size(0)
    one_hot = F.one_hot(train_labels, num_classes).float()  # (N_train, C)
    for i in range(0, total, chunk):
        q = test_feats[i:i + chunk]                          # (B, D)
        sims = q @ train_feats.t()                           # (B, N_train)
        topk_sims, topk_idx = sims.topk(k, dim=1)
        weights = (topk_sims / temperature).softmax(dim=1)   # (B, k)
        votes = one_hot[topk_idx]                            # (B, k, C)
        scores = (votes * weights.unsqueeze(-1)).sum(dim=1)  # (B, C)
        pred = scores.argmax(dim=1)
        correct += (pred == test_labels[i:i + chunk]).sum().item()
    return 100.0 * correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="k-NN feature probe")
    add_common_args(p)
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])
    args = p.parse_args()
    device = device_from_args(args)

    model, feat_dim = build_feature_extractor(
        args.model, args.activation, args.checkpoint, device,
    )

    train_loader = build_loader(
        args.data_root, train=True,
        batch_size=args.batch_size, workers=args.workers, augment=False,
    )
    test_loader = build_loader(
        args.data_root, train=False,
        batch_size=args.batch_size, workers=args.workers,
    )

    t0 = time.time()
    print(f"[knn] dumping features (dim={feat_dim})...")
    train_feats, train_labels = dump_features(model, train_loader, device)
    test_feats, test_labels = dump_features(model, test_loader, device)
    dump_t = time.time() - t0

    accs: dict[str, float] = {}
    for k in args.ks:
        acc = knn_classify(
            train_feats, train_labels, test_feats, test_labels,
            k=k, device=device,
        )
        accs[f"k{k}"] = acc
        print(f"  k={k:>3d}  acc={acc:6.2f}%")

    results = {
        "probe": "knn",
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "feature_dim": feat_dim,
        "n_train": int(train_feats.size(0)),
        "n_test": int(test_feats.size(0)),
        "knn_top1": accs,
        "dump_seconds": dump_t,
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[knn] → {args.output}")


if __name__ == "__main__":
    main()
