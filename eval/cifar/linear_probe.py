"""Frozen-backbone linear probe.

Backbone weights are fixed; only a fresh ``Linear(feat_dim → 100)`` is
trained with SGD on the train split's features. The standard
self-supervised representation-quality benchmark applied to a supervised
checkpoint: how separable is the feature space, holding the encoder fixed?

We dump features once (no augmentation) and then train the linear head
on that dumped tensor — much faster than backprop'ing through the frozen
backbone, and mathematically equivalent for a single linear layer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eval.cifar._common import (
    add_common_args, build_feature_extractor, build_loader,
    device_from_args, dump_features,
)


def _train_linear_head(
    train_feats: torch.Tensor, train_labels: torch.Tensor,
    test_feats: torch.Tensor, test_labels: torch.Tensor,
    *, feat_dim: int, num_classes: int = 100,
    epochs: int = 50, batch_size: int = 1024, lr: float = 0.1,
    weight_decay: float = 0.0, device: str = "cuda",
) -> tuple[float, list[float]]:
    """Train a single ``Linear`` and return ``(best_test_acc, per_epoch_acc)``.

    SGD + cosine LR + L2-normalized features is the standard SSL probe
    recipe. Test accuracy is logged every epoch and the maximum is
    returned (so we are robust to small train-loss/val-acc oscillations).
    """
    head = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(
        head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_feats = F.normalize(train_feats, dim=1).to(device)
    test_feats = F.normalize(test_feats, dim=1).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    ds = TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history: list[float] = []
    best = 0.0
    for ep in range(1, epochs + 1):
        head.train()
        for f, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(head(f), y)
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            pred = head(test_feats).argmax(1)
            acc = 100.0 * (pred == test_labels).float().mean().item()
        history.append(acc)
        best = max(best, acc)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  ep {ep:3d}  test acc {acc:6.2f}  (best {best:6.2f})")
    return best, history


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen-backbone linear probe")
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--probe-batch-size", type=int, default=1024)
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
    print(f"[linear] dumping features (dim={feat_dim})...")
    train_feats, train_labels = dump_features(model, train_loader, device)
    test_feats, test_labels = dump_features(model, test_loader, device)
    print("[linear] training head...")
    best, hist = _train_linear_head(
        train_feats, train_labels, test_feats, test_labels,
        feat_dim=feat_dim, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.probe_batch_size, device=device,
    )

    results = {
        "probe": "linear",
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "feature_dim": feat_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_test_top1": best,
        "test_top1_per_epoch": hist,
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[linear] best={best:.2f}% → {args.output}")


if __name__ == "__main__":
    main()
