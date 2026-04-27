"""Geometric probes on penultimate features. No training.

Three quantities, all computed from the (N × D) test-split feature matrix:

* **Effective rank** — exp(entropy(eigvals(cov(F)))). The "soft" count of
  active feature dimensions; higher = features spread across more
  directions = richer representation.
* **Anisotropy** — mean pairwise cosine similarity (centered features).
  Lower = more isotropic, which correlates with better downstream
  separability (Ethayarajh 2019).
* **Fisher discriminant ratio** — ratio of between-class to within-class
  variance, averaged over feature dimensions. A direct proxy for linear
  separability without actually training a probe.
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


def effective_rank(F_mat: torch.Tensor) -> float:
    """exp(entropy of normalized covariance eigenvalues)."""
    Fc = F_mat - F_mat.mean(dim=0, keepdim=True)
    cov = (Fc.t() @ Fc) / max(Fc.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.clamp(min=0)
    s = eigvals.sum()
    if s.item() <= 0:
        return 0.0
    p = eigvals / s
    p = p[p > 0]
    H = -(p * p.log()).sum().item()
    return float(torch.tensor(H).exp())


def participation_ratio(F_mat: torch.Tensor) -> float:
    """(sum λ)^2 / sum λ^2 — alternative dimensionality estimate."""
    Fc = F_mat - F_mat.mean(dim=0, keepdim=True)
    cov = (Fc.t() @ Fc) / max(Fc.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    num = eigvals.sum().pow(2)
    den = eigvals.pow(2).sum()
    if den.item() <= 0:
        return 0.0
    return float(num / den)


def anisotropy(F_mat: torch.Tensor, *, max_pairs: int = 4096) -> float:
    """Mean cosine similarity of (centered) feature pairs.

    Sub-sampled to ``max_pairs`` rows × ``max_pairs`` rows to keep the
    computation O(D * max_pairs²) and cheap.
    """
    Fc = F_mat - F_mat.mean(dim=0, keepdim=True)
    Fn = F.normalize(Fc, dim=1)
    n = Fn.size(0)
    if n > max_pairs:
        idx = torch.randperm(n)[:max_pairs]
        Fn = Fn[idx]
    sims = Fn @ Fn.t()
    # Exclude the diagonal (self-similarity = 1).
    mask = ~torch.eye(Fn.size(0), dtype=torch.bool, device=Fn.device)
    return float(sims[mask].mean())


def fisher_ratio(F_mat: torch.Tensor, labels: torch.Tensor) -> float:
    """Average per-dimension between/within class variance ratio."""
    classes = labels.unique()
    overall_mean = F_mat.mean(dim=0, keepdim=True)              # (1, D)
    between = torch.zeros(F_mat.size(1))
    within = torch.zeros(F_mat.size(1))
    total = 0
    for c in classes:
        m = labels == c
        n_c = int(m.sum())
        if n_c < 2:
            continue
        sub = F_mat[m]
        mu_c = sub.mean(dim=0, keepdim=True)                    # (1, D)
        between += n_c * (mu_c - overall_mean).pow(2).squeeze(0)
        within += ((sub - mu_c).pow(2)).sum(dim=0)
        total += n_c
    # Normalize by counts; epsilon to avoid divide-by-zero on dead dims.
    eps = 1e-8
    ratio = (between / max(len(classes) - 1, 1)) / (within / max(total - len(classes), 1) + eps)
    return float(ratio.mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Feature-geometry probes")
    add_common_args(p)
    p.add_argument("--use-train", action="store_true",
                   help="compute on train split (default: test split)")
    args = p.parse_args()
    device = device_from_args(args)

    model, feat_dim = build_feature_extractor(
        args.model, args.activation, args.checkpoint, device,
    )

    loader = build_loader(
        args.data_root, train=args.use_train,
        batch_size=args.batch_size, workers=args.workers, augment=False,
    )

    t0 = time.time()
    print(f"[geom] dumping features (dim={feat_dim}, split={'train' if args.use_train else 'test'})...")
    F_mat, labels = dump_features(model, loader, device)
    F_mat = F_mat.float()  # bf16 ckpts → fp32 for eigh
    labels = labels.long()

    print("[geom] computing...")
    er = effective_rank(F_mat)
    pr = participation_ratio(F_mat)
    aniso = anisotropy(F_mat)
    fisher = fisher_ratio(F_mat, labels)

    print(f"  effective_rank        {er:8.2f}  (D={feat_dim})")
    print(f"  participation_ratio   {pr:8.2f}")
    print(f"  anisotropy            {aniso:+8.4f}")
    print(f"  fisher_ratio          {fisher:8.4f}")

    results = {
        "probe": "geometry",
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "feature_dim": feat_dim,
        "split": "train" if args.use_train else "test",
        "n_samples": int(F_mat.size(0)),
        "effective_rank": er,
        "participation_ratio": pr,
        "anisotropy": aniso,
        "fisher_ratio": fisher,
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[geom] → {args.output}")


if __name__ == "__main__":
    main()
