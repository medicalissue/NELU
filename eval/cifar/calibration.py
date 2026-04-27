"""Calibration on clean CIFAR-100 + per-severity CIFAR-100-C.

Reports the standard 15-bin Expected Calibration Error (Guo et al. 2017),
average confidence on correct vs wrong predictions, and Brier score.
A model with the same accuracy but lower ECE is, in a precise sense,
giving more honest probabilities — so it is using its representation
"better" in the bayesian-decision sense.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathlib import Path

from eval.cifar._common import (
    add_common_args, build_eval_model, build_loader,
    device_from_args, CORRUPTIONS, SEVERITIES,
)
from eval.cifar_robustness import forward_corruption_all_severities


@torch.no_grad()
def _collect_probs(model, loader: DataLoader, device: str):
    confs: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        c, idx = p.max(dim=1)
        confs.append(c.cpu()); preds.append(idx.cpu())
        targets.append(y); probs.append(p.cpu())
    return (
        torch.cat(confs), torch.cat(preds), torch.cat(targets),
        torch.cat(probs),
    )


def _stats_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute calibration stats given pre-collected logits.

    Used by both the corruption loop (one forward over all 5 severities,
    sliced afterwards) and the clean loader (separate forward).
    """
    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    acc = 100.0 * (preds == targets).float().mean().item()
    return {
        "top1": acc,
        "ece": ece(confs, preds, targets),
        "ece_5bin": ece(confs, preds, targets, n_bins=5),
        "brier": brier_score(probs, targets),
        "mean_conf": float(confs.mean()),
        "mean_conf_correct": float(confs[preds == targets].mean()) if (preds == targets).any() else 0.0,
        "mean_conf_wrong": float(confs[preds != targets].mean()) if (preds != targets).any() else 0.0,
    }


def ece(confidences: torch.Tensor, predictions: torch.Tensor,
        targets: torch.Tensor, *, n_bins: int = 15) -> float:
    edges = torch.linspace(0, 1, n_bins + 1)
    correct = (predictions == targets).float()
    n = confidences.size(0)
    out = 0.0
    for i in range(n_bins):
        m = (confidences > edges[i]) & (confidences <= edges[i + 1])
        if not m.any():
            continue
        acc = correct[m].mean().item()
        conf = confidences[m].mean().item()
        out += (m.sum().item() / n) * abs(acc - conf)
    return float(out)


def brier_score(probs: torch.Tensor, targets: torch.Tensor,
                num_classes: int = 100) -> float:
    one_hot = F.one_hot(targets, num_classes).float()
    return float((probs - one_hot).pow(2).sum(dim=1).mean())


def evaluate_loader(model, loader, device):
    confs, preds, targets, probs = _collect_probs(model, loader, device)
    acc = 100.0 * (preds == targets).float().mean().item()
    return {
        "top1": acc,
        "ece": ece(confs, preds, targets),
        "ece_5bin": ece(confs, preds, targets, n_bins=5),
        "brier": brier_score(probs, targets),
        "mean_conf": float(confs.mean()),
        "mean_conf_correct": float(confs[preds == targets].mean()) if (preds == targets).any() else 0.0,
        "mean_conf_wrong": float(confs[preds != targets].mean()) if (preds != targets).any() else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration on clean + corrupt CIFAR-100")
    add_common_args(p)
    p.add_argument("--skip-corruption", action="store_true",
                   help="only evaluate on clean test split")
    p.add_argument("--share-logits-dir", default=None,
                   help="if set, dump per-corruption logits/targets here so "
                        "cifar_c probe can reuse them and skip re-forwarding")
    args = p.parse_args()
    device = device_from_args(args)

    model = build_eval_model(args.model, args.activation, args.checkpoint, device)

    t0 = time.time()
    clean_loader = build_loader(
        args.data_root, train=False,
        batch_size=args.batch_size, workers=args.workers,
    )
    print("[calib] clean...")
    clean = evaluate_loader(model, clean_loader, device)
    print(f"  acc={clean['top1']:.2f}  ECE={clean['ece']:.4f}  Brier={clean['brier']:.4f}")

    out: dict = {"clean": clean, "corruption": {}}
    if not args.skip_corruption:
        # Save logits for cifar_c probe to reuse — one forward over the
        # full 50 000 images per corruption, sliced to severities here.
        # Sharing is opt-in: only enabled if --share-logits-dir is set.
        share_dir: Path | None = (
            Path(args.share_logits_dir) if args.share_logits_dir else None
        )
        if share_dir is not None:
            share_dir.mkdir(parents=True, exist_ok=True)

        cifar_c_root = str(Path(args.data_root) / "CIFAR-100-C")
        for corr in CORRUPTIONS:
            logits, targets = forward_corruption_all_severities(
                model, cifar_c_root, corr,
                batch_size=args.batch_size, workers=args.workers, device=device,
            )
            if share_dir is not None:
                torch.save(
                    {"logits": logits, "targets": targets},
                    share_dir / f"{corr}.pt",
                )
            per_sev: dict = {}
            for s in SEVERITIES:
                a = (s - 1) * 10_000
                b = s * 10_000
                per_sev[f"sev{s}"] = _stats_from_logits(logits[a:b], targets[a:b])
            mean_ece = sum(v["ece"] for v in per_sev.values()) / len(per_sev)
            mean_acc = sum(v["top1"] for v in per_sev.values()) / len(per_sev)
            out["corruption"][corr] = {
                "per_severity": per_sev, "mean_ece": mean_ece, "mean_top1": mean_acc,
            }
            print(f"  {corr:<25s}  acc={mean_acc:5.2f}  ECE={mean_ece:.4f}")
        out["mean_corrupt_ece"] = sum(
            v["mean_ece"] for v in out["corruption"].values()
        ) / len(out["corruption"])

    results = {
        "probe": "calibration",
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        **out,
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[calib] → {args.output}")


if __name__ == "__main__":
    main()
