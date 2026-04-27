"""Pairwise Centered Kernel Alignment between two checkpoints.

Forwards a fixed batch of CIFAR-100 test images through both models with
``register_forward_hook`` collecting outputs of every block-like module
(BasicBlock / InvertedResidual / ConvNeXtBlock / etc — anything that is a
direct child of a sequential stage). Then computes a ``len(A) × len(B)``
linear-CKA matrix (Kornblith 2019).

The intent is to read which depths first diverge between, e.g., NELU and
GELU on the same architecture: low CKA at the deep end with similar
final acc means the two models converged to *different* representations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from eval.cifar._common import (
    ACTIVATIONS, MODELS, build_eval_model, build_loader, device_from_args,
)


def _is_block_like(m: nn.Module) -> bool:
    """Heuristic: count modules that contain at least one Conv2d but
    are not themselves leaves. This picks up residual blocks, MBConv
    blocks, ConvNeXt blocks, etc. — anything one layer up from raw conv.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU,
                      nn.GELU, nn.SiLU)):
        return False
    has_conv = any(isinstance(c, nn.Conv2d) for c in m.modules() if c is not m)
    has_self_conv = any(isinstance(c, nn.Conv2d) for c in m.children())
    return has_conv and not has_self_conv  # block-level, not the whole net


def _collect_hook_targets(model: nn.Module) -> list[tuple[str, nn.Module]]:
    targets: list[tuple[str, nn.Module]] = []
    for name, m in model.named_modules():
        if name == "" or not _is_block_like(m):
            continue
        targets.append((name, m))
    return targets


@torch.no_grad()
def _gather_activations(
    model: nn.Module, x: torch.Tensor, targets: list[tuple[str, nn.Module]],
) -> dict[str, torch.Tensor]:
    """One forward, capture every target's output (flattened to 2-D)."""
    cache: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    def _make_hook(name):
        def _hook(_m, _inp, out):
            t = out
            if isinstance(t, (tuple, list)):
                t = t[0]
            cache[name] = torch.flatten(t, 1).detach()
        return _hook
    for n, m in targets:
        handles.append(m.register_forward_hook(_make_hook(n)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return cache


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centered linear CKA between two ``(N, D)`` activation matrices."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    # CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    cross = (Y.t() @ X).pow(2).sum()
    nx = (X.t() @ X).pow(2).sum().sqrt()
    ny = (Y.t() @ Y).pow(2).sum().sqrt()
    if nx.item() == 0 or ny.item() == 0:
        return 0.0
    return float(cross / (nx * ny))


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise CKA matrix")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--activation-a", required=True, choices=ACTIVATIONS)
    p.add_argument("--activation-b", required=True, choices=ACTIVATIONS)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--output", required=True)
    p.add_argument("--n-samples", type=int, default=1024,
                   help="number of CIFAR-100 test images to forward")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = device_from_args(args)
    torch.manual_seed(args.seed)

    model_a = build_eval_model(args.model, args.activation_a, args.checkpoint_a, device)
    model_b = build_eval_model(args.model, args.activation_b, args.checkpoint_b, device)

    targets_a = _collect_hook_targets(model_a)
    targets_b = _collect_hook_targets(model_b)
    print(f"[cka] {len(targets_a)} blocks (A), {len(targets_b)} blocks (B)")

    loader = build_loader(
        args.data_root, train=False,
        batch_size=args.batch_size, workers=args.workers,
    )
    # Take the first ``n_samples`` images deterministically.
    xs: list[torch.Tensor] = []
    n = 0
    for x, _ in loader:
        xs.append(x)
        n += x.size(0)
        if n >= args.n_samples:
            break
    X = torch.cat(xs, dim=0)[: args.n_samples].to(device)

    t0 = time.time()
    acts_a = _gather_activations(model_a, X, targets_a)
    acts_b = _gather_activations(model_b, X, targets_b)
    names_a = [n for n, _ in targets_a]
    names_b = [n for n, _ in targets_b]

    matrix = [[0.0] * len(names_b) for _ in names_a]
    for i, na in enumerate(names_a):
        Xi = acts_a[na].float()
        for j, nb in enumerate(names_b):
            Yj = acts_b[nb].float()
            matrix[i][j] = linear_cka(Xi, Yj)
    diag = [matrix[i][i] for i in range(min(len(names_a), len(names_b)))]
    print(f"[cka] mean diag CKA = {sum(diag) / max(len(diag), 1):.3f}")

    results = {
        "probe": "cka",
        "model": args.model,
        "activation_a": args.activation_a,
        "activation_b": args.activation_b,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "n_samples": int(X.size(0)),
        "blocks_a": names_a,
        "blocks_b": names_b,
        "cka_matrix": matrix,
        "diag_mean": sum(diag) / max(len(diag), 1),
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[cka] → {args.output}")


if __name__ == "__main__":
    main()
