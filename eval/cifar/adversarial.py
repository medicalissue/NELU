"""FGSM and PGD ℓ∞ adversarial robustness on CIFAR-100 test split.

ε is given in 0–255 units and applied in *normalized* image space — we
re-scale per channel using the CIFAR-100 std so that an ε of "k pixels"
truly clamps the perturbation in pixel space. PGD uses 10 steps with
α = ε/4 and a uniform-random start, the standard recipe.

The point isn't to bench-mark the model adversarially per se; it's to
read off the local smoothness of the representation. If two models hit
the same clean acc but one falls off a cliff at ε=2/255 while the other
loses only a few points, the second model's penultimate space is a
flatter, more linear manifold around training data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from eval.cifar._common import (
    add_common_args, build_eval_model, build_loader, device_from_args,
)
from train.cifar import CIFAR100_MEAN, CIFAR100_STD


def _per_channel(vals, device):
    return torch.tensor(vals, device=device).view(1, 3, 1, 1)


def fgsm_attack(model, x, y, eps_normalized):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps_normalized * grad.sign()).detach()


def pgd_attack(model, x, y, eps_normalized, *, steps: int = 10, alpha=None):
    if alpha is None:
        alpha = eps_normalized / 4
    x_adv = x + (torch.rand_like(x) * 2 - 1) * eps_normalized
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # ℓ∞ ball around the original (in normalized space).
        delta = torch.clamp(x_adv - x, -eps_normalized, eps_normalized)
        x_adv = x + delta
    return x_adv.detach()


def evaluate_attack(model, loader, device, *, eps_pix: float, attack: str,
                    pgd_steps: int = 10):
    """``eps_pix`` is in 0..255 pixel units. We convert to per-channel
    normalized magnitude so the ℓ∞ ball is the right size in pixel space.
    """
    eps_norm = (eps_pix / 255.0) / _per_channel(CIFAR100_STD, device)
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if attack == "clean":
            x_adv = x
        elif attack == "fgsm":
            x_adv = fgsm_attack(model, x, y, eps_norm)
        elif attack == "pgd":
            x_adv = pgd_attack(model, x, y, eps_norm, steps=pgd_steps)
        else:
            raise ValueError(attack)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main() -> None:
    p = argparse.ArgumentParser(description="FGSM/PGD adversarial robustness")
    add_common_args(p)
    p.add_argument("--eps-pix", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                   help="ℓ∞ ε in 0..255 pixel units")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap to first N batches for a quick check")
    args = p.parse_args()
    device = device_from_args(args)

    model = build_eval_model(args.model, args.activation, args.checkpoint, device)
    model.requires_grad_(False)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    loader = build_loader(
        args.data_root, train=False,
        batch_size=args.batch_size, workers=args.workers,
    )
    if args.max_batches is not None:
        # Slice the loader.
        from itertools import islice
        loader = list(islice(loader, args.max_batches))

    t0 = time.time()
    out: dict = {"clean": evaluate_attack(model, loader, device,
                                          eps_pix=0.0, attack="clean")}
    print(f"  clean        {out['clean']:6.2f}%")
    for eps in args.eps_pix:
        f = evaluate_attack(model, loader, device, eps_pix=eps, attack="fgsm")
        pg = evaluate_attack(model, loader, device, eps_pix=eps, attack="pgd",
                             pgd_steps=args.pgd_steps)
        out[f"fgsm_eps{eps}"] = f
        out[f"pgd_eps{eps}"] = pg
        print(f"  ε={eps:>4.1f}/255  fgsm={f:6.2f}%  pgd={pg:6.2f}%")

    results = {
        "probe": "adversarial",
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "eps_pix": args.eps_pix,
        "pgd_steps": args.pgd_steps,
        **out,
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[adv] → {args.output}")


if __name__ == "__main__":
    main()
