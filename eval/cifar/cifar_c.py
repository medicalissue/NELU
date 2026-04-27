"""Thin wrapper over :mod:`eval.cifar_robustness` matching the new harness CLI.

The original :mod:`eval.cifar_robustness` already implements CIFAR-100-C
end-to-end. This wrapper just adapts the argument layout to the shared
``add_common_args`` schema so the launcher can call every probe with the
same flags.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eval.cifar._common import (
    add_common_args, build_eval_model, device_from_args,
)
from eval.cifar_robustness import (
    CORRUPTIONS, SEVERITIES, forward_corruption_all_severities,
)


def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-100-C robustness")
    add_common_args(p)
    p.add_argument("--share-logits-dir", default=None,
                   help="if set, reuse logits dumped here by the calibration "
                        "probe instead of re-forwarding each corruption")
    args = p.parse_args()
    device = device_from_args(args)

    root = Path(args.data_root) / "CIFAR-100-C"
    if not root.is_dir():
        raise FileNotFoundError(f"CIFAR-100-C directory not found at {root}")

    share = Path(args.share_logits_dir) if args.share_logits_dir else None
    if share and all((share / f"{c}.pt").is_file() for c in CORRUPTIONS):
        print(f"[cifar-c] reusing cached logits from {share}")
        model = None
    else:
        model = build_eval_model(args.model, args.activation, args.checkpoint, device)
        print(f"[cifar-c] model={args.model} act={args.activation}")

    t0 = time.time()
    out: dict = {}
    for corr in CORRUPTIONS:
        if share and (share / f"{corr}.pt").is_file():
            d = torch.load(share / f"{corr}.pt", weights_only=True)
            logits, targets = d["logits"], d["targets"]
        else:
            logits, targets = forward_corruption_all_severities(
                model, str(root), corr,
                batch_size=args.batch_size, workers=args.workers, device=device,
            )
        per_sev: list[float] = []
        for s in SEVERITIES:
            a = (s - 1) * 10_000
            b = s * 10_000
            preds = logits[a:b].argmax(dim=1)
            acc = 100.0 * (preds == targets[a:b]).float().mean().item()
            per_sev.append(acc)
        mean = sum(per_sev) / len(per_sev)
        out[corr] = {"per_severity": per_sev, "mean": mean}
        print(f"  {corr:<25s} {mean:6.2f}%")
    means = [v["mean"] for v in out.values()]
    out["_mean"] = sum(means) / len(means)
    print(f"  {'MEAN':<25s} {out['_mean']:6.2f}%")

    results = {
        "probe": "cifar_c",
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "cifar_100_c": out,
        "seconds": time.time() - t0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[cifar-c] → {args.output}")


if __name__ == "__main__":
    main()
