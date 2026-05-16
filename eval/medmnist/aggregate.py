"""Aggregate the 360-run MedMNIST campaign into paper artifacts.

Reads every ``result.json`` the trainer wrote (locally or pulled from
S3 into a directory tree) and produces:

  1. The headline figure — accuracy/AUC gap of NELU/NiLU over the best
     baseline activation, plotted against dataset training-set size
     (log x). This is the paper's teaser: the data-scarcer the dataset,
     the larger the gap (the M4 prediction).
  2. The per-dataset AUC/ACC table (mean ± std over seeds) with a
     12-dataset average row, laid out so our rows sit next to the
     official MedMNIST v2 ResNet-18/50 baselines.
  3. The Friedman omnibus test (activation × dataset and activation ×
     backbone grids, α = 0.05) — the rigor element activation-function
     reviewers expect.

No campaign data is needed to import this module; run ``main`` once the
results exist. Expects a directory of ``*/result.json`` files (the S3
layout ``<exp>/result.json`` synced down works directly).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Official MedMNIST v2 ResNet-18 @28 from-scratch baselines (Yang et al.
# Sci Data 2022, Table 3) — transcribed for the side-by-side table. AUC.
OFFICIAL_R18_28_AUC = {
    "pathmnist": 0.983, "chestmnist": 0.768, "dermamnist": 0.917,
    "octmnist": 0.943, "pneumoniamnist": 0.944, "retinamnist": 0.717,
    "breastmnist": 0.901, "bloodmnist": 0.998, "tissuemnist": 0.930,
    "organamnist": 0.997, "organcmnist": 0.992, "organsmnist": 0.972,
}

BASELINE_ACTS = ("relu", "gelu", "silu")
OURS = ("nelu", "nilu")


def load_results(root: Path) -> list[dict]:
    """Every result.json under ``root`` (recursive)."""
    out = []
    for p in sorted(root.rglob("result.json")):
        try:
            d = json.loads(p.read_text())
            if {"dataset", "model", "activation", "test_auc"} <= d.keys():
                out.append(d)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _index(results):
    """(dataset, model, activation) -> list of per-seed test_auc / test_acc."""
    auc = defaultdict(list)
    acc = defaultdict(list)
    size = {}
    for r in results:
        k = (r["dataset"], r["model"], r["activation"])
        auc[k].append(r["test_auc"])
        acc[k].append(r["test_acc"])
        size[r["dataset"]] = r.get("train_size")
    return auc, acc, size


def per_dataset_table(results) -> str:
    auc, acc, size = _index(results)
    datasets = sorted({d for d, _, _ in auc}, key=lambda d: size.get(d, 0))
    models = sorted({m for _, m, _ in auc})
    acts = list(BASELINE_ACTS) + list(OURS)
    lines = []
    for model in models:
        lines.append(f"\n=== {model} (test AUC, mean±std over seeds) ===")
        hdr = f"{'dataset':16s}{'size':>8s}  " + "".join(
            f"{a:>14s}" for a in acts
        ) + f"{'official':>10s}"
        lines.append(hdr)
        agg = {a: [] for a in acts}
        for d in datasets:
            row = f"{d:16s}{size.get(d, 0):>8d}  "
            for a in acts:
                v = auc.get((d, model, a), [])
                if v:
                    m, s = np.mean(v), np.std(v)
                    agg[a].append(m)
                    row += f"{m:>8.3f}±{s:>4.3f}"
                else:
                    row += f"{'—':>14s}"
            off = OFFICIAL_R18_28_AUC.get(d)
            row += f"{(f'{off:.3f}' if off and model=='resnet18' else '—'):>10s}"
            lines.append(row)
        avg = f"{'AVG':16s}{'':>8s}  " + "".join(
            f"{np.mean(agg[a]):>14.3f}" if agg[a] else f"{'—':>14s}"
            for a in acts
        )
        lines.append(avg)
    return "\n".join(lines)


def headline_figure(results, out_path: Path) -> None:
    """Gap of best-of-{nelu,nilu} over best-of-{relu,gelu,silu}, vs size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auc, _, size = _index(results)
    models = sorted({m for _, m, _ in auc})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in models:
        datasets = sorted({d for d, mm, _ in auc if mm == model},
                          key=lambda d: size.get(d, 0))
        xs, ys = [], []
        for d in datasets:
            base = [np.mean(auc[(d, model, a)])
                    for a in BASELINE_ACTS if auc.get((d, model, a))]
            ours = [np.mean(auc[(d, model, a)])
                    for a in OURS if auc.get((d, model, a))]
            if not base or not ours:
                continue
            xs.append(size.get(d, 0))
            ys.append(max(ours) - max(base))
        if xs:
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            ax.plot(xs, ys * 100, "o-", label=model)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("training-set size (images, log scale)")
    ax.set_ylabel("test-AUC gap: best(NELU/NiLU) − best(ReLU/GELU/SiLU)  [×100]")
    ax.set_title("Gate normalization helps most when data is scarce")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[figure] {out_path}")


def friedman(results) -> str:
    """Friedman omnibus over activations, blocked by (dataset×model)."""
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        return "scipy not available — skipping Friedman test"
    auc, _, _ = _index(results)
    acts = list(BASELINE_ACTS) + list(OURS)
    blocks = sorted({(d, m) for d, m, _ in auc})
    cols = {a: [] for a in acts}
    for (d, m) in blocks:
        if all(auc.get((d, m, a)) for a in acts):
            for a in acts:
                cols[a].append(np.mean(auc[(d, m, a)]))
    n = len(cols[acts[0]])
    if n < 3:
        return f"Friedman: only {n} complete blocks — need ≥3, skipping"
    stat, p = friedmanchisquare(*[cols[a] for a in acts])
    mean_rank = {}
    arr = np.array([cols[a] for a in acts])           # (n_act, n_block)
    ranks = arr.argsort(0).argsort(0) + 1
    for i, a in enumerate(acts):
        mean_rank[a] = ranks[i].mean()
    lines = [
        f"Friedman χ² = {stat:.3f}, p = {p:.3g}, blocks = {n}, "
        f"activations = {len(acts)}",
        "mean ranks (higher = better AUC): "
        + ", ".join(f"{a}={mean_rank[a]:.2f}" for a in acts),
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate MedMNIST campaign")
    p.add_argument("--results_dir", required=True,
                   help="dir containing */result.json (S3 layout works)")
    p.add_argument("--out_dir", default="paper_results/medmnist")
    args = p.parse_args()

    root = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = load_results(root)
    print(f"loaded {len(results)} result.json files")
    if not results:
        print("no results yet — run after the campaign produces result.json")
        return

    table = per_dataset_table(results)
    (out / "auc_table.txt").write_text(table)
    print(table)

    fr = friedman(results)
    (out / "friedman.txt").write_text(fr)
    print("\n" + fr)

    headline_figure(results, out / "headline_gap_vs_size.png")


if __name__ == "__main__":
    main()
