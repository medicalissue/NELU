"""
Figure 1 (teaser): PCA arrow plot showing direction rotation.

Encoding:
  - ε=0: thick solid arrow with dot at tip
  - ε>0 (scale up): solid arrows, thinner for larger ε
  - ε<0 (scale down): dashed arrows, thinner for larger |ε|
  - Only ε_min and ε_max labeled at arrow tips

Usage:
    python experiments/fig1_teaser.py
"""

import math
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def gelu(z):
    return z * 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))

def nelu(z, eps=1e-6):
    rms = z.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

def relu(z):
    return torch.relu(z)

def unit_dir(y):
    return y / (y.norm() + 1e-10)

def angle_deg(a, b):
    cos = torch.dot(unit_dir(a), unit_dir(b)).clamp(-1, 1)
    return torch.acos(cos).item() * 180 / math.pi


@torch.no_grad()
def find_worst_case(d=768, n_candidates=10000, rho_range=(1.0, 6.0)):
    best_z, best_angle = None, 0.0
    for _ in range(n_candidates):
        z = torch.randn(d)
        z = z / z.norm() * math.sqrt(d)
        rho = np.random.uniform(*rho_range)
        z = z * rho
        y1 = gelu(z)
        y2 = gelu(0.5 * z)
        ang = angle_deg(y1, y2)
        if ang > best_angle:
            best_angle = ang
            best_z = z.clone()
    return best_z


@torch.no_grad()
def collect_directions(z, f, epsilons):
    dirs = []
    for eps in epsilons:
        y = f((1.0 + eps) * z)
        dirs.append(unit_dir(y).numpy())
    return np.stack(dirs)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    epsilons = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0])

    print("Finding worst-case input...")
    z = find_worst_case()
    rms_val = z.pow(2).mean().sqrt().item()
    print(f"  rms = {rms_val:.2f}")

    acts = {"ReLU": relu, "GELU": gelu, "NELU": nelu}
    colors = {"ReLU": "#888888", "GELU": "#d62728", "NELU": "#1f77b4"}

    all_dirs = {}
    for name, fn in acts.items():
        all_dirs[name] = collect_directions(z, fn, epsilons)

    # PCA
    combined = np.concatenate(list(all_dirs.values()))
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = {name: all_dirs[name] @ Vt[:2].T for name in acts}

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    zero_idx = np.argmin(np.abs(epsilons))
    all_pts = np.concatenate(list(proj.values()))
    margin = np.abs(all_pts).max() * 1.5

    for idx, name in enumerate(["ReLU", "GELU", "NELU"]):
        ax = axes[idx]
        pts = proj[name]
        c = colors[name]

        # Draw arrows: solid for ε≥0, dashed for ε<0
        for i, ev in enumerate(epsilons):
            dx, dy = pts[i]
            is_zero = (i == zero_idx)
            is_neg = (ev < 0)

            lw = 2.5 if is_zero else 1.2
            ls = "--" if is_neg else "-"
            alpha = 1.0 if is_zero else 0.5

            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c,
                                        alpha=alpha, lw=lw, linestyle=ls,
                                        mutation_scale=14))

        # Dot at ε=0 tip
        ax.plot(*pts[zero_idx], "o", color=c, markersize=6, zorder=6)
        ax.plot(0, 0, "o", color="#999999", markersize=3, zorder=5)

        # Label only the two extremes (first and last)
        # Place them away from arrows
        for li, txt in [(0, f"$\\times${1+epsilons[0]:.1f}"),
                        (-1, f"$\\times${1+epsilons[-1]:.1f}")]:
            px, py = pts[li]
            # Nudge outward from origin
            r = math.sqrt(px**2 + py**2) + 1e-10
            ox, oy = px / r * margin * 0.12, py / r * margin * 0.12
            ax.text(px + ox, py + oy, txt, fontsize=8, color=c,
                    ha="center", va="center", alpha=0.8)

        # Rotation angle
        ang = angle_deg(
            torch.tensor(all_dirs[name][0]),
            torch.tensor(all_dirs[name][-1])
        )
        ax.set_title(f"{name}  ({ang:.1f}$^\\circ$ total rotation)", fontweight="bold")

        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("PC 1")
        if idx == 0:
            ax.set_ylabel("PC 2")

    # Legend explaining line styles (once, outside)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", lw=2.5, label="$\\varepsilon$=0 (original)"),
        Line2D([0], [0], color="gray", lw=1.2, label="$\\varepsilon$>0 (scale up)"),
        Line2D([0], [0], color="gray", lw=1.2, ls="--", label="$\\varepsilon$<0 (scale down)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Output direction under scale perturbation "
        "$\\mathbf{z} \\to (1+\\varepsilon)\\mathbf{z}$",
        fontweight="bold", fontsize=12, y=0.98
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_dir / "fig1_teaser.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_teaser.pdf", bbox_inches="tight", facecolor="white")
    print(f"\n  Saved {out_dir}/fig1_teaser.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
