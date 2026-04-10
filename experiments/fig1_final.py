"""Figure 1 (final): Scale-equivariance of ReLU vs GELU vs NELU.

Two-row layout:
  Top:    2D geometry — unit-circle inputs → output-direction curve for multiple α.
  Bottom: θ_out vs θ_in map — identity line = perfect direction preservation,
          curves overlapping = scale-invariant.
"""

import math, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "savefig.dpi": 300,
})

_INV_SQRT2 = 1.0 / math.sqrt(2)

def gelu(z):
    return z * 0.5 * (1.0 + torch.erf(z * _INV_SQRT2))

def nelu(z, eps=1e-6):
    rms = z.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

def relu(z):
    return torch.relu(z)


@torch.no_grad()
def output_direction(fn, alpha, n=400):
    theta = torch.linspace(0, 2 * math.pi, n)
    z = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    y = fn(alpha * z)
    norm = y.norm(dim=-1, keepdim=True) + 1e-12
    return (y / norm).numpy()


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    alphas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_theta = 500

    acts = [
        ("ReLU", relu, "#888888"),
        ("GELU", gelu, "#c0392b"),
        ("NELU", nelu, "#2471a3"),
    ]

    fig = plt.figure(figsize=(7.0, 4.5))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1.05, 1.0],
        hspace=0.45, wspace=0.18,
        left=0.08, right=0.98, top=0.93, bottom=0.13,
    )

    # ─── Top row: 2D output-direction curves ─────────────────────────
    for idx, (name, fn, base_color) in enumerate(acts):
        ax = fig.add_subplot(gs[0, idx])

        tt = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(tt), np.sin(tt), color="#bbbbbb", lw=0.7,
                linestyle="--", zorder=1)

        cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", ["#eeeeee", base_color, "#1a1a1a"], N=256)

        for ai, alpha in enumerate(alphas):
            curve = output_direction(fn, alpha, n=n_theta)
            curve = np.vstack([curve, curve[:1]])
            frac = ai / max(len(alphas) - 1, 1)
            color = cmap(0.25 + 0.65 * frac)
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.4,
                    alpha=0.9, zorder=2 + ai)

        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3); spine.set_color("#dddddd")
        ax.set_title(f"{name}", fontsize=10, fontweight="bold", pad=4)

    # ─── Bottom row: θ_out vs θ_in ───────────────────────────────────
    for idx, (name, fn, base_color) in enumerate(acts):
        ax = fig.add_subplot(gs[1, idx])

        # Identity reference
        ax.plot([-180, 180], [-180, 180], color="#bbbbbb",
                lw=0.6, linestyle="--", zorder=0)

        cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", ["#eeeeee", base_color, "#1a1a1a"], N=256)

        theta = np.linspace(-np.pi, np.pi, 600)

        for ai, alpha in enumerate(alphas):
            z = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], -1),
                             dtype=torch.float32)
            y = fn(alpha * z).numpy()
            tout = np.degrees(np.arctan2(y[:, 1], y[:, 0]))
            tin = np.degrees(theta)

            # Split on big jumps (ReLU 3rd-quadrant 0/0 discontinuity)
            jumps = np.where(np.abs(np.diff(tout)) > 90)[0]
            segs = np.split(np.arange(len(tin)), jumps + 1) \
                if len(jumps) else [np.arange(len(tin))]

            frac = ai / max(len(alphas) - 1, 1)
            color = cmap(0.25 + 0.65 * frac)

            for seg in segs:
                if len(seg) < 2:
                    continue
                ax.plot(tin[seg], tout[seg], color=color, lw=1.2,
                        alpha=0.9,
                        label=(f"$\\alpha\\!=\\!{alpha:g}$"
                               if idx == 0 and seg is segs[0] else None))

        ax.set_xlabel(r"$\theta_{\mathrm{in}}$ (deg)", fontsize=8)
        if idx == 0:
            ax.set_ylabel(r"$\theta_{\mathrm{out}}$ (deg)", fontsize=8)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=False, fontsize=7,
               handlelength=1.5, columnspacing=1.3)

    fig.savefig(out_dir / "fig1_final.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_final.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_final.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
