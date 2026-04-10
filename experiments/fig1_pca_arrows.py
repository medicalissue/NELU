"""Figure 1 (PCA arrows): Output direction rotation under scale perturbation.

100 random vectors, PCA-projected. Each dot = one vector's output direction
at a given scale α. Connected by lines to show trajectory.

GELU: dots spread out (direction depends on scale).
ReLU/NELU: dots collapse to single point (direction invariant).
"""

import math, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def gelu(z):
    return z * 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))

def nelu(z, eps=1e-6):
    rms = z.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return z * 0.5 * (1.0 + torch.erf(z / (rms.detach() * math.sqrt(2))))

def relu(z):
    return torch.relu(z)


@torch.no_grad()
def compute_trajectories(fn, z_batch, alphas):
    """For each sample and each α, compute unit output direction.
    Returns (n_samples, n_alphas, d)."""
    trajs = []
    for alpha in alphas:
        y = fn(alpha * z_batch)
        y_norm = y / (y.norm(dim=-1, keepdim=True) + 1e-10)
        trajs.append(y_norm)
    return torch.stack(trajs, dim=1)  # (n, n_alphas, d)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    d = 256
    n_samples = 100
    alphas = np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0])
    n_alphas = len(alphas)
    alpha_ref_idx = 3  # α=1.0

    torch.manual_seed(42)
    z = torch.randn(n_samples, d)
    z = z / z.pow(2).mean(-1, keepdim=True).sqrt()  # unit RMS

    acts = [
        ("ReLU", relu, "#888888"),
        ("GELU", gelu, "#c0392b"),
        ("NELU", nelu, "#2471a3"),
    ]

    # Compute all trajectories
    all_trajs = {}
    for name, fn, _ in acts:
        all_trajs[name] = compute_trajectories(fn, z, alphas).numpy()

    # PCA: fit on GELU trajectories (where variation exists)
    gelu_flat = all_trajs["GELU"].reshape(-1, d)
    mean = gelu_flat.mean(axis=0)
    centered = gelu_flat - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2]  # (2, d)

    # Project all
    proj = {}
    for name in [n for n, _, _ in acts]:
        flat = all_trajs[name].reshape(-1, d)
        p = (flat - mean) @ basis.T
        proj[name] = p.reshape(n_samples, n_alphas, 2)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Color by α: blue (small) → red (large)
    alpha_cmap = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_alphas))

    for idx, (name, _, base_color) in enumerate(acts):
        ax = axes[idx]
        P = proj[name]  # (n_samples, n_alphas, 2)

        # Draw trajectories (thin lines connecting α values per sample)
        for s in range(n_samples):
            ax.plot(P[s, :, 0], P[s, :, 1],
                    color=base_color, alpha=0.08, lw=0.4, zorder=1)

        # Draw dots colored by α
        for a_idx in range(n_alphas):
            ax.scatter(P[:, a_idx, 0], P[:, a_idx, 1],
                       c=[alpha_cmap[a_idx]], s=3, alpha=0.5, zorder=2,
                       edgecolors="none")

        # Highlight α=1.0 (reference)
        ax.scatter(P[:, alpha_ref_idx, 0], P[:, alpha_ref_idx, 1],
                   c="black", s=5, alpha=0.6, zorder=3, edgecolors="none")

        # Compute mean rotation for title
        ref_dirs = torch.tensor(all_trajs[name][:, alpha_ref_idx])
        max_dirs = torch.tensor(all_trajs[name][:, -1])  # α=2.0
        cos = (ref_dirs * max_dirs).sum(-1).clamp(-1, 1)
        mean_angle = torch.rad2deg(torch.acos(cos)).mean().item()

        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#cccccc")

        ax.text(0.5, -0.1, f"({chr(97+idx)}) {name}  ({mean_angle:.1f}$^\\circ$)",
                transform=ax.transAxes, fontsize=9, fontweight="bold", ha="center")

    # Colorbar for α
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=alphas[0], vmax=alphas[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.02, pad=0.02, aspect=25)
    cbar.set_label("Scale $\\alpha$", fontsize=8)
    cbar.set_ticks([0.5, 1.0, 1.5, 2.0])
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    fig.subplots_adjust(bottom=0.15, wspace=0.08)

    fig.savefig(out_dir / "fig1_pca_arrows.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_pca_arrows.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved {out_dir}/fig1_pca_arrows.{{png,pdf}}")
    plt.close()


if __name__ == "__main__":
    main()
