"""Figure 1 (2D geometry): Show scale-equivariance directly in the plane.

No PCA, no projection — work in d=2 where we can draw actual vectors.
Sample unit-norm inputs around the circle, apply activation at different
scales α, and plot the output direction.

GELU: unit circle → distorted shape that changes with α (direction flips!)
ReLU: angle-preserving but sparsifying (onto coordinate cone)
NELU: unit circle → same shape at every α (scale equivariant)

This is a DIRECT, no-approximation demonstration.
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
from matplotlib.patches import FancyArrowPatch

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
def trace_curve(fn, alpha, n=400):
    """Apply fn(α·z) for z on unit circle, return the (x, y) curve."""
    theta = torch.linspace(0, 2 * math.pi, n)
    z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (n, 2)
    y = fn(alpha * z)
    return y.numpy()


@torch.no_grad()
def trace_direction(fn, alpha, n=400):
    """Unit-normalized output direction."""
    y = trace_curve(fn, alpha, n)
    norms = np.linalg.norm(y, axis=-1, keepdims=True) + 1e-12
    return y / norms


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    alphas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_theta = 400

    acts = [
        ("ReLU", relu, "#888888"),
        ("GELU", gelu, "#c0392b"),
        ("NELU", nelu, "#2471a3"),
    ]

    # ═══════════════════════════════════════════════════════════════
    # Style A: Output curves (normalized) overlaid for multiple α
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))

    # One colormap per activation, darker with larger α
    for idx, (name, fn, base_color) in enumerate(acts):
        ax = axes[idx]

        # Draw the unit circle (reference)
        tt = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(tt), np.sin(tt), color="#bbbbbb", lw=0.6,
                linestyle="--", zorder=1, label="input unit circle")

        # Build color gradient: light → dark with α
        cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", ["#f0f0f0", base_color, "#1a1a1a"], N=256)

        for ai, alpha in enumerate(alphas):
            curve = trace_direction(fn, alpha, n=n_theta)
            # Close the curve
            curve = np.vstack([curve, curve[:1]])
            frac = ai / max(len(alphas) - 1, 1)
            color = cmap(0.2 + 0.7 * frac)
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.3,
                    alpha=0.9, zorder=2 + ai,
                    label=f"$\\alpha={alpha:g}$" if idx == 1 else None)

        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3); spine.set_color("#dddddd")
        ax.set_title(f"{name}", fontsize=9, fontweight="bold", pad=4)

    # Legend on middle panel
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=5, frameon=False, fontsize=6.5,
                   handlelength=1.2, columnspacing=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, wspace=0.08)
    fig.savefig(out_dir / "fig1_geo_curves.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_geo_curves.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_geo_curves.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style B: Output-angle map  θ_in → θ_out  for multiple α
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)

    for idx, (name, fn, base_color) in enumerate(acts):
        ax = axes[idx]
        theta = np.linspace(-np.pi, np.pi, 500)
        ax.axhline(0, color="#dddddd", lw=0.4, zorder=0)

        cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", ["#f0f0f0", base_color, "#1a1a1a"], N=256)

        for ai, alpha in enumerate(alphas):
            z = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], -1), dtype=torch.float32)
            y = fn(alpha * z).numpy()
            theta_out = np.arctan2(y[:, 1], y[:, 0])
            # Signed deviation from θ_in
            dev = np.unwrap(theta_out - theta)
            dev = (dev + np.pi) % (2 * np.pi) - np.pi
            frac = ai / max(len(alphas) - 1, 1)
            color = cmap(0.2 + 0.7 * frac)
            ax.plot(np.degrees(theta), np.degrees(dev), color=color, lw=1.1,
                    alpha=0.9, label=f"$\\alpha={alpha:g}$" if idx == 1 else None)

        ax.set_xlabel("Input angle $\\theta_{\\mathrm{in}}$ (deg)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("$\\theta_{\\mathrm{out}} - \\theta_{\\mathrm{in}}$ (deg)", fontsize=7)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlim(-180, 180)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{name}", fontsize=9, fontweight="bold", pad=3)

    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
                   ncol=5, frameon=False, fontsize=6.5,
                   handlelength=1.2, columnspacing=0.8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.08)
    fig.savefig(out_dir / "fig1_geo_angle.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_geo_angle.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_geo_angle.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style C: Combined — top row curves, bottom row θ_out vs θ_in
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(7, 4.4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 0.85],
                           hspace=0.45, wspace=0.10)

    # Top row: 2D curves
    for idx, (name, fn, base_color) in enumerate(acts):
        ax = fig.add_subplot(gs[0, idx])
        tt = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(tt), np.sin(tt), color="#bbbbbb", lw=0.6,
                linestyle="--", zorder=1)

        cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", ["#f5f5f5", base_color, "#1a1a1a"], N=256)

        for ai, alpha in enumerate(alphas):
            curve = trace_direction(fn, alpha, n=n_theta)
            curve = np.vstack([curve, curve[:1]])
            frac = ai / max(len(alphas) - 1, 1)
            color = cmap(0.25 + 0.65 * frac)
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.3,
                    alpha=0.9, zorder=2 + ai)

        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3); spine.set_color("#dddddd")
        ax.set_title(f"{name}", fontsize=9, fontweight="bold", pad=3)

    # Bottom row: θ_out vs θ_in (direct output angle)
    for idx, (name, fn, base_color) in enumerate(acts):
        ax = fig.add_subplot(gs[1, idx])
        theta = np.linspace(-np.pi, np.pi, 500)
        # Identity reference θ_out = θ_in
        ax.plot([-180, 180], [-180, 180], color="#dddddd",
                lw=0.4, linestyle="--", zorder=0)

        cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", ["#f5f5f5", base_color, "#1a1a1a"], N=256)

        for ai, alpha in enumerate(alphas):
            z = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], -1),
                             dtype=torch.float32)
            y = fn(alpha * z).numpy()
            theta_out = np.arctan2(y[:, 1], y[:, 0])
            # Break lines on discontinuity (ReLU's 3rd-quadrant jump)
            tout_deg = np.degrees(theta_out)
            tin_deg = np.degrees(theta)
            jumps = np.where(np.abs(np.diff(tout_deg)) > 90)[0]
            segments = np.split(np.arange(len(tin_deg)),
                                jumps + 1) if len(jumps) else [np.arange(len(tin_deg))]
            frac = ai / max(len(alphas) - 1, 1)
            color = cmap(0.25 + 0.65 * frac)
            for seg in segments:
                if len(seg) < 2:
                    continue
                ax.plot(tin_deg[seg], tout_deg[seg], color=color, lw=1.1,
                        alpha=0.9,
                        label=(f"$\\alpha={alpha:g}$"
                               if idx == 0 and seg is segments[0] else None))

        ax.set_xlabel("$\\theta_{\\mathrm{in}}$ (deg)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("$\\theta_{\\mathrm{out}}$ (deg)", fontsize=7)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=5, frameon=False, fontsize=7,
               handlelength=1.5, columnspacing=1.2)

    fig.savefig(out_dir / "fig1_geo_combined.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_geo_combined.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_geo_combined.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style D: Polar plot — radial = α, angular = θ_in, color = |Δθ|
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8),
                              subplot_kw={"projection": "polar"})

    alpha_dense = np.geomspace(0.25, 16.0, 60)
    theta_dense = np.linspace(-np.pi, np.pi, 200)

    # Compute max deviation for shared color scale
    vmax = 0
    dev_maps = {}
    for name, fn, _ in acts:
        M = np.zeros((len(alpha_dense), len(theta_dense)))
        for ai, alpha in enumerate(alpha_dense):
            z = torch.tensor(np.stack([np.cos(theta_dense), np.sin(theta_dense)], -1),
                             dtype=torch.float32)
            y = fn(alpha * z).numpy()
            theta_out = np.arctan2(y[:, 1], y[:, 0])
            dev = (theta_out - theta_dense + np.pi) % (2 * np.pi) - np.pi
            M[ai] = np.degrees(np.abs(dev))
        dev_maps[name] = M
        vmax = max(vmax, M.max())

    for idx, (name, _, _) in enumerate(acts):
        ax = axes[idx]
        T, R = np.meshgrid(theta_dense, alpha_dense)
        im = ax.pcolormesh(T, np.log2(R), dev_maps[name],
                           cmap="magma", vmin=0, vmax=vmax,
                           shading="auto")
        ax.set_yticks([np.log2(0.5), 0, np.log2(2), np.log2(8)])
        ax.set_yticklabels(["0.5", "1", "2", "8"], fontsize=5)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["0$^\\circ$", "90$^\\circ$", "180$^\\circ$", "270$^\\circ$"],
                           fontsize=5)
        ax.set_title(f"{name}", fontsize=9, fontweight="bold", pad=8)
        ax.grid(alpha=0.2)

    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.025,
                        pad=0.04, aspect=30)
    cbar.set_label("$|\\Delta\\theta|$ (deg)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(out_dir / "fig1_geo_polar.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_geo_polar.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_geo_polar.{{png,pdf}}")
    plt.close()

    # Print summary
    print("\nMax angular deviation per activation (d=2, α ∈ [0.25, 16]):")
    for name, _, _ in acts:
        print(f"  {name}: {dev_maps[name].max():.2f}°")


if __name__ == "__main__":
    main()
