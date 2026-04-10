"""Figure 1 (direction field): How output direction changes under scale perturbation.

Multiple visualization approaches:
  (A) Quiver / velocity field — arrows show direction shift when α goes 1→2
  (B) Angular deviation heatmap — color = degrees of rotation per input point
  (C) Cosine similarity matrix — pairwise cos-sim across α values
  (D) Stream plot — continuous flow of direction change

All projected to 2D via PCA on the GELU outputs (where variation exists).

GELU: large, diverse arrows (direction depends on scale).
ReLU/NELU: zero-length arrows (direction invariant).
"""

import math, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

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
def unit_dir(fn, z):
    """Compute unit output direction."""
    y = fn(z)
    return y / (y.norm(dim=-1, keepdim=True) + 1e-10)


@torch.no_grad()
def compute_directions(fn, z_batch, alphas):
    """For each sample and each α, compute unit output direction.
    Returns (n_samples, n_alphas, d)."""
    dirs = []
    for alpha in alphas:
        dirs.append(unit_dir(fn, alpha * z_batch))
    return torch.stack(dirs, dim=1)  # (n, n_alphas, d)


def pca_project(data_flat, n_components=2, fit_data=None):
    """PCA projection. fit_data used to compute basis if provided."""
    if fit_data is None:
        fit_data = data_flat
    mean = fit_data.mean(axis=0)
    centered = fit_data - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:n_components]
    return (data_flat - mean) @ basis.T, mean, basis


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    d = 32  # lower dim → larger angular effect in PCA
    n_samples = 200
    alphas = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    alpha_ref_idx = 2  # α = 1.0
    alpha_end_idx = -1  # α = 8.0

    torch.manual_seed(42)
    z = torch.randn(n_samples, d)
    z = z / z.pow(2).mean(-1, keepdim=True).sqrt()  # unit RMS

    acts = [
        ("ReLU", relu, "#888888"),
        ("GELU", gelu, "#c0392b"),
        ("NELU", nelu, "#2471a3"),
    ]

    # Compute all directions
    all_dirs = {}
    for name, fn, _ in acts:
        all_dirs[name] = compute_directions(fn, z, alphas).numpy()

    # PCA basis from GELU (where variation exists)
    gelu_flat = all_dirs["GELU"].reshape(-1, d)
    _, pca_mean, pca_basis = pca_project(gelu_flat)

    # Project all to 2D
    proj = {}
    for name in [n for n, _, _ in acts]:
        flat = all_dirs[name].reshape(-1, d)
        p = (flat - pca_mean) @ pca_basis.T
        proj[name] = p.reshape(n_samples, len(alphas), 2)

    # Compute angular deviations (α=1 → α=2)
    angles = {}
    for name in [n for n, _, _ in acts]:
        ref = torch.tensor(all_dirs[name][:, alpha_ref_idx])
        end = torch.tensor(all_dirs[name][:, alpha_end_idx])
        cos = (ref * end).sum(-1).clamp(-1, 1)
        angles[name] = torch.rad2deg(torch.acos(cos)).numpy()

    # ═══════════════════════════════════════════════════════════════
    # Style 1: Quiver plot (velocity field)
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Shared axes range (use GELU which has the most spread)
    P_g = proj["GELU"]
    all_qx = np.concatenate([P_g[:, alpha_ref_idx, 0], P_g[:, alpha_end_idx, 0]])
    all_qy = np.concatenate([P_g[:, alpha_ref_idx, 1], P_g[:, alpha_end_idx, 1]])
    qpad = (all_qx.max() - all_qx.min()) * 0.12
    q_xlim = (all_qx.min() - qpad, all_qx.max() + qpad)
    q_ylim = (all_qy.min() - qpad, all_qy.max() + qpad)

    q_max_disp = max(
        np.sqrt((proj[n][:, alpha_end_idx, 0] - proj[n][:, alpha_ref_idx, 0])**2 +
                (proj[n][:, alpha_end_idx, 1] - proj[n][:, alpha_ref_idx, 1])**2).max()
        for n in ["ReLU", "GELU", "NELU"]
    )
    q_norm = Normalize(vmin=0, vmax=q_max_disp)

    for idx, (name, _, color) in enumerate(acts):
        ax = axes[idx]
        P = proj[name]
        x0 = P[:, alpha_ref_idx, 0]
        y0 = P[:, alpha_ref_idx, 1]
        dx = P[:, alpha_end_idx, 0] - x0
        dy = P[:, alpha_end_idx, 1] - y0
        arrow_len = np.sqrt(dx**2 + dy**2)

        ax.scatter(x0, y0, c=color, s=4, alpha=0.25, edgecolors="none", zorder=1)

        ax.quiver(x0, y0, dx, dy, arrow_len,
                  cmap="inferno", norm=q_norm,
                  angles='xy', scale_units='xy', scale=1,
                  width=0.005, headwidth=3.5, headlength=4,
                  headaxislength=3.2,
                  alpha=0.75, zorder=2)

        mean_angle = angles[name].mean()
        ax.set_aspect("equal")
        ax.set_xlim(q_xlim); ax.set_ylim(q_ylim)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#cccccc")

        ax.text(0.5, -0.12, f"({chr(97+idx)}) {name}  ({mean_angle:.1f}$^\\circ$)",
                transform=ax.transAxes, fontsize=9, fontweight="bold", ha="center")

    sm = ScalarMappable(cmap="inferno", norm=q_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.02, pad=0.02, aspect=25)
    cbar.set_label("PCA displacement", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    fig.subplots_adjust(bottom=0.18, wspace=0.08)
    fig.savefig(out_dir / "fig1_quiver.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_quiver.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_quiver.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style 2: Angular deviation histogram
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.2))

    bins = np.linspace(0, max(angles["GELU"].max() * 1.1, 1.0), 40)
    for name, _, color in acts:
        ax.hist(angles[name], bins=bins, color=color, alpha=0.6,
                label=f"{name} ({angles[name].mean():.1f}$^\\circ$)",
                edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Angular deviation (degrees)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_angle_hist.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_angle_hist.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_angle_hist.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style 3: Cosine similarity heatmap (α × α)
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))

    for idx, (name, _, _) in enumerate(acts):
        ax = axes[idx]
        D = torch.tensor(all_dirs[name])  # (n, n_alphas, d)
        # Mean cosine similarity across samples for each α pair
        cos_mat = np.zeros((len(alphas), len(alphas)))
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                cos = (D[:, i] * D[:, j]).sum(-1).clamp(-1, 1)
                cos_mat[i, j] = cos.mean().item()

        im = ax.imshow(cos_mat, cmap="RdYlGn", vmin=0.95, vmax=1.0,
                       origin="lower", aspect="equal")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.1f}" for a in alphas], fontsize=5, rotation=45)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([f"{a:.1f}" for a in alphas], fontsize=5)
        if idx == 0:
            ax.set_ylabel("Scale $\\alpha$", fontsize=7)
        ax.set_title(name, fontsize=9, fontweight="bold", pad=3)

        # Annotate cells
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                val = cos_mat[i, j]
                color = "white" if val < 0.97 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                       fontsize=3.5, color=color)

    fig.tight_layout(pad=0.3, w_pad=0.5)
    fig.savefig(out_dir / "fig1_cossim.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_cossim.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_cossim.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style 4: Combined — quiver (top) + histogram (bottom)
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(7, 4.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 0.7],
                           hspace=0.35, wspace=0.10)

    # Shared xlim/ylim across all 3 quiver panels (use GELU range)
    P_gelu = proj["GELU"]
    all_x = np.concatenate([P_gelu[:, alpha_ref_idx, 0], P_gelu[:, alpha_end_idx, 0]])
    all_y = np.concatenate([P_gelu[:, alpha_ref_idx, 1], P_gelu[:, alpha_end_idx, 1]])
    pad = (all_x.max() - all_x.min()) * 0.12
    shared_xlim = (all_x.min() - pad, all_x.max() + pad)
    shared_ylim = (all_y.min() - pad, all_y.max() + pad)

    # Global max displacement for consistent color scale
    global_max_disp = max(
        np.sqrt((proj[n][:, alpha_end_idx, 0] - proj[n][:, alpha_ref_idx, 0])**2 +
                (proj[n][:, alpha_end_idx, 1] - proj[n][:, alpha_ref_idx, 1])**2).max()
        for n in ["ReLU", "GELU", "NELU"]
    )
    shared_norm = Normalize(vmin=0, vmax=global_max_disp)

    for idx, (name, _, color) in enumerate(acts):
        ax = fig.add_subplot(gs[0, idx])
        P = proj[name]
        x0 = P[:, alpha_ref_idx, 0]
        y0 = P[:, alpha_ref_idx, 1]
        dx = P[:, alpha_end_idx, 0] - x0
        dy = P[:, alpha_end_idx, 1] - y0
        arrow_len = np.sqrt(dx**2 + dy**2)

        # Faint dot at start position
        ax.scatter(x0, y0, c=color, s=4, alpha=0.25, edgecolors="none", zorder=1)

        # Quiver arrows
        ax.quiver(x0, y0, dx, dy, arrow_len,
                  cmap="inferno", norm=shared_norm,
                  angles='xy', scale_units='xy', scale=1,
                  width=0.005, headwidth=3.5, headlength=4,
                  headaxislength=3.2,
                  alpha=0.75, zorder=2)

        # Faint dot at end position
        ax.scatter(x0 + dx, y0 + dy, c=color, s=2, alpha=0.12,
                   edgecolors="none", zorder=1, marker='.')

        mean_angle = angles[name].mean()
        ax.set_aspect("equal")
        ax.set_xlim(shared_xlim); ax.set_ylim(shared_ylim)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3); spine.set_color("#dddddd")
        ax.set_title(f"{name}", fontsize=9, fontweight="bold", pad=3)

    # Bottom row: histogram spanning all columns
    ax_hist = fig.add_subplot(gs[1, :])
    bins = np.linspace(0, max(angles["GELU"].max() * 1.15, 1.0), 45)

    # Plot ReLU+NELU first (they overlap at 0), then GELU
    for name, _, color in [acts[0], acts[2], acts[1]]:
        ax_hist.hist(angles[name], bins=bins, color=color, alpha=0.6,
                     label=f"{name} ({angles[name].mean():.1f}$^\\circ$)",
                     edgecolor="white", linewidth=0.3)

    alpha_label = f"{alphas[alpha_ref_idx]:.0f}" + " $\\to$ " + f"{alphas[alpha_end_idx]:.0f}"
    ax_hist.set_xlabel(f"Angular deviation ($\\alpha$: {alpha_label})", fontsize=8)
    ax_hist.set_ylabel("Count", fontsize=8)
    ax_hist.legend(frameon=False, fontsize=7, loc="upper center")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.savefig(out_dir / "fig1_direction_combined.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_direction_combined.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_direction_combined.{{png,pdf}}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Style 5: Streamplot — clean, smooth flow lines
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    from scipy.interpolate import RBFInterpolator

    for idx, (name, _, color) in enumerate(acts):
        ax = axes[idx]
        P = proj[name]
        x0 = P[:, alpha_ref_idx, 0]
        y0 = P[:, alpha_ref_idx, 1]
        dx = P[:, alpha_end_idx, 0] - x0
        dy = P[:, alpha_end_idx, 1] - y0

        # Grid covering the data
        pad = 0.03
        xmin, xmax = x0.min() - pad, x0.max() + pad
        ymin, ymax = y0.min() - pad, y0.max() + pad
        grid_n = 50
        gx = np.linspace(xmin, xmax, grid_n)
        gy = np.linspace(ymin, ymax, grid_n)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

        points = np.column_stack([x0, y0])

        if np.sqrt(dx**2 + dy**2).max() > 1e-6:
            # RBF interpolation — smooth field
            rbf_u = RBFInterpolator(points, dx, kernel='thin_plate_spline', smoothing=1.5)
            rbf_v = RBFInterpolator(points, dy, kernel='thin_plate_spline', smoothing=1.5)
            U = rbf_u(grid_pts).reshape(grid_n, grid_n)
            V = rbf_v(grid_pts).reshape(grid_n, grid_n)

            # Mask outside convex hull of data points
            from scipy.spatial import ConvexHull, Delaunay
            hull = Delaunay(points)
            inside = hull.find_simplex(grid_pts) >= 0
            mask = inside.reshape(grid_n, grid_n)
            U[~mask] = 0; V[~mask] = 0

            speed = np.sqrt(U**2 + V**2)
            lw = 0.4 + 1.4 * (speed / max(speed.max(), 1e-10))

            strm = ax.streamplot(gx, gy, U, V,
                                 color=speed, cmap="magma_r",
                                 linewidth=lw, density=1.0,
                                 arrowsize=0.7, arrowstyle='-|>',
                                 broken_streamlines=True,
                                 minlength=0.2)
            if strm.arrows:
                strm.arrows.set_alpha(0.7)
            strm.lines.set_alpha(0.75)
        else:
            ax.text(0.5, 0.5, "Scale invariant\n(zero flow)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="#aaaaaa", style="italic")

        # Scatter points lightly
        ax.scatter(x0, y0, c=color, s=2, alpha=0.15, edgecolors="none", zorder=3)

        mean_angle = angles[name].mean()
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3); spine.set_color("#dddddd")
        ax.text(0.5, -0.12, f"({chr(97+idx)}) {name}  ({mean_angle:.1f}$^\\circ$)",
                transform=ax.transAxes, fontsize=9, fontweight="bold", ha="center")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, wspace=0.08)
    fig.savefig(out_dir / "fig1_streamplot.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "fig1_streamplot.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved fig1_streamplot.{{png,pdf}}")
    plt.close()

    print("\nAll direction field visualizations done.")
    print(f"  Angular deviations (α=1→2):")
    for name, _, _ in acts:
        a = angles[name]
        print(f"    {name}: mean={a.mean():.2f}°  std={a.std():.2f}°  max={a.max():.2f}°")


if __name__ == "__main__":
    main()
