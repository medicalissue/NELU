"""Render all visualizations from saved landscape data.

Usage:
    python experiments/fig1_render_all.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from pathlib import Path
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "savefig.dpi": 300,
})

out_dir = Path(__file__).resolve().parent.parent / "results"
data = np.load(out_dir / "fig1_landscape_data.npz")
cs = data["coords"]
grids = {"ReLU": data["relu"], "GELU": data["gelu"], "NELU": data["nelu"]}
X, Y = np.meshgrid(cs, cs)
acts = ["ReLU", "GELU", "NELU"]
act_colors = {"ReLU": "#888888", "GELU": "#c0392b", "NELU": "#2471a3"}


# ═══════════════════════════════════════════════════════════════════
# 1. 3D surface (standard)
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(9, 3.2))
vmin = min(g.min() for g in grids.values())
vmax = np.percentile(np.stack(list(grids.values())), 90)

for idx, name in enumerate(acts):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    Z = grids[name]
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0,
                    antialiased=True, alpha=0.9, vmin=vmin, vmax=vmax)
    ax.view_init(elev=30, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none'); ax.grid(False)
    ax.text2D(0.5, -0.02, f"({chr(97+idx)}) {name}", transform=ax.transAxes,
              fontsize=10, fontweight="bold", ha="center")
fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, bottom=0.08, top=0.95)
fig.savefig(out_dir / "fig1_3d.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_3d.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 2. 3D surface with lighting (Mish-style)
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(9, 3.2))
ls = LightSource(azdeg=315, altdeg=45)

for idx, name in enumerate(acts):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    Z = grids[name]
    Z_smooth = gaussian_filter(Z, sigma=1.0)
    # Normalize for shading
    Z_norm = (Z_smooth - Z_smooth.min()) / (Z_smooth.max() - Z_smooth.min() + 1e-10)
    rgb = ls.shade(Z_norm, cmap=cm.coolwarm, blend_mode='soft')
    ax.plot_surface(X, Y, Z_smooth, facecolors=rgb, linewidth=0,
                    antialiased=True, shade=False)
    ax.view_init(elev=35, azim=-55)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none'); ax.grid(False)
    ax.text2D(0.5, -0.02, f"({chr(97+idx)}) {name}", transform=ax.transAxes,
              fontsize=10, fontweight="bold", ha="center")
fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, bottom=0.08, top=0.95)
fig.savefig(out_dir / "fig1_3d_lit.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_3d_lit.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 3. 3D log-scale (sharper features)
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(9, 3.2))
for idx, name in enumerate(acts):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    Z = grids[name]
    Z_log = np.log(Z - Z.min() + 0.01)
    Z_log_smooth = gaussian_filter(Z_log, sigma=0.8)
    ax.plot_surface(X, Y, Z_log_smooth, cmap=cm.viridis, linewidth=0,
                    antialiased=True, alpha=0.9)
    ax.view_init(elev=30, azim=-60)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none'); ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none'); ax.grid(False)
    ax.text2D(0.5, -0.02, f"({chr(97+idx)}) {name}", transform=ax.transAxes,
              fontsize=10, fontweight="bold", ha="center")
fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, bottom=0.08, top=0.95)
fig.savefig(out_dir / "fig1_3d_log.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_3d_log.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 4. 2D contour (Li et al. style)
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
vmin_c = min(g.min() for g in grids.values())
vmax_c = np.percentile(np.stack(list(grids.values())), 85)

for idx, name in enumerate(acts):
    ax = axes[idx]
    Z = grids[name]
    cf = ax.contourf(X, Y, Z, levels=30, cmap="RdYlBu_r", vmin=vmin_c, vmax=vmax_c)
    ax.contour(X, Y, Z, levels=15, colors="white", linewidths=0.3, alpha=0.5)
    ax.plot(0, 0, "k*", markersize=5, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
    ax.text(0.5, -0.08, f"({chr(97+idx)}) {name}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="center")
fig.tight_layout(pad=0.3, w_pad=0.5)
fig.subplots_adjust(bottom=0.12)
fig.savefig(out_dir / "fig1_contour.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_contour.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 5. 2D contour (log scale)
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
for idx, name in enumerate(acts):
    ax = axes[idx]
    Z = grids[name]
    Z_log = np.log(Z - Z.min() + 0.01)
    ax.contourf(X, Y, Z_log, levels=30, cmap="coolwarm")
    ax.contour(X, Y, Z_log, levels=15, colors="black", linewidths=0.2, alpha=0.3)
    ax.plot(0, 0, "k*", markersize=5, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
    ax.text(0.5, -0.08, f"({chr(97+idx)}) {name}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="center")
fig.tight_layout(pad=0.3, w_pad=0.5)
fig.subplots_adjust(bottom=0.12)
fig.savefig(out_dir / "fig1_contour_log.png", bbox_inches="tight", facecolor="white")
print("Saved fig1_contour_log.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6. Hessian analysis (curvature from finite differences)
# ═══════════════════════════════════════════════════════════════════

print("\n=== Hessian / Curvature Analysis ===")
dx = cs[1] - cs[0]

for name in acts:
    Z = grids[name]
    c = Z.shape[0] // 2  # center index

    # 1D cross-section through minimum
    loss_x = Z[c, :]  # horizontal slice
    loss_y = Z[:, c]  # vertical slice

    # Curvature at minimum: d²L/dx² via finite differences
    hess_xx = (loss_x[c+1] - 2*loss_x[c] + loss_x[c-1]) / dx**2
    hess_yy = (loss_y[c+1] - 2*loss_y[c] + loss_y[c-1]) / dx**2
    hess_xy = (Z[c+1,c+1] - Z[c+1,c-1] - Z[c-1,c+1] + Z[c-1,c-1]) / (4*dx**2)

    # Eigenvalues of 2x2 Hessian
    trace = hess_xx + hess_yy
    det = hess_xx * hess_yy - hess_xy**2
    disc = max(trace**2 - 4*det, 0)
    eig1 = (trace + np.sqrt(disc)) / 2
    eig2 = (trace - np.sqrt(disc)) / 2
    condition = eig1 / (abs(eig2) + 1e-10)

    # Sharpness: avg loss in neighborhood vs center
    r = 3  # radius in grid points
    neighborhood = Z[c-r:c+r+1, c-r:c+r+1]
    sharpness = neighborhood.mean() - Z[c, c]

    print(f"\n  {name}:")
    print(f"    Loss at min:    {Z[c,c]:.4f}")
    print(f"    Hess diag:      xx={hess_xx:.2f}  yy={hess_yy:.2f}  xy={hess_xy:.2f}")
    print(f"    Eigenvalues:    λ1={eig1:.2f}  λ2={eig2:.2f}")
    print(f"    Condition:      κ={condition:.2f}")
    print(f"    Sharpness:      {sharpness:.4f}")

# ── Curvature comparison figure ──
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

# (a) 1D cross-sections
ax = axes[0]
c = cs.shape[0] // 2
for name in acts:
    Z = grids[name]
    ax.plot(cs, Z[c, :] - Z[c, c], color=act_colors[name], lw=1.3, label=name)
ax.set_xlabel("Direction 1")
ax.set_ylabel("Loss $-$ Loss$_{\\min}$")
ax.legend(frameon=False, fontsize=7)
ax.set_xlim(cs[0], cs[-1])
ax.set_ylim(bottom=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.5, -0.2, "(a) Cross-section", transform=ax.transAxes,
        fontsize=9, fontweight="bold", ha="center")

# (b) Condition number / sharpness bar
ax = axes[1]
metrics = {}
for name in acts:
    Z = grids[name]
    c_idx = Z.shape[0] // 2
    loss_x = Z[c_idx, :]
    loss_y = Z[:, c_idx]
    hxx = (loss_x[c_idx+1] - 2*loss_x[c_idx] + loss_x[c_idx-1]) / dx**2
    hyy = (loss_y[c_idx+1] - 2*loss_y[c_idx] + loss_y[c_idx-1]) / dx**2
    hxy = (Z[c_idx+1,c_idx+1] - Z[c_idx+1,c_idx-1] - Z[c_idx-1,c_idx+1] + Z[c_idx-1,c_idx-1]) / (4*dx**2)
    tr = hxx + hyy
    dt = hxx*hyy - hxy**2
    disc = max(tr**2 - 4*dt, 0)
    e1 = (tr + np.sqrt(disc)) / 2
    e2 = (tr - np.sqrt(disc)) / 2
    metrics[name] = {"eig1": e1, "eig2": e2, "kappa": e1/(abs(e2)+1e-10)}

x_pos = np.arange(len(acts))
bars = ax.bar(x_pos, [metrics[n]["kappa"] for n in acts],
              color=[act_colors[n] for n in acts], width=0.5, edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels(acts)
ax.set_ylabel("Condition number $\\kappa$")
for bar, name in zip(bars, acts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{metrics[name]['kappa']:.1f}", ha="center", fontsize=7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.5, -0.2, "(b) Hessian condition number", transform=ax.transAxes,
        fontsize=9, fontweight="bold", ha="center")

fig.tight_layout(pad=0.4)
fig.subplots_adjust(bottom=0.2)
fig.savefig(out_dir / "fig1_hessian.png", bbox_inches="tight", facecolor="white")
print("\nSaved fig1_hessian.png")
plt.close()

print("\nAll renders done.")
