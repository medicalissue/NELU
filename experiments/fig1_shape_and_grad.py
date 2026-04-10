"""Figure 1: GELU's behaviour drifts with pre-activation scale; NELU's does not.

Top row — forward angular drift (cosine-angle deviation).
  For unit Gaussian z ∈ R^128, compute the output direction of f(ρ·z)
  for ρ ∈ {0.5, 1, 2, 4} and measure the angle (deg) to the reference
  direction at ρ=1.
  (a) GELU: directions drift with scale — non-trivial angular spread
  (b) NELU: directions are exact (spike at 0) — scale equivariance

Bottom row — distribution of the derivative f'(z) for z ~ N(0, ρ²).
  (c) GELU: graded at ρ=1, binary at ρ=4
  (d) NELU: histograms overlap (scale-invariant derivative)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm as _norm

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

Phi = _norm.cdf
phi = _norm.pdf

# ── Color palette: ρ=1 is blue, increasing ρ → warmer ──────────────
rhos_all = [0.5, 1.0, 2.0, 4.0]
colors = {
    0.5: "#76b7b2",   # teal
    1.0: "#4e79a7",   # blue (canonical, ρ=1)
    2.0: "#f28e2b",   # orange
    4.0: "#e15759",   # red
}

fig, axes = plt.subplots(2, 2, figsize=(7, 4.6))

np.random.seed(42)

# ── Forward angular drift (top row) ────────────────────────────────
#
# For each base sample z ∈ R^d, compute direction of f(ρz) and its
# angle to direction at ρ_ref=1.  d=128 matches Figure 2.
#
d = 128
n_samples = 5000
z_base = np.random.randn(n_samples, d)

def gelu_vec(z):
    return z * Phi(z)

def nelu_vec(z, eps=1e-6):
    rms = np.sqrt((z ** 2).mean(axis=-1, keepdims=True) + eps)
    return z * Phi(z / rms)

def angle_drift(fn, z_base, rho_test, rho_ref=1.0):
    y_ref = fn(rho_ref * z_base)
    y_test = fn(rho_test * z_base)
    y_ref /= np.linalg.norm(y_ref, axis=-1, keepdims=True).clip(min=1e-12)
    y_test /= np.linalg.norm(y_test, axis=-1, keepdims=True).clip(min=1e-12)
    cos = np.clip((y_ref * y_test).sum(-1), -1 + 1e-7, 1 - 1e-7)
    return np.degrees(np.arccos(cos))

drift_range = (-0.2, 12)
n_bins_drift = 55

# ═══════════════════════════════════════════════════════════════════
# (a) GELU angular drift — directions change with ρ
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 0]
for rho in rhos_all:
    angles = angle_drift(gelu_vec, z_base, rho)
    lbl = f"$\\rho$={rho:g}" + (" (ref)" if rho == 1.0 else "")
    ax.hist(angles, bins=n_bins_drift, range=drift_range, density=True,
            histtype="step", linewidth=1.7 if rho == 1.0 else 1.2,
            color=colors[rho], label=lbl)

ax.set_xlabel(r"$\angle\,(\mathrm{GELU}(\rho z),\, \mathrm{GELU}(z))$  (deg)")
ax.set_ylabel("density")
ax.set_xlim(drift_range)
ax.set_yscale("log")
ax.set_ylim(3e-3, 50)
ax.legend(frameon=False, loc="upper right", borderpad=0, labelspacing=0.3)

# ═══════════════════════════════════════════════════════════════════
# (b) NELU angular drift — directions are identical (spike at 0)
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 1]
for rho in rhos_all:
    angles = angle_drift(nelu_vec, z_base, rho)
    lbl = f"$\\rho$={rho:g}"
    ax.hist(angles, bins=n_bins_drift, range=drift_range, density=True,
            histtype="step", linewidth=1.7 if rho == 1.0 else 1.2,
            color=colors[rho], label=lbl,
            alpha=0.9 if rho != 1.0 else 1.0)

ax.set_xlabel(r"$\angle\,(\mathrm{NELU}(\rho z),\, \mathrm{NELU}(z))$  (deg)")
ax.set_ylabel("density")
ax.set_xlim(drift_range)
ax.set_yscale("log")
ax.set_ylim(3e-3, 50)
ax.legend(frameon=False, loc="upper right", borderpad=0, labelspacing=0.3)

# ── Re-sample scalar z for the gradient histograms ────────────────
n = 200_000

# ═══════════════════════════════════════════════════════════════════
# (c) GELU gradient histogram — graded at ρ=1, binary at ρ=4
# ═══════════════════════════════════════════════════════════════════
ax = axes[1, 0]
grad_range = (-0.15, 1.15)
n_bins_grad = 60

for rho in [1.0, 4.0]:
    z = np.random.normal(0, rho, n)
    grads = Phi(z) + z * phi(z)       # GELU'(z)
    ax.hist(grads, bins=n_bins_grad, range=grad_range, density=True,
            alpha=0.55, color=colors[rho], label=f"$\\rho$={rho:g}",
            edgecolor="none")

ax.set_xlabel("GELU$'(z)$")
ax.set_ylabel("density")
ax.set_xlim(grad_range)
ax.set_ylim(0, 6)
ax.legend(frameon=False, loc="upper center")

# ═══════════════════════════════════════════════════════════════════
# (d) NELU gradient histogram — overlapping
# ═══════════════════════════════════════════════════════════════════
ax = axes[1, 1]
for rho in [1.0, 4.0]:
    z = np.random.normal(0, rho, n)
    t = z / rho                        # NELU'(z) = h(z/ρ)
    grads = Phi(t) + t * phi(t)
    ax.hist(grads, bins=n_bins_grad, range=grad_range, density=True,
            alpha=0.55, color=colors[rho], label=f"$\\rho$={rho:g}",
            edgecolor="none")

ax.set_xlabel("NELU$'(z)$")
ax.set_ylabel("density")
ax.set_xlim(grad_range)
ax.set_ylim(0, 6)
ax.legend(frameon=False, loc="upper center")

# ═══════════════════════════════════════════════════════════════════
# Cleanup & subplot labels
# ═══════════════════════════════════════════════════════════════════
for i, ax in enumerate(axes.flat):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(width=0.4, length=3)
    ax.text(0.5, -0.30, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="center")

fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.4)
fig.subplots_adjust(bottom=0.12)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "fig1_shape_and_grad.png",
            bbox_inches="tight", facecolor="white")
fig.savefig(out_dir / "fig1_shape_and_grad.pdf",
            bbox_inches="tight", facecolor="white")
print(f"Saved {out_dir}/fig1_shape_and_grad.{{png,pdf}}")
plt.close()
