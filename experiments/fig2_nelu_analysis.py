"""Figure 2: NELU activation and gradient behavior.

(a) NELU shape at different rms scales.
(b) GELU gradient distribution — collapses to binary at high ρ.
(c) NELU gradient distribution — stable across ρ.
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

colors_rho = {"1": "#4e79a7", "4": "#e15759"}

fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))

# ── (a) Activation shape ─────────────────────────────────────────

ax = axes[0]
z = np.linspace(-6, 6, 800)
rhos_a = [0.5, 1.0, 2.0, 4.0]
colors_a = ["#4e79a7", "#e15759", "#59a14f", "#9c755f"]

for rho, c in zip(rhos_a, colors_a):
    y = z * Phi(z / rho)
    lw = 1.4 if rho == 1.0 else 1.0
    lbl = f"$\\rho$={rho}"
    if rho == 1.0:
        lbl += " (GELU)"
    ax.plot(z, y, color=c, lw=lw, label=lbl)

ax.axhline(0, color="#e0e0e0", lw=0.4, zorder=0)
ax.axvline(0, color="#e0e0e0", lw=0.4, zorder=0)
ax.set_xlabel("$z$")
ax.set_ylabel("NELU$(z)$")
ax.legend(frameon=False, loc="upper left", borderpad=0, labelspacing=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(-1.5, 6)

# ── (b) GELU gradient histogram ─────────────────────────────────

ax = axes[1]
np.random.seed(42)
n = 50000

for rho, label, c in [(1.0, "$\\rho$=1", colors_rho["1"]),
                       (4.0, "$\\rho$=4", colors_rho["4"])]:
    samples = np.random.normal(0, rho, n)
    grads = Phi(samples) + samples * phi(samples)  # GELU'(z)
    ax.hist(grads, bins=80, range=(-0.2, 1.15), density=True,
            alpha=0.55, color=c, label=label, edgecolor="none")

ax.set_xlabel("GELU$'(z)$")
ax.set_ylabel("density")
ax.set_xlim(-0.2, 1.15)
ax.legend(frameon=False, loc="upper center")

# ── (c) NELU gradient histogram ─────────────────────────────────

ax = axes[2]

for rho, label, c in [(1.0, "$\\rho$=1", colors_rho["1"]),
                       (4.0, "$\\rho$=4", colors_rho["4"])]:
    samples = np.random.normal(0, rho, n)
    t = samples / rho  # z / rms = normalized
    grads = Phi(t) + t * phi(t)  # NELU'(z) = h(z/ρ)
    ax.hist(grads, bins=80, range=(-0.2, 1.15), density=True,
            alpha=0.55, color=c, label=label, edgecolor="none")

ax.set_xlabel("NELU$'(z)$")
ax.set_ylabel("")
ax.set_xlim(-0.2, 1.15)
ax.legend(frameon=False, loc="upper center")

# ── Cleanup + labels ─────────────────────────────────────────────

for i, ax in enumerate(axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(width=0.4, length=3)
    ax.text(0.5, -0.28, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="center")

fig.tight_layout(pad=0.3, w_pad=1.0)
fig.subplots_adjust(bottom=0.24)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "fig2_nelu_analysis.png", bbox_inches="tight", facecolor="white")
fig.savefig(out_dir / "fig2_nelu_analysis.pdf", bbox_inches="tight", facecolor="white")
print(f"Saved {out_dir}/fig2_nelu_analysis.{{png,pdf}}")
plt.close()
