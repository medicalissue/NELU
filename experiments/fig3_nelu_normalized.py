"""Figure 3: NELU dynamically adapts to the input scale ρ.

Single panel with dual y-axis:
  solid lines  — NELU(z) = z · Φ(z / ρ)  (left y-axis)
  dashed lines — NELU'(z) = Φ(z/ρ) + (z/ρ) · ϕ(z/ρ)  (right y-axis)

  small ρ → sharp gate (approaches ReLU)
  large ρ → soft gate (approaches z/2)

Every derivative is bounded by e/√(2π) ≈ 1.084 regardless of ρ.
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

# Match fig1 palette — ρ=1 is canonical blue
rhos = [0.5, 1.0, 2.0, 4.0]
colors = {
    0.5: "#76b7b2",   # teal
    1.0: "#4e79a7",   # blue (canonical, ρ=1)
    2.0: "#f28e2b",   # orange
    4.0: "#e15759",   # red
}

fig, ax1 = plt.subplots(1, 1, figsize=(4.5, 3.2))
ax2 = ax1.twinx()

z = np.linspace(-6, 6, 1000)

# ── Lines ────────────────────────────────────────────────────────
for rho in rhos:
    y_act  = z * Phi(z / rho)
    t      = z / rho
    y_grad = Phi(t) + t * phi(t)

    lw = 1.7 if rho == 1.0 else 1.1
    lbl = f"$\\rho$={rho:g}" + (" (std)" if rho == 1.0 else "")

    # Activation (solid)
    ax1.plot(z, y_act, color=colors[rho], lw=lw, label=lbl, zorder=3)
    # Derivative (dashed) — no label, color-matched
    ax2.plot(z, y_grad, color=colors[rho], lw=lw * 0.85,
             linestyle=(0, (4, 2)), alpha=0.9, zorder=2)

# Derivative upper bound
bound = np.e / np.sqrt(2 * np.pi)
ax2.axhline(bound, color="#888888", lw=0.5, linestyle=":", zorder=1)
ax2.text(5.8, bound + 0.02,
         f"$e/\\sqrt{{2\\pi}}\\approx{bound:.3f}$",
         fontsize=6.5, color="#666666", ha="right", va="bottom")

# Zero reference lines
ax1.axhline(0, color="#e0e0e0", lw=0.4, zorder=0)
ax1.axvline(0, color="#e0e0e0", lw=0.4, zorder=0)

# ── Axes ─────────────────────────────────────────────────────────
ax1.set_xlabel("$z$")
ax1.set_ylabel("NELU$(z) = z\\,\\Phi(z/\\rho)$", color="#222222")
ax2.set_ylabel("NELU$'(z)$  (dashed)", color="#666666")

ax1.set_xlim(-6, 6)
ax1.set_ylim(-1.5, 6)
ax2.set_ylim(-0.25, 1.3)

ax1.tick_params(axis="y", colors="#222222")
ax2.tick_params(axis="y", colors="#666666")

# ── Legend ───────────────────────────────────────────────────────
# Activation legend from ax1; also add one "NELU'(z)" dashed marker
from matplotlib.lines import Line2D
leg1 = ax1.legend(frameon=False, loc="upper left",
                  borderpad=0, labelspacing=0.3, handlelength=1.6)
# Dashed-style indicator
style_handles = [
    Line2D([0], [0], color="#444444", lw=1.3, linestyle="-",
           label="NELU"),
    Line2D([0], [0], color="#444444", lw=1.1, linestyle=(0, (4, 2)),
           label="NELU$'$"),
]
ax1.add_artist(leg1)
ax1.legend(handles=style_handles, frameon=False, loc="lower right",
           borderpad=0, labelspacing=0.3, handlelength=1.8)

# ── Spines ───────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["right"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(width=0.4, length=3)

fig.tight_layout(pad=0.4)

out_dir = Path(__file__).resolve().parent.parent / "results"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "fig3_nelu_normalized.png",
            bbox_inches="tight", facecolor="white")
fig.savefig(out_dir / "fig3_nelu_normalized.pdf",
            bbox_inches="tight", facecolor="white")
print(f"Saved {out_dir}/fig3_nelu_normalized.{{png,pdf}}")
plt.close()
