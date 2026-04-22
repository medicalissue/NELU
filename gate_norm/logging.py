"""Runtime statistics for Gate Normalization layers.

Exports a single helper that walks an ``nn.Module`` tree, collects the γ
scalars from every Gate Normalization instance, and returns a flat dict of
scalars suitable for ``wandb.log``. The caller decides when to log
(typically once per epoch).
"""

from __future__ import annotations

import torch.nn as nn


def collect_gamma_stats(model: nn.Module, prefix: str = "gate_norm") -> dict:
    """Collect per-module γ values and their aggregates."""
    gammas: list[float] = []
    out: dict[str, float] = {}
    for m in model.modules():
        if not getattr(m, "_gate_norm_module", False):
            continue
        if not hasattr(m, "gamma"):
            continue
        g = m.gamma.detach().float().item()
        out[f"{prefix}/gamma/layer_{len(gammas)}"] = g
        gammas.append(g)

    if not gammas:
        return {}

    n = len(gammas)
    mean = sum(gammas) / n
    var = sum((x - mean) ** 2 for x in gammas) / max(1, n - 1)
    out[f"{prefix}/gamma/mean"] = mean
    out[f"{prefix}/gamma/min"] = min(gammas)
    out[f"{prefix}/gamma/max"] = max(gammas)
    out[f"{prefix}/gamma/std"] = var ** 0.5
    out[f"{prefix}/gamma/abs_mean"] = sum(abs(x) for x in gammas) / n
    out[f"{prefix}/gamma/n_negative"] = float(sum(1 for x in gammas if x < 0))
    out[f"{prefix}/gamma/n_modules"] = float(n)
    return out
