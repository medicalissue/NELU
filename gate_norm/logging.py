"""Runtime statistics for Gate Normalization layers.

Walks an ``nn.Module`` tree, collects the γ scalar from every Gate
Normalization instance, and returns a flat dict of scalars suitable for
``wandb.log``. The caller decides when to log (typically once per epoch).

Key shape: ``<prefix>/gamma/layer_<i>`` for per-module values and
``<prefix>/gamma/<agg>`` for aggregates, where ``agg ∈ {mean, min, max,
std, abs_mean, n_negative, n_modules}``.
"""

from __future__ import annotations

import torch.nn as nn


def _aggregate(values: list[float], out: dict, key: str) -> None:
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(1, n - 1)
    out[f"{key}/mean"] = mean
    out[f"{key}/min"] = min(values)
    out[f"{key}/max"] = max(values)
    out[f"{key}/std"] = var ** 0.5
    out[f"{key}/abs_mean"] = sum(abs(x) for x in values) / n
    out[f"{key}/n_negative"] = float(sum(1 for x in values if x < 0))
    out[f"{key}/n_modules"] = float(n)


def collect_gamma_stats(model: nn.Module, prefix: str = "gate_norm") -> dict:
    """Collect per-module γ (and β when present) values with aggregates."""
    gammas: list[float] = []
    betas: list[float] = []
    out: dict[str, float] = {}
    for m in model.modules():
        if not getattr(m, "_gate_norm_module", False):
            continue
        if hasattr(m, "gamma"):
            g = m.gamma.detach().float().item()
            out[f"{prefix}/gamma/layer_{len(gammas)}"] = g
            gammas.append(g)
        # NELU_LN / NiLU_LN carry a learnable β that shifts the gate's
        # operating point. Logged separately so the trade-off slider's
        # trajectory is visible in W&B.
        if hasattr(m, "beta"):
            b = m.beta.detach().float().item()
            out[f"{prefix}/beta/layer_{len(betas)}"] = b
            betas.append(b)

    if gammas:
        _aggregate(gammas, out, f"{prefix}/gamma")
    if betas:
        _aggregate(betas, out, f"{prefix}/beta")
    return out
