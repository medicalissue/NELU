"""Runtime diagnostics for Gate Normalization layers.

Three complementary probes are exposed, each returning a flat dict of scalars
suitable for ``wandb.log``:

* :func:`gamma_stats` — per-layer and aggregate statistics of the learnable
  γ parameters. Surfaces the main learning signal of Gate Normalization.
* :func:`gate_stats` — entropy and variance of the gate values ``g(·)`` for
  every gated activation, computed on a fixed probe batch. Works for both
  NELU/NiLU *and* the GELU/SiLU baselines.
* :func:`weight_norms` — Frobenius norms of all ≥2-D parameters. Standard
  scale-dynamics diagnostic.

All probes unwrap ``torch.compile`` wrappers transparently.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from gate_norm.core import GateNorm
from gate_norm.layout import resolve_axes
from gate_norm.logging import collect_gamma_stats
from gate_norm.stats import layer_stats

from .swap import GELU_TYPES, SILU_TYPES


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap ``torch.compile`` (``OptimizedModule``) so hooks fire normally."""
    return getattr(model, "_orig_mod", model)


# ── γ statistics ─────────────────────────────────────────────────────────


def gamma_stats(model: nn.Module) -> Dict[str, float]:
    """Delegate to :func:`gate_norm.collect_gamma_stats`."""
    return collect_gamma_stats(_unwrap(model))


# ── Gate statistics (probe-batch based) ──────────────────────────────────


def _record(out: Dict[str, float], gate: torch.Tensor, idx: int) -> None:
    p = gate.clamp(1e-7, 1.0 - 1e-7)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    out[f"gate_entropy/layer_{idx}"] = entropy.mean().item()
    out[f"gate_var/layer_{idx}"] = gate.var().item()


def _gated_hook(idx: int, stats: Dict[str, float]):
    """Reconstruct the gate value seen inside the module's forward.

    Standard ``GateNorm`` (NELU/NiLU) computes ``g(γ · z / rms(z))``. The
    LN-β variant uses LayerNorm-style stats plus a β shift, and the
    affine variant skips normalization entirely. Channel-wise variants
    use a length-C γ, β. We dispatch on type to mirror the right forward;
    if the module is none of the known shapes we just measure the
    output / input ratio as a stand-in gate.
    """
    # Lazy import to avoid circular deps at module load time.
    from gate_norm.ln_beta import _GateNormLN
    from gate_norm.affine import _GateAffine, _GateAffineCW

    def fn(module: GateNorm, inp, out):
        with torch.no_grad():
            z = inp[0]
            z32 = z.float() if z.dtype != torch.float32 else z
            if isinstance(module, _GateAffineCW):
                # Channel-wise affine: γ_c, β_c are length-C vectors
                shape = (1, z.size(1), 1, 1) if z.ndim == 4 else (1,) * (z.ndim - 1) + (z.size(-1),)
                gate = type(module)._gate_python(
                    module.gamma.view(shape) * z32 + module.beta.view(shape)
                )
            elif isinstance(module, _GateAffine):
                gate = type(module)._gate_python(module.gamma * z32 + module.beta)
            elif isinstance(module, _GateNormLN):
                axes = resolve_axes(z.ndim, module.norm_axes)
                mu = z32.mean(dim=axes, keepdim=True)
                var = z32.var(dim=axes, keepdim=True, unbiased=False)
                rsigma = (var + module.eps).rsqrt()
                t = module.gamma * (z32 - mu) * rsigma + module.beta
                gate = type(module)._gate_python(t)
            else:
                # Standard NELU / NiLU: scalar γ, RMS-only normalize
                axes = resolve_axes(z.ndim, module.norm_axes)
                rsigma = layer_stats(z, axes, module.eps)
                t = module.gamma * z32 * rsigma
                gate = type(module)._gate_python(t)
            _record(stats, gate, idx)
    return fn


def _gelu_hook(idx: int, stats: Dict[str, float]):
    def fn(_module, inp, _out):
        with torch.no_grad():
            gate = 0.5 * (1.0 + torch.erf(inp[0] * _INV_SQRT2))
            _record(stats, gate, idx)
    return fn


def _silu_hook(idx: int, stats: Dict[str, float]):
    def fn(_module, inp, _out):
        with torch.no_grad():
            _record(stats, torch.sigmoid(inp[0]), idx)
    return fn


def gate_stats(
    model: nn.Module, probe: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Collect gate entropy and variance on a fixed probe batch."""
    model = _unwrap(model)
    probe = probe.to(device)
    stats: Dict[str, float] = {}
    hooks = []
    idx = 0

    for m in model.modules():
        if isinstance(m, GateNorm):
            hooks.append(m.register_forward_hook(_gated_hook(idx, stats)))
            idx += 1
        elif isinstance(m, GELU_TYPES):
            hooks.append(m.register_forward_hook(_gelu_hook(idx, stats)))
            idx += 1
        elif isinstance(m, SILU_TYPES):
            hooks.append(m.register_forward_hook(_silu_hook(idx, stats)))
            idx += 1

    if not hooks:
        return {}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(probe)
    finally:
        if was_training:
            model.train()
        for h in hooks:
            h.remove()

    entropies = [v for k, v in stats.items() if k.startswith("gate_entropy/layer_")]
    variances = [v for k, v in stats.items() if k.startswith("gate_var/layer_")]
    if entropies:
        stats["gate_entropy/mean"] = sum(entropies) / len(entropies)
        stats["gate_entropy/min"] = min(entropies)
        stats["gate_entropy/max"] = max(entropies)
    if variances:
        stats["gate_var/mean"] = sum(variances) / len(variances)
        stats["gate_var/min"] = min(variances)
        stats["gate_var/max"] = max(variances)
    return stats


# ── Weight norms ─────────────────────────────────────────────────────────


def weight_norms(model: nn.Module) -> Dict[str, float]:
    """Frobenius norm of every ≥2-D parameter, plus mean and total."""
    model = _unwrap(model)
    out: Dict[str, float] = {}
    values = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim >= 2:
            v = p.data.norm(2).item()
            out[f"weight_norm/{name}"] = v
            values.append(v)
    if values:
        out["weight_norm/mean"] = sum(values) / len(values)
        out["weight_norm/total"] = sum(values)
    return out
