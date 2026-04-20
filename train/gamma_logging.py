"""Logging utilities for tracking gate normalization dynamics during training.

Provides three complementary views into how NELU/NiLU layers evolve:

1. Gamma stats — tracks the per-layer learnable scalar that controls
   the strength of gate normalization. The trajectory of gamma is the
   main diagnostic for whether the activation is "opening up" or
   staying near-linear.

2. Gate stats (entropy + variance) — measures how the gate g(z) varies
   across channels/spatial positions. Works for ALL activation types:
   - NELU/NiLU: gate = Phi/sigmoid(gamma*z/rms(z))  [normalized]
   - GELU:      gate = Phi(z)                         [unnormalized]
   - SiLU:      gate = sigmoid(z)                     [unnormalized]
   Key insight: GELU/SiLU gate couples with ||W||; NELU/NiLU gate is
   decoupled (controlled by gamma alone).

3. Weight norms — standard diagnostic for tracking scale dynamics.
"""

import math
from typing import Dict

import torch
import torch.nn as nn

from nelu.activations import NELU, NiLU, _GatedBase, collect_gamma_stats


def log_gamma_stats(model: nn.Module) -> Dict[str, float]:
    return collect_gamma_stats(model)


def _record_gate(out: Dict[str, float], gate: torch.Tensor, idx: int) -> None:
    p = gate.clamp(1e-7, 1.0 - 1e-7)
    h = -p * p.log() - (1.0 - p) * (1.0 - p).log()
    out[f"gate_entropy/layer_{idx}"] = h.mean().item()
    out[f"gate_var/layer_{idx}"] = gate.var().item()


def measure_gate_stats(model: nn.Module, probe_batch: torch.Tensor,
                       device: torch.device) -> Dict[str, float]:
    """Compute gate entropy and variance for all activation layers.

    Works for NELU/NiLU (normalized gate) AND GELU/SiLU (unnormalized gate).
    Running on a fixed probe batch lets us track evolution independently of
    data variation.

    Returns keys:
        gate_entropy/layer_{i}, gate_entropy/mean, min, max
        gate_var/layer_{i},     gate_var/mean,     min, max
    """
    probe_batch = probe_batch.to(device)
    stats: Dict[str, float] = {}
    hooks = []
    layer_idx = 0

    def _make_gated_hook(idx):
        def _fn(module, inp, _out):
            with torch.no_grad():
                z = inp[0]
                rho = module.compute_rms(z)
                t = module.gamma * z / rho
                if isinstance(module, NELU):
                    gate = 0.5 * (1.0 + torch.erf(t * (1.0 / math.sqrt(2.0))))
                else:
                    gate = torch.sigmoid(t)
                _record_gate(stats, gate, idx)
        return _fn

    def _make_gelu_hook(idx):
        def _fn(_module, inp, _out):
            with torch.no_grad():
                z = inp[0]
                gate = 0.5 * (1.0 + torch.erf(z * (1.0 / math.sqrt(2.0))))
                _record_gate(stats, gate, idx)
        return _fn

    def _make_silu_hook(idx):
        def _fn(_module, inp, _out):
            with torch.no_grad():
                gate = torch.sigmoid(inp[0])
                _record_gate(stats, gate, idx)
        return _fn

    for module in model.modules():
        if isinstance(module, _GatedBase):
            hooks.append(module.register_forward_hook(_make_gated_hook(layer_idx)))
            layer_idx += 1
        elif isinstance(module, nn.GELU):
            hooks.append(module.register_forward_hook(_make_gelu_hook(layer_idx)))
            layer_idx += 1
        elif isinstance(module, nn.SiLU):
            hooks.append(module.register_forward_hook(_make_silu_hook(layer_idx)))
            layer_idx += 1

    if not hooks:
        return {}

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(probe_batch)
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


# Keep old name for backward compat
def measure_gate_entropy(model: nn.Module, probe_batch: torch.Tensor,
                         device: torch.device) -> Dict[str, float]:
    return measure_gate_stats(model, probe_batch, device)


def log_weight_norms(model: nn.Module) -> Dict[str, float]:
    """Per-layer weight Frobenius norms + summary stats."""
    norms = {}
    all_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            n = param.data.norm(2).item()
            norms[f"weight_norm/{name}"] = n
            all_norms.append(n)
    if all_norms:
        norms["weight_norm/mean"] = sum(all_norms) / len(all_norms)
        norms["weight_norm/total"] = sum(all_norms)
    return norms
