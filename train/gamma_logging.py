"""Logging utilities for tracking gate normalization dynamics during training.

Provides three complementary views into how NELU/NiLU layers evolve:

1. Gamma stats — tracks the per-layer learnable scalar that controls
   the strength of gate normalization. The trajectory of gamma is the
   main diagnostic for whether the activation is "opening up" or
   staying near-linear.

2. Gate entropy — measures how much the gate g(gamma*z/rms(z)) varies
   across channels. High entropy = gate is actively shaping activations;
   low entropy = gate is nearly constant (behaving like identity).

3. Weight norms — standard diagnostic, useful for detecting whether
   gate normalization changes the effective learning rate.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from nelu.activations import NELU, NiLU, _GatedBase, collect_gamma_stats


def log_gamma_stats(model: nn.Module) -> Dict[str, float]:
    """Collect per-layer gamma values and summary statistics.

    Wrapper around nelu.activations.collect_gamma_stats that returns a
    dict suitable for wandb.log() or any other logger.
    """
    return collect_gamma_stats(model)


def measure_gate_entropy(model: nn.Module, probe_batch: torch.Tensor,
                         device: torch.device) -> Dict[str, float]:
    """Measure per-layer mean gate entropy on a fixed probe batch.

    The "gate" is g(gamma*z/rms(z)) — either Phi (NELU) or sigmoid (NiLU).
    We compute the binary entropy H = -p*log(p) - (1-p)*log(1-p) where
    p is the gate value, then average over the batch and spatial dims.

    A fixed probe batch (same inputs every time) lets us track how the
    gate distribution evolves independently of data variation.

    Args:
        model: Model containing NELU/NiLU layers.
        probe_batch: Fixed input tensor (e.g. first batch of validation set).
        device: Device to run the forward pass on.

    Returns:
        Dict mapping layer names to mean gate entropy values.
    """
    probe_batch = probe_batch.to(device)
    entropies = {}
    hooks = []

    layer_idx = 0

    def _make_hook(idx):
        def hook_fn(module, input, output):
            with torch.no_grad():
                z = input[0]
                rho = z.pow(2).mean(dim=-1, keepdim=True).add(module.eps).sqrt()
                t = module.gamma * z / rho

                if isinstance(module, NELU):
                    # Gate is Phi(t / sqrt(2))
                    gate = 0.5 * (1.0 + torch.erf(t * (1.0 / math.sqrt(2.0))))
                else:
                    # Gate is sigmoid(t)
                    gate = torch.sigmoid(t)

                # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
                # Clamp to avoid log(0)
                p = gate.clamp(1e-7, 1.0 - 1e-7)
                h = -p * p.log() - (1.0 - p) * (1.0 - p).log()

                # Average over everything except the layer index
                entropies[f"gate_entropy/layer_{idx}"] = h.mean().item()
        return hook_fn

    for module in model.modules():
        if isinstance(module, _GatedBase):
            handle = module.register_forward_hook(_make_hook(layer_idx))
            hooks.append(handle)
            layer_idx += 1

    if not hooks:
        return {}

    # Run the probe batch through the model
    model.eval()
    with torch.no_grad():
        model(probe_batch)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Add summary stats
    if entropies:
        vals = list(entropies.values())
        entropies["gate_entropy/mean"] = sum(vals) / len(vals)
        entropies["gate_entropy/min"] = min(vals)
        entropies["gate_entropy/max"] = max(vals)

    return entropies


def log_weight_norms(model: nn.Module) -> Dict[str, float]:
    """Return dict of per-layer weight Frobenius norms.

    Useful for tracking whether gate normalization changes the effective
    learning dynamics (e.g. weight growth patterns).
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            norms[f"weight_norm/{name}"] = param.data.norm(2).item()
    return norms
