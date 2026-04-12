"""Training diagnostics for NELU/NiLU — captures pre-collapse signals.

Usage (in deit-train/main.py):
    from nelu.diagnostics import NELUDiagnostics
    diag = NELUDiagnostics(model, log_every=50, wandb_run=_wandb_run)

Then in engine.py train_one_epoch, after backward:
    diag.step(global_step)      # logs grad norms + cached hook data
    diag.step_forward_done()    # resets per-step hook accumulators
"""

import torch
import torch.nn as nn
from collections import defaultdict


class NELUDiagnostics:
    def __init__(self, model, log_every=50, wandb_run=None):
        self.model = model
        self.log_every = log_every
        self.wandb_run = wandb_run
        self._step_count = 0
        self._hook_data = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on NELU-like modules and attention layers."""
        for name, mod in self.model.named_modules():
            # ρ capture for any module with .eps (NELU variants)
            if hasattr(mod, 'eps') and hasattr(mod, 'gamma'):
                h = mod.register_forward_hook(self._make_rho_hook(name))
                self._handles.append(h)
            elif hasattr(mod, 'eps') and (
                mod.__class__.__name__.startswith('NELU') or
                mod.__class__.__name__.startswith('NiLU')
            ):
                h = mod.register_forward_hook(self._make_rho_hook(name))
                self._handles.append(h)

            # Attention weight capture (DeiT uses timm's Attention)
            if mod.__class__.__name__ == 'Attention' and hasattr(mod, 'qkv'):
                h = mod.register_forward_hook(self._make_attn_hook(name))
                self._handles.append(h)

    def _make_rho_hook(self, name):
        data = self._hook_data

        def hook(module, inp, out):
            z = inp[0].detach()
            if z.dim() == 4:
                dim = (1, 2, 3)
            else:
                dim = -1
            eps = getattr(module, 'eps', 1e-6)
            rho = z.pow(2).mean(dim=dim, keepdim=True).add(eps).sqrt()
            data[f'rho_mean/{name}'] = rho.mean().item()
            data[f'rho_std/{name}'] = rho.std().item()
            data[f'rho_min/{name}'] = rho.min().item()
            data[f'rho_max/{name}'] = rho.max().item()

            # Activation stats
            data[f'act_mean/{name}'] = out.detach().mean().item()
            data[f'act_std/{name}'] = out.detach().std().item()
            data[f'act_absmax/{name}'] = out.detach().abs().max().item()

            # gamma stats if present
            if hasattr(module, 'gamma') and module.gamma is not None:
                g = module.gamma.detach()
                data[f'gamma_mean/{name}'] = g.mean().item()
                data[f'gamma_std/{name}'] = g.std().item()
                data[f'gamma_min/{name}'] = g.min().item()
                data[f'gamma_max/{name}'] = g.max().item()

        return hook

    def _make_attn_hook(self, name):
        data = self._hook_data

        def hook(module, inp, out):
            # Capture attention entropy from QKV
            # DeiT Attention: self.qkv(x) → (B, N, 3*C) → reshape → attn
            # We can't easily get attn weights without modifying Attention.
            # Instead, capture the pre-softmax scale of QK^T via input stats.
            x = inp[0].detach()
            data[f'attn_input_std/{name}'] = x.std().item()
            data[f'attn_input_absmax/{name}'] = x.abs().max().item()

        return hook

    def step(self, global_step):
        """Call after loss.backward(). Logs grad norms + hook data."""
        self._step_count += 1
        if self._step_count % self.log_every != 0:
            return

        log_dict = {'diag_step': global_step}

        # Per-layer gradient norms
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gn = param.grad.detach().norm().item()
                log_dict[f'grad_norm/{name}'] = gn

                # Flag if gradient is huge
                if gn > 1e4:
                    print(f"  [DIAG] WARNING: grad_norm/{name} = {gn:.1f}")

        # Add hook data (rho, gamma, act stats)
        log_dict.update(self._hook_data)

        # Overall grad norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm().item() ** 2
        log_dict['grad_norm_total'] = total_norm ** 0.5

        # Log
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(log_dict, step=global_step)
            except Exception:
                pass

        # Also print summary
        rho_mins = {k: v for k, v in self._hook_data.items() if k.startswith('rho_min/')}
        if rho_mins:
            worst_rho = min(rho_mins.values())
            worst_key = min(rho_mins, key=rho_mins.get)
            if worst_rho < 0.01:
                print(f"  [DIAG] WARNING: {worst_key} = {worst_rho:.6f}")

    def step_forward_done(self):
        """Reset per-step hook accumulators. Call at the start of each step."""
        self._hook_data.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
