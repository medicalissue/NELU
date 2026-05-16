"""Per-layer mechanism diagnostics for the data-scarce thesis.

Measures, for every activation site in a trained MedMNIST model, the two
failure modes the paper attributes to baseline activations under data
scarcity and claims NELU/NiLU prevent:

  M1 (shallow-layer overfitting)
      train-vs-test gap of the per-layer collapse metric, and per-channel
      activation variance. A baseline that overfits shallow layers shows a
      large train/test divergence early in the network.

  M2 (deep-layer collapse / linearization toward a shallow model)
      * V^ℓ = W^ℓ / B^ℓ  — the Hui, Belkin & Nakkiran (2022, §3.1)
        normalized within-/between-class variance ratio, computed PER
        LAYER. Lower V^ℓ = more collapse. We cite their metric only;
        they explicitly reject a collapse→generalization causal claim,
        so the causal story is shown on our own data, not attributed to
        them. (formula: W^ℓ = E||hℓ(x)-μ_y||², B^ℓ = E_i||μ_i-μ||²)
      * gate saturation rate — fraction of gate values in the dead
        (g<ε, output→0) or pass-through (g>1-ε, output→x≈linear) regime.
        Self-gated activations (NELU=Φ, NiLU=σ, GELU=Φ, SiLU=σ) all emit
        a [0,1] gate, so this is one comparable scale across them; ReLU
        is handled as a hard 0/1 gate.
      * per-layer effective rank — collapse/linearization drives the
        feature covariance toward low rank.

The headline experiment is to run this across MedMNIST datasets ordered
by training-set size and show baseline V^ℓ / eff-rank collapse in deep
layers as data shrinks, while NELU/NiLU hold them up.

Pure metric helpers (effective_rank, participation_ratio) are reused
from eval.cifar.geometry to keep one definition of each quantity.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.cifar.geometry import effective_rank, participation_ratio
from gate_norm import NELU, NiLU
from gate_norm.layout import resolve_axes
from train.medmnist import (
    INFO,
    _task_of,
    build_model,
    get_dataloaders,
)


# ── Gate reconstruction ────────────────────────────────────────────────
#
# The activation modules do not expose the gate; we recompute it from the
# captured input exactly as the module's forward does, so the saturation
# metric is faithful (no approximation).

def _gate_of(module: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Reconstruct the [0,1] gate a self-gated activation applies to ``z``.

    NELU/NiLU: gate = g(γ·LN_c(z) + β), g = Φ (NELU) or σ (NiLU), with
        the same per-position normalize the module uses.
    GELU:      gate = Φ(z)        SiLU: gate = σ(z)
    ReLU:      gate = 1[z>0]      (hard 0/1)
    Returns a tensor broadcastable to ``z``'s shape.
    """
    tname = type(module).__name__
    if tname in ("NELU", "NiLU"):
        axes = resolve_axes(z.ndim, module.norm_axes)
        if z.ndim == 4:
            shape = (1, z.size(1), 1, 1)
        else:
            shape = (1,) * (z.ndim - 1) + (z.size(-1),)
        z32 = z.float()
        mu = z32.mean(dim=axes, keepdim=True)
        var = z32.var(dim=axes, keepdim=True, unbiased=False)
        z_norm = (z32 - mu) * (var + module.eps).rsqrt()
        gamma = module.gamma.view(shape)
        beta = module.beta.view(shape)
        return type(module)._gate_python(gamma * z_norm + beta)
    if tname == "GELU":
        # x·Φ(x) ⇒ gate = Φ(x) = 0.5(1+erf(x/√2))
        return 0.5 * (1.0 + torch.erf(z.float() / (2.0 ** 0.5)))
    if tname == "SiLU":
        return torch.sigmoid(z.float())
    if tname == "ReLU":
        return (z > 0).float()
    return None  # unknown / not a gated activation


def _is_activation(module: nn.Module) -> bool:
    return type(module).__name__ in ("NELU", "NiLU", "GELU", "SiLU", "ReLU")


# ── Per-layer accumulators ─────────────────────────────────────────────

class _LayerStats:
    """Streaming accumulator for one activation site.

    Holds per-class feature sums (for V^ℓ), per-channel variance running
    moments, and gate-saturation counts. Features are global-average-
    pooled over the spatial axes so V^ℓ / eff-rank are computed on a
    (N, C) matrix — the standard layer-feature representation Hui et al.
    use (flatten/pool conv maps to a vector).
    """

    def __init__(self, name: str):
        self.name = name
        self.feats = []          # list of (B, C) pooled features
        self.labels = []         # list of (B,) int labels
        self.gate_total = 0
        self.gate_dead = 0       # gate < eps
        self.gate_pass = 0       # gate > 1 - eps
        # per-channel variance via Welford on channel means per sample
        self.ch_sum = None
        self.ch_sqsum = None
        self.ch_n = 0

    def update(self, z: torch.Tensor, gate: torch.Tensor,
               labels: torch.Tensor, eps: float):
        # Pool over spatial axes for conv tensors -> (B, C).
        if z.ndim == 4:
            pooled = z.float().mean(dim=(2, 3))
        elif z.ndim == 3:           # (B, T, C)
            pooled = z.float().mean(dim=1)
        else:                       # (B, C)
            pooled = z.float()
        self.feats.append(pooled.cpu())
        self.labels.append(labels.cpu())

        # per-channel variance of the pooled activation
        if self.ch_sum is None:
            C = pooled.size(1)
            self.ch_sum = torch.zeros(C)
            self.ch_sqsum = torch.zeros(C)
        self.ch_sum += pooled.sum(dim=0).cpu()
        self.ch_sqsum += (pooled ** 2).sum(dim=0).cpu()
        self.ch_n += pooled.size(0)

        if gate is not None:
            g = gate.float()
            self.gate_total += g.numel()
            self.gate_dead += int((g < eps).sum())
            self.gate_pass += int((g > 1.0 - eps).sum())

    def finalize(self) -> dict:
        F_mat = torch.cat(self.feats, dim=0)          # (N, C)
        y = torch.cat(self.labels, dim=0).view(-1)
        v = _collapse_ratio(F_mat, y)
        er = effective_rank(F_mat)
        pr = participation_ratio(F_mat)
        ch_mean = self.ch_sum / max(self.ch_n, 1)
        ch_var = self.ch_sqsum / max(self.ch_n, 1) - ch_mean ** 2
        out = {
            "layer": self.name,
            "n": int(F_mat.size(0)),
            "dim": int(F_mat.size(1)),
            "collapse_V": v,                       # Hui et al. W/B ratio
            "effective_rank": er,
            "participation_ratio": pr,
            "channel_var_mean": float(ch_var.clamp(min=0).mean()),
        }
        if self.gate_total > 0:
            out["gate_dead_frac"] = self.gate_dead / self.gate_total
            out["gate_pass_frac"] = self.gate_pass / self.gate_total
            out["gate_saturated_frac"] = (
                self.gate_dead + self.gate_pass
            ) / self.gate_total
        return out


def _collapse_ratio(F_mat: torch.Tensor, labels: torch.Tensor) -> float:
    """Hui, Belkin & Nakkiran (2022) §3.1 metric, per layer:

        V = E_x ||h(x) - μ_y||²   /   E_i ||μ_i - μ||²
          = (within-class scatter) / (between-class scatter)

    Lower V ⇒ more collapse. Returns +inf-safe float."""
    classes = labels.unique()
    mus = []
    within = 0.0
    total = 0
    for c in classes:
        m = labels == c
        n_c = int(m.sum())
        if n_c < 1:
            continue
        sub = F_mat[m]
        mu_c = sub.mean(dim=0, keepdim=True)
        within += (sub - mu_c).pow(2).sum().item()
        total += n_c
        mus.append(mu_c.squeeze(0))
    if not mus or total == 0:
        return float("nan")
    M = torch.stack(mus, dim=0)                       # (K, C)
    mu = M.mean(dim=0, keepdim=True)
    between = (M - mu).pow(2).sum().item() / max(M.size(0), 1)
    within = within / total
    if between <= 0:
        return float("inf")
    return within / between


# ── Driver ─────────────────────────────────────────────────────────────

@torch.no_grad()
def diagnose(model: nn.Module, loader, device: str, eps: float = 1e-3,
             max_batches: int | None = None) -> list[dict]:
    """Run the model over ``loader`` collecting per-activation-site stats."""
    model.eval()
    sites = [(n, m) for n, m in model.named_modules() if _is_activation(m)]
    stats = {n: _LayerStats(n) for n, _ in sites}
    handles = []

    def mk_hook(name, module):
        def hook(_mod, inp, _out):
            z = inp[0]
            gate = _gate_of(module, z)
            stats[name].update(z, gate, mk_hook._labels, eps)
        return hook

    for n, m in sites:
        handles.append(m.register_forward_hook(mk_hook(n, m)))

    for bi, (x, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        y = yb
        if y.ndim > 1 and y.size(-1) == 1:
            y = y.squeeze(-1)
        # multi-label (chest) has no single class label -> use argmax of
        # the label vector only for the V^ℓ grouping (diagnostic only).
        if y.ndim > 1:
            y = y.float().argmax(dim=-1)
        mk_hook._labels = y.long()
        model(x)

    for h in handles:
        h.remove()
    return [stats[n].finalize() for n, _ in sites]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-layer mechanism diagnostics (M1/M2) on MedMNIST"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--activation", required=True,
                   choices=["relu", "gelu", "silu", "nelu", "nilu"])
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint.pt from train.medmnist (uses best_state)")
    p.add_argument("--data_dir", default="/tmp/medmnist_data")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="gate saturation threshold")
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flag = args.dataset
    num_classes = len(INFO[flag]["label"])

    model = build_model(args.model, args.activation, num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("best_state") or ckpt["model"]
    model.load_state_dict(state)
    model = model.to(device)

    train_loader, _, test_loader = get_dataloaders(
        flag, args.data_dir, train_batch_size=128, num_workers=2
    )

    train_stats = diagnose(model, train_loader, device, args.eps,
                           args.max_batches)
    test_stats = diagnose(model, test_loader, device, args.eps,
                          args.max_batches)

    # M1 = per-layer train/test divergence of the collapse metric.
    by_name = {s["layer"]: s for s in test_stats}
    for s in train_stats:
        t = by_name.get(s["layer"])
        if t and s["collapse_V"] not in (float("inf"),) and \
                t["collapse_V"] not in (float("inf"),):
            s["collapse_V_train_test_gap"] = abs(
                s["collapse_V"] - t["collapse_V"]
            )

    result = {
        "dataset": flag,
        "train_size": INFO[flag]["n_samples"]["train"],
        "task": _task_of(flag),
        "model": args.model,
        "activation": args.activation,
        "eps": args.eps,
        "train": train_stats,
        "test": test_stats,
    }
    out = args.out or f"mechanism_{flag}_{args.model}_{args.activation}.json"
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[mechanism] {flag} {args.model} {args.activation} -> {out}")
    # Quick console summary: deepest layer's V and gate saturation.
    last = test_stats[-1]
    print(f"  deepest layer {last['layer']}: "
          f"V={last['collapse_V']:.4f} "
          f"effrank={last['effective_rank']:.2f} "
          f"gate_sat={last.get('gate_saturated_frac', float('nan')):.3f}")


if __name__ == "__main__":
    main()
