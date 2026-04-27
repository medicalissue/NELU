"""Collect γ, β, and effective rank per layer from a trained backbone.

Used to verify the β-adaptive NELU hypothesis: after classification
pretrain, β should be near 0 (NELU-like selective gating) and
effective rank should be compressed; after AE finetune, β should
move positive (wide gating) and effective rank should recover.

Usage:
  python -m eval.cifar.beta_trajectory \\
      --model vgg16_bn --activation nelu_ln \\
      --checkpoint runs/.../checkpoint.pt --output trajectory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gate_norm import NELU, NiLU, NELU_LN, NiLU_LN
from train.cifar import build_model, CIFAR100_MEAN, CIFAR100_STD


def _load_state(model: nn.Module, path: str) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state_dict", ck))
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v
             for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if len(real_missing) > 4:
        raise RuntimeError(f"{len(real_missing)} missing keys: {real_missing[:4]}")
    return int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1


def effective_rank(feat: torch.Tensor) -> float:
    """exp(entropy of normalized singular values) of feat (N, D)."""
    feat = feat - feat.mean(0, keepdim=True)
    # SVD of covariance; cheaper than full SVD for large N.
    cov = feat.t() @ feat / max(feat.size(0) - 1, 1)
    s = torch.linalg.eigvalsh(cov).clamp_min(0)
    s_sum = s.sum()
    if s_sum < 1e-12:
        return 0.0
    p = s / s_sum
    p = p[p > 0]
    H = -(p * p.log()).sum()
    return float(H.exp().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--activation", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--output", required=True)
    p.add_argument("--n-samples", type=int, default=2000,
                   help="how many test images to use for eff-rank computation")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, activation=args.activation, num_classes=100)
    epoch = _load_state(model, args.checkpoint)
    model = model.to(device).eval()

    # Per-layer γ, β
    gn_layers = []
    gn_modules: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (NELU, NiLU, NELU_LN, NiLU_LN)):
            entry = {
                "name": name,
                "type": type(mod).__name__,
                "gamma": float(mod.gamma.detach().cpu().item()),
            }
            if hasattr(mod, "beta"):
                entry["beta"] = float(mod.beta.detach().cpu().item())
            gn_layers.append(entry)
            gn_modules[name] = mod
    print(f"[traj] {len(gn_layers)} gate-norm layers")

    # Build small CIFAR-100 test loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    ds = datasets.CIFAR100(args.data_root, train=False, download=False, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Hook each gate-norm layer to capture its post-activation features
    captured: dict[str, list[torch.Tensor]] = {n: [] for n in gn_modules}

    def make_hook(name: str):
        def _hook(_m, _i, output):
            o = output.detach()
            # Collapse spatial dims: (B, C, H, W) → (B, C) by mean over H,W
            if o.ndim == 4:
                o = o.mean(dim=(2, 3))
            elif o.ndim == 3:
                o = o.mean(dim=1)
            captured[name].append(o.cpu())
        return _hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in gn_modules.items()]

    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            model(x)
            seen += x.size(0)
            if seen >= args.n_samples:
                break
    for h in handles:
        h.remove()

    # Compute effective rank per layer
    for entry in gn_layers:
        feats = torch.cat(captured[entry["name"]], dim=0)[: args.n_samples]
        entry["eff_rank"] = effective_rank(feats.float())
        entry["feat_dim"] = int(feats.size(1))

    # Summary statistics
    summary = {
        "model": args.model,
        "activation": args.activation,
        "checkpoint": args.checkpoint,
        "ckpt_epoch": epoch,
        "n_layers": len(gn_layers),
        "n_samples_used": min(seen, args.n_samples),
        "gamma_mean": (sum(l["gamma"] for l in gn_layers) / len(gn_layers)) if gn_layers else 0.0,
    }
    if gn_layers and "beta" in gn_layers[0]:
        bs = [l["beta"] for l in gn_layers]
        summary.update({
            "beta_mean": sum(bs) / len(bs),
            "beta_min": min(bs),
            "beta_max": max(bs),
        })
    if gn_layers:
        ers = [l["eff_rank"] for l in gn_layers]
        summary.update({
            "eff_rank_mean": sum(ers) / len(ers),
            "eff_rank_min": min(ers),
            "eff_rank_max": max(ers),
        })

    out = {"summary": summary, "layers": gn_layers}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[traj] {summary} → {args.output}")


if __name__ == "__main__":
    main()
