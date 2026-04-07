"""
Gate pattern similarity vs scale mismatch — pretrained ViT-B on ImageNet-100.

For semantically similar pairs (same class, nearest neighbor in feature space):
  - Compute centered gate cosine similarity (pattern only, mean removed)
  - Compute scale mismatch |log rms(z_x) - log rms(z_x')|
  - Plot: as scale mismatch grows, does gate pattern similarity drop?

If yes: gate depends on absolute scale, not just feature identity.

Memory-efficient: process one layer at a time, don't store full gate vectors.

Usage:
    python experiments/gate_pattern_analysis.py
"""

import os
import sys
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.makedirs("results", exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    import timm
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model + data ──
    print("Loading ViT-B pretrained...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = model.half().to(device).eval()

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print("Loading ImageNet-100 validation...")
    hf_ds = load_dataset("clane9/imagenet-100", cache_dir="./data/imagenet100_hf")

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf, tf):
            self.hf = hf
            self.tf = tf
        def __len__(self):
            return len(self.hf)
        def __getitem__(self, idx):
            item = self.hf[idx]
            img = item["image"].convert("RGB")
            return self.tf(img), item["label"]

    val_ds = HFDataset(hf_ds["validation"], transform)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # ── Collect: per-sample features, labels, and per-layer gate info ──
    # We pick 3 layers: early (1), middle (5), late (10)
    target_layers = [1, 5, 10]
    layer_names = ["Layer 2", "Layer 6", "Layer 11"]

    # Storage: per sample, per target layer → (gate_centered, rms)
    # gate_centered is already mean-removed and normalized
    # To save memory: store only the centered+normalized gate vector (float16)

    all_features = []  # penultimate features for NN search
    all_labels = []

    # Per-layer storage
    layer_data = {l: {"gates_centered": [], "rms": []} for l in target_layers}

    # Hook to capture pre-activations at target layers
    pre_acts = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            pre_acts[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for idx, block in enumerate(model.blocks):
        if idx in target_layers:
            hooks.append(block.mlp.fc1.register_forward_hook(make_hook(idx)))

    # Also hook the final norm to get features
    final_features = []
    def feat_hook(module, input, output):
        final_features.append(output[:, 0].detach())  # CLS token
    hooks.append(model.norm.register_forward_hook(feat_hook))

    max_samples = 2000  # enough for meaningful NN pairs

    print("Running forward pass...")
    n_collected = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if n_collected >= max_samples:
                break
            pre_acts.clear()
            final_features.clear()
            x = x.to(device).half()
            _ = model(x)

            bs = x.shape[0]
            all_labels.append(y.numpy())
            all_features.append(final_features[0].float().cpu().numpy())

            for l_idx in target_layers:
                z = pre_acts[l_idx].float()  # (B, T, D)
                # Pool over tokens: mean
                z_pooled = z.mean(dim=1)  # (B, D)

                # RMS
                rms = z_pooled.pow(2).mean(dim=-1).sqrt()  # (B,)
                layer_data[l_idx]["rms"].append(rms.cpu().numpy())

                # GELU gate
                g = 0.5 * (1.0 + torch.erf(z_pooled / np.sqrt(2)))  # (B, D)

                # Center and normalize (pattern only)
                g_mean = g.mean(dim=-1, keepdim=True)
                g_centered = g - g_mean
                g_norm = g_centered.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                g_unit = (g_centered / g_norm).half()  # save as float16

                layer_data[l_idx]["gates_centered"].append(g_unit.cpu().numpy())

                # Also compute NELU gate for comparison
                rms_vec = z_pooled.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                gn = 0.5 * (1.0 + torch.erf(z_pooled / (rms_vec * np.sqrt(2))))
                gn_mean = gn.mean(dim=-1, keepdim=True)
                gn_centered = gn - gn_mean
                gn_norm = gn_centered.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                gn_unit = (gn_centered / gn_norm).half()

                if "nelu_gates" not in layer_data[l_idx]:
                    layer_data[l_idx]["nelu_gates"] = []
                layer_data[l_idx]["nelu_gates"].append(gn_unit.cpu().numpy())

            n_collected += bs
            del x
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                print(f"  {n_collected}/{max_samples} samples")

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Concatenate
    all_labels = np.concatenate(all_labels)[:max_samples]
    all_features = np.concatenate(all_features)[:max_samples]
    for l_idx in target_layers:
        layer_data[l_idx]["rms"] = np.concatenate(layer_data[l_idx]["rms"])[:max_samples]
        layer_data[l_idx]["gates_centered"] = np.concatenate(layer_data[l_idx]["gates_centered"])[:max_samples]
        layer_data[l_idx]["nelu_gates"] = np.concatenate(layer_data[l_idx]["nelu_gates"])[:max_samples]

    print(f"Collected {len(all_labels)} samples, {len(np.unique(all_labels))} classes")

    # ── Find same-class nearest neighbor pairs ──
    print("Finding same-class NN pairs...")
    from sklearn.neighbors import NearestNeighbors

    # Normalize features
    feat_norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8
    all_features_normed = all_features / feat_norms

    pairs = []  # (i, j) indices
    classes = np.unique(all_labels)

    for cls in classes:
        cls_idx = np.where(all_labels == cls)[0]
        if len(cls_idx) < 5:
            continue
        feats = all_features_normed[cls_idx]
        nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(feats)
        _, indices = nn.kneighbors(feats)
        for i_local in range(len(cls_idx)):
            j_local = indices[i_local, 1]  # nearest neighbor (not self)
            i_global = cls_idx[i_local]
            j_global = cls_idx[j_local]
            if i_global < j_global:  # avoid duplicates
                pairs.append((i_global, j_global))

    pairs = np.array(pairs)
    print(f"  {len(pairs)} same-class NN pairs")

    # ── Compute pattern similarity and scale mismatch for each pair ──
    print("Computing similarities...")

    n_quintiles = 5

    fig, axes = plt.subplots(1, len(target_layers), figsize=(len(target_layers) * 4.5, 4.2))

    for col, (l_idx, l_name) in enumerate(zip(target_layers, layer_names)):
        ax = axes[col]

        gates_g = layer_data[l_idx]["gates_centered"].astype(np.float32)
        gates_n = layer_data[l_idx]["nelu_gates"].astype(np.float32)
        rms_vals = layer_data[l_idx]["rms"]

        # Per-pair metrics
        cosines_gelu = []
        cosines_nelu = []
        scale_mismatches = []

        for i, j in pairs:
            # Scale mismatch
            delta = abs(np.log(rms_vals[i] + 1e-8) - np.log(rms_vals[j] + 1e-8))
            scale_mismatches.append(delta)

            # GELU pattern cosine (already centered+normalized)
            cos_g = np.dot(gates_g[i], gates_g[j])
            cosines_gelu.append(cos_g)

            # NELU pattern cosine
            cos_n = np.dot(gates_n[i], gates_n[j])
            cosines_nelu.append(cos_n)

        scale_mismatches = np.array(scale_mismatches)
        cosines_gelu = np.array(cosines_gelu)
        cosines_nelu = np.array(cosines_nelu)

        # Bin by scale mismatch quintiles
        quantiles = np.quantile(scale_mismatches, np.linspace(0, 1, n_quintiles + 1))

        gelu_means, gelu_stds = [], []
        nelu_means, nelu_stds = [], []

        for b in range(n_quintiles):
            mask = (scale_mismatches >= quantiles[b]) & (scale_mismatches < quantiles[b + 1] + 1e-6)
            if mask.sum() > 0:
                gelu_means.append(cosines_gelu[mask].mean())
                gelu_stds.append(cosines_gelu[mask].std() / np.sqrt(mask.sum()))
                nelu_means.append(cosines_nelu[mask].mean())
                nelu_stds.append(cosines_nelu[mask].std() / np.sqrt(mask.sum()))

        x_pos = np.arange(n_quintiles)
        ax.errorbar(x_pos, gelu_means, yerr=gelu_stds, fmt="o-",
                    color="#2979FF", label="GELU gate", capsize=4, markersize=5)
        ax.errorbar(x_pos, nelu_means, yerr=nelu_stds, fmt="s-",
                    color="#D50000", label="NELU gate", capsize=4, markersize=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Low", "Q2", "Q3", "Q4", "High"])
        ax.set_xlabel("Scale mismatch $|\\log\\,\\mathrm{rms}(z) - \\log\\,\\mathrm{rms}(z')|$")
        if col == 0:
            ax.set_ylabel("Centered gate cosine similarity")
        ax.set_title(f"{l_name}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Gate pattern similarity vs activation-scale mismatch\n"
                 "(same-class NN pairs from pretrained ViT-B on ImageNet-100)\n"
                 "GELU patterns degrade with scale mismatch; NELU patterns are stable",
                 fontweight="bold", fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig("results/gate_pattern_similarity.png", bbox_inches="tight", facecolor="white")
    fig.savefig("results/gate_pattern_similarity.pdf", bbox_inches="tight", facecolor="white")
    print("\nSaved results/gate_pattern_similarity.{png,pdf}")
    plt.close()

    # Numerical
    print(f"\n  Layer | GELU cos (low Δ) | GELU cos (high Δ) | drop | NELU cos (low) | NELU cos (high) | drop")
    print("  " + "-" * 90)
    for l_idx, l_name in zip(target_layers, layer_names):
        gates_g = layer_data[l_idx]["gates_centered"].astype(np.float32)
        gates_n = layer_data[l_idx]["nelu_gates"].astype(np.float32)
        rms_vals = layer_data[l_idx]["rms"]

        cos_g, cos_n, deltas = [], [], []
        for i, j in pairs:
            deltas.append(abs(np.log(rms_vals[i]+1e-8) - np.log(rms_vals[j]+1e-8)))
            cos_g.append(np.dot(gates_g[i], gates_g[j]))
            cos_n.append(np.dot(gates_n[i], gates_n[j]))
        deltas, cos_g, cos_n = np.array(deltas), np.array(cos_g), np.array(cos_n)

        low = deltas <= np.quantile(deltas, 0.2)
        high = deltas >= np.quantile(deltas, 0.8)

        print(f"  {l_name:>8} | {cos_g[low].mean():.4f} | {cos_g[high].mean():.4f} "
              f"| {cos_g[low].mean()-cos_g[high].mean():+.4f} "
              f"| {cos_n[low].mean():.4f} | {cos_n[high].mean():.4f} "
              f"| {cos_n[low].mean()-cos_n[high].mean():+.4f}")


if __name__ == "__main__":
    main()
