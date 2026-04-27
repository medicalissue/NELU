"""CIFAR-100 autoencoder finetune from a pretrained classification backbone.

Drops the classifier head, freezes (or unfreezes) the conv stack, and
attaches a small upsampling decoder that reconstructs the input image
under MSE. This is the dense-prediction probe for the β-adaptive NELU
hypothesis: when the head changes from classification (which rewards
class-level collapse) to pixel-level reconstruction (which needs
per-spatial-location information), a learnable β in NELU_LN should
*open up* — recovering effective rank that classification compressed.

Resume / lease compatibility
----------------------------
The orchestrator (``orchestrate_cifar_slot.sh``) drives this trainer
exactly like ``train.cifar``: it syncs the ``output_dir`` from S3,
finds a partial ``checkpoint.pt`` if any, passes ``--resume``, and
expects a ``complete`` sentinel file when the run finishes. We also
checkpoint every epoch and write the sentinel in the same place so a
spot interruption costs at most one epoch of progress.

CLI mirrors ``train.cifar`` for slot-script compatibility:
  --config CFG --activation ACT --seed N --output_dir DIR
  --resume CKPT (optional) --wandb --wandb_project PROJ
The classification pretrain ckpt is read from ``<CKPT_BUCKET>/<base>-<act>-s<seed>/checkpoint.pt``
and pulled into ``output_dir`` before this script is invoked, so we
detect it automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gate_norm import NELU, NiLU, NELU_LN, NiLU_LN, NELU_AFF, NiLU_AFF
from train.cifar import (
    CIFAR100_MEAN, CIFAR100_STD, build_model,
    _load_config_with_includes,  # reuse cls trainer's yaml loader
)


# ── Encoder feature extractor ────────────────────────────────────────────


def _vgg16_encoder(model: nn.Module) -> tuple[nn.Module, int]:
    feats = list(model.features.children())
    last_pool = max(i for i, m in enumerate(feats) if isinstance(m, nn.MaxPool2d))
    encoder = nn.Sequential(*feats[:last_pool])
    return encoder, 512


def _resnet56_encoder(model: nn.Module) -> tuple[nn.Module, int]:
    children = dict(model.named_children())
    layers = []
    for name in ("conv1", "bn1", "relu", "layer1", "layer2", "layer3"):
        if name in children:
            layers.append(children[name])
    encoder = nn.Sequential(*layers)
    return encoder, 64


_ENCODERS = {
    "vgg16_bn": _vgg16_encoder,
    "resnet56": _resnet56_encoder,
}


def make_encoder(model: nn.Module, name: str) -> tuple[nn.Module, int]:
    if name not in _ENCODERS:
        raise ValueError(f"No encoder rule for {name!r}; supported: {list(_ENCODERS)}")
    return _ENCODERS[name](model)


# ── Decoder ──────────────────────────────────────────────────────────────


class Decoder(nn.Module):
    """Symmetric upsampling decoder back to 32×32×3 from a (C, H, W) latent."""

    def __init__(self, in_ch: int, target_size: int = 32, latent_size: int = 8):
        super().__init__()
        n_up = 0
        s = latent_size
        while s < target_size:
            s *= 2
            n_up += 1
        ch = in_ch
        layers: list[nn.Module] = []
        for _ in range(n_up):
            out_ch = max(ch // 2, 16)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Data ─────────────────────────────────────────────────────────────────


def get_loaders(data_dir: str, batch_size: int, workers: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train = datasets.CIFAR100(data_dir, train=True, download=False, transform=transform)
    val   = datasets.CIFAR100(data_dir, train=False, download=False, transform=transform)
    train_l = DataLoader(train, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=True, drop_last=True)
    val_l   = DataLoader(val, batch_size=batch_size, shuffle=False,
                         num_workers=workers, pin_memory=True)
    return train_l, val_l


def collect_beta_stats(model: nn.Module) -> list[dict]:
    out: list[dict] = []
    for name, mod in model.named_modules():
        if isinstance(mod, (NELU_LN, NiLU_LN, NELU_AFF, NiLU_AFF)):
            out.append({
                "name": name,
                "gamma": float(mod.gamma.detach().cpu().item()),
                "beta": float(mod.beta.detach().cpu().item()),
            })
    return out


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-100 AE finetune (resume-aware)")
    # Slot-script-compatible flags
    p.add_argument("--config", required=True, help="cls model yaml (e.g. configs/cifar100/vgg16_bn.yaml)")
    p.add_argument("--activation", required=True,
                   choices=["relu", "gelu", "silu",
                            "nelu", "nilu", "nelu_ln", "nilu_ln",
                            "nelu_aff", "nilu_aff"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default=None,
                   help="if set, resume AE training from this ckpt")
    p.add_argument("--data_root", default="/data")
    p.add_argument("--cls_ckpt", default=None,
                   help="path to classification pretrain ckpt; if not set, "
                        "looked up at <output_dir>/../<base>-<act>-s<seed>/checkpoint.pt")
    p.add_argument("--ae_mode", choices=["full", "beta_only"], default="full")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="beta-adaptive-nelu")
    args = p.parse_args()

    # Resolve model name from cfg yaml.
    cfg = _load_config_with_includes(args.config)
    model_name = cfg.get("model")
    if model_name is None:
        raise ValueError(f"--config {args.config} has no 'model' field")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Locate the classification pretrain ckpt.
    if args.cls_ckpt:
        cls_ckpt = args.cls_ckpt
    else:
        # Convention: AE output dir lives at /tmp/runs/<exp>-ae, the cls
        # output dir is at /tmp/runs/<exp>. We look one level up.
        ae_dir = Path(args.output_dir)
        # exp = <model>-<act>-s<seed>; cls dir is sibling without "-ae" suffix
        cls_dir_name = ae_dir.name.removesuffix("-ae")
        cls_ckpt = str(ae_dir.parent / cls_dir_name / "checkpoint.pt")
    if not os.path.isfile(cls_ckpt):
        raise FileNotFoundError(
            f"classification ckpt not found at {cls_ckpt} — AE finetune cannot start"
        )

    # ── Build model + load pretrain weights ────────────────────────
    model = build_model(model_name, activation=args.activation, num_classes=100)
    ck = torch.load(cls_ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("state_dict", ck))
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v
             for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if len(real_missing) > 4:
        raise RuntimeError(f"too many missing keys: {real_missing[:4]}")
    print(f"[ae] cls ckpt loaded from {cls_ckpt} (epoch={ck.get('epoch','?')})")

    pre_stats = collect_beta_stats(model)
    if pre_stats:
        bs = [l["beta"] for l in pre_stats]
        print(f"[ae] pre-finetune β: n={len(bs)}, "
              f"mean={sum(bs)/len(bs):+.4f}, "
              f"min={min(bs):+.4f}, max={max(bs):+.4f}")

    encoder, latent_ch = make_encoder(model, model_name)
    encoder.to(device).eval()
    with torch.no_grad():
        d = torch.zeros(1, 3, 32, 32, device=device)
        latent = encoder(d)
    latent_size = latent.shape[-1]
    print(f"[ae] latent: {tuple(latent.shape)} (latent_ch={latent_ch})")

    decoder = Decoder(latent_ch, target_size=32, latent_size=latent_size).to(device)

    # ── Trainable param selection ──────────────────────────────────
    if args.ae_mode == "beta_only":
        for p_ in encoder.parameters():
            p_.requires_grad_(False)
        beta_params = []
        for mod in encoder.modules():
            if isinstance(mod, (NELU_LN, NiLU_LN, NELU_AFF, NiLU_AFF)):
                mod.beta.requires_grad_(True)
                beta_params.append(mod.beta)
        param_groups = [
            {"params": beta_params, "lr": args.lr},
            {"params": decoder.parameters(), "lr": args.lr},
        ]
        print(f"[ae] mode=beta_only β params={len(beta_params)} dec={sum(p.numel() for p in decoder.parameters())}")
    else:
        encoder.train()
        param_groups = [
            {"params": encoder.parameters(), "lr": args.lr_backbone},
            {"params": decoder.parameters(), "lr": args.lr},
        ]
        print(f"[ae] mode=full unfrozen ({sum(p.numel() for p in encoder.parameters())} encoder + "
              f"{sum(p.numel() for p in decoder.parameters())} decoder params)")

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume from AE-side checkpoint if present ─────────────────
    start_ep = 0
    history: list[dict] = []
    best_val = float("inf")
    wandb_id = None
    if args.resume and os.path.isfile(args.resume):
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        encoder.load_state_dict(rk["encoder"])
        decoder.load_state_dict(rk["decoder"])
        optimizer.load_state_dict(rk["optimizer"])
        scheduler.load_state_dict(rk["scheduler"])
        start_ep = int(rk["epoch"]) + 1
        history = rk.get("history", [])
        best_val = float(rk.get("best_val_mse", float("inf")))
        wandb_id = rk.get("wandb_id")
        print(f"[ae] resumed from {args.resume} at epoch={start_ep} best={best_val:.5f}")

    # ── W&B ────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            run_name = f"{model_name}_{args.activation}_s{args.seed}_ae_{args.ae_mode}"
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name,
                id=wandb_id, resume="allow" if wandb_id else None,
                config={
                    "stage": "ae",
                    "model": model_name, "activation": args.activation,
                    "seed": args.seed, "ae_mode": args.ae_mode,
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "lr_backbone": args.lr_backbone,
                },
            )
            wandb_id = wandb_run.id
        except Exception as e:
            print(f"[ae] wandb setup failed: {e}")

    train_l, val_l = get_loaders(args.data_root, args.batch_size, args.workers)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        encoder.train(args.ae_mode == "full"); decoder.train()
        train_mse = 0.0; n = 0
        for x, _ in train_l:
            x = x.to(device, non_blocking=True)
            recon = decoder(encoder(x))
            loss = ((recon - x) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_mse += loss.item() * x.size(0); n += x.size(0)
        train_mse /= n
        scheduler.step()

        encoder.eval(); decoder.eval()
        val_mse = 0.0; n = 0
        with torch.no_grad():
            for x, _ in val_l:
                x = x.to(device, non_blocking=True)
                val_mse += ((recon := decoder(encoder(x.to(device)))) - x).pow(2).mean().item() * x.size(0)
                n += x.size(0)
        val_mse /= n

        beta_now = collect_beta_stats(encoder)
        beta_mean = (sum(l["beta"] for l in beta_now) / len(beta_now)) if beta_now else 0.0
        gamma_mean = (sum(l["gamma"] for l in beta_now) / len(beta_now)) if beta_now else 0.0

        if val_mse < best_val:
            best_val = val_mse
        history.append({
            "epoch": ep, "train_mse": train_mse, "val_mse": val_mse,
            "best_val_mse": best_val,
            "beta_mean": beta_mean, "gamma_mean": gamma_mean,
        })

        if wandb_run is not None:
            metrics = {
                "epoch": ep,
                "ae/train_mse": train_mse, "ae/val_mse": val_mse,
                "ae/best_val_mse": best_val,
                "ae/beta_mean": beta_mean, "ae/gamma_mean": gamma_mean,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for i, l in enumerate(beta_now):
                metrics[f"ae/beta/layer_{i}"] = l["beta"]
                metrics[f"ae/gamma/layer_{i}"] = l["gamma"]
            wandb_run.log(metrics)

        print(f"  ep {ep:3d}  train={train_mse:.5f}  val={val_mse:.5f}  best={best_val:.5f}  "
              f"β̄={beta_mean:+.4f}  γ̄={gamma_mean:+.4f}")

        # Checkpoint EVERY epoch so a spot preempt costs at most one epoch.
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": ep,
            "best_val_mse": best_val,
            "history": history,
            "wandb_id": wandb_id,
            "args": vars(args),
        }
        # Use the same filename the slot script syncs: "checkpoint.pt"
        torch.save(ckpt, Path(args.output_dir, "checkpoint.pt"))

    # Final result + complete sentinel (matches what the slot script expects)
    final = {
        "stage": "ae",
        "model": model_name,
        "activation": args.activation,
        "seed": args.seed,
        "ae_mode": args.ae_mode,
        "epochs": args.epochs,
        "best_val_mse": best_val,
        "history": history,
        "pre_finetune_beta": pre_stats,
        "post_finetune_beta": collect_beta_stats(encoder),
        "seconds": time.time() - t0,
    }
    Path(args.output_dir, "ae_result.json").write_text(json.dumps(final, indent=2))
    Path(args.output_dir, "complete").write_text("ae done\n")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[ae] done. best_val_mse={best_val:.5f} → {args.output_dir}/")


if __name__ == "__main__":
    main()
