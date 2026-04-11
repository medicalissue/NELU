#!/usr/bin/env python3
"""COCO Mask R-CNN with ConvNeXt-T backbone — GELU vs NELU.

Used for §4.4 of the paper. Two runs only:

    GELU: backbone = timm ConvNeXt-T pretrained on ImageNet (GELU).
    NELU: backbone = our ConvNeXt-T NELU trained in §4.3 Phase 3.

Both runs share the same Mask R-CNN head, FPN, and 1× schedule
(12 epochs ≈ 90k iters at batch 16). The activation in the
ConvNeXt blocks is the ONLY difference — all other layers
(neck/head/RoI) are unchanged GELU/ReLU as in detectron2 defaults.

Status: SCAFFOLD — wires through arg parsing and backbone loading,
but the detectron2 training loop itself is not implemented yet.
The 1x recipe targets:
    optimizer  : AdamW lr=1e-4 wd=0.05
    schedule   : step at [8, 11], end at 12
    batch      : 16 images (2/GPU × 8 GPUs)
    augment    : ResizeShortestEdge[480..800], RandomFlip
    image size : 800 short edge
    AMP        : on
    EMA        : off

Usage:
    torchrun --nproc_per_node=8 experiments/train_coco_maskrcnn.py \
        --backbone convnext_tiny --act gelu \
        --data /data/coco --schedule 1x --wandb
    torchrun --nproc_per_node=8 experiments/train_coco_maskrcnn.py \
        --backbone convnext_tiny --act nelu \
        --backbone-ckpt results/imagenet/convnext_tiny_nelu/best.pt \
        --data /data/coco --schedule 1x --wandb
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nelu import NELU, NELUCUDA, NiLU, NiLUCUDA  # noqa: F401

_NELU_CLS = NELUCUDA if NELUCUDA is not None else NELU


def _ddp_init():
    if int(os.environ.get("RANK", -1)) != -1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def build_backbone(name, act, ckpt_path):
    """Build the ConvNeXt-T backbone with the requested activation,
    optionally loading our trained weights from §4.3."""
    import timm
    if name != "convnext_tiny":
        raise NotImplementedError(f"backbone {name} not wired")
    if act == "gelu":
        # GELU baseline = timm ImageNet pretrained
        model = timm.create_model("convnext_tiny.fb_in1k", pretrained=True,
                                  features_only=True, out_indices=(0, 1, 2, 3))
    else:
        # NELU = scratch model with weights loaded from our §4.3 ckpt.
        model = timm.create_model("convnext_tiny", pretrained=False,
                                  features_only=True, out_indices=(0, 1, 2, 3))
        # Replace GELU → NELU to match the trained checkpoint topology
        def _swap(parent):
            for child_name, child in parent.named_children():
                if isinstance(child, nn.GELU):
                    setattr(parent, child_name, _NELU_CLS())
                else:
                    _swap(child)
        _swap(model)
        if ckpt_path is None:
            raise ValueError("--backbone-ckpt required when --act nelu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        # features_only models drop the classifier — strip "head.*" keys
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  loaded backbone from {ckpt_path}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnext_tiny",
                   choices=["convnext_tiny"])
    p.add_argument("--act", default="gelu", choices=["gelu", "nelu"])
    p.add_argument("--backbone-ckpt", default=None,
                   help="Pretrained backbone for --act nelu (our §4.3 ckpt)")
    p.add_argument("--data", required=True, help="COCO root")
    p.add_argument("--schedule", default="1x", choices=["1x", "3x"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            f"results/coco/maskrcnn_{args.backbone}_{args.act}")
    os.makedirs(args.output_dir, exist_ok=True)

    distributed, local_rank, rank, world = _ddp_init()
    is_main = rank == 0

    if is_main:
        print(f"Mask R-CNN + {args.backbone} ({args.act})  "
              f"schedule={args.schedule}  world={world}")

    # Backbone — verifies the checkpoint loads cleanly before we
    # spend hours setting up detectron2.
    backbone = build_backbone(args.backbone, args.act, args.backbone_ckpt)
    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    if is_main:
        print(f"  backbone params: {n_params:.1f}M")

    raise NotImplementedError(
        "Detectron2 training loop is not yet implemented. Steps:\n"
        "  1. Wrap `backbone` in detectron2.modeling.Backbone with FPN.\n"
        "     The features_only timm output gives strides 4/8/16/32.\n"
        "  2. Build Mask R-CNN via LazyConfig from\n"
        "     'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.py'\n"
        "     and swap in the ConvNeXt backbone.\n"
        "  3. Recipe (1x): AdamW lr=1e-4 wd=0.05, step at [8,11]/12,\n"
        "     batch 16 (2/GPU × 8 GPU), AMP on, ResizeShortestEdge\n"
        "     [480..800] + RandomFlip, image short=800.\n"
        "  4. Train via detectron2.engine.DefaultTrainer (or LazyTrainer)\n"
        "     reading from --data.\n"
        "  5. Eval with COCOEvaluator → mAP_box, mAP_mask.\n"
        "  6. Write results/coco/maskrcnn_{backbone}_{act}/result.json\n"
        "     with both metrics so the pipeline `skip_if_done` works.\n"
    )


if __name__ == "__main__":
    main()
