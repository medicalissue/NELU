# Reproducing the paper

The full ImageNet-1k experiment matrix covers eight models trained with a
baseline activation (GELU or SiLU) and with the Gate Normalization
counterpart (NELU or NiLU). Each configuration is run once.

| Model                | Baseline | Variant | Config                                      | Source        |
|----------------------|----------|---------|---------------------------------------------|---------------|
| ConvNeXt-Tiny        | GELU     | NELU    | `configs/imagenet/convnext_tiny.yaml`       | MMPretrain    |
| ConvNeXt-Small       | GELU     | NELU    | `configs/imagenet/convnext_small.yaml`      | MMPretrain    |
| DeiT-Small/16        | GELU     | NELU    | `configs/imagenet/deit_small.yaml`          | MMPretrain    |
| DeiT-Base/16         | GELU     | NELU    | `configs/imagenet/deit_base.yaml`           | MMPretrain    |
| Swin-Tiny            | GELU     | NELU    | `configs/imagenet/swin_tiny.yaml`           | MMPretrain    |
| Swin-Small           | GELU     | NELU    | `configs/imagenet/swin_small.yaml`          | MMPretrain    |
| EfficientNet-B0      | SiLU     | NiLU    | `configs/imagenet/efficientnet_b0.yaml`     | timm          |
| EfficientNet-B2      | SiLU     | NiLU    | `configs/imagenet/efficientnet_b2.yaml`     | timm          |

"Source" refers to where the *baseline* recipe was ported from. Our
paper verifies that the baseline reproductions match the numbers reported
by the cited source before swapping the activation and re-training.

## Hardware

Every recipe is intended for 8× H100 80 GB. On smaller GPUs, reduce
``batch_size`` and set ``grad_accum_steps`` so the effective batch
matches the config's intent (the header comment on each config records
this value).

## Command matrix

For each row `C` of the table and each activation `A ∈ {baseline, NELU/NiLU}`:

```bash
torchrun --nproc_per_node=8 -m train.imagenet \
    --config ${C} \
    --activation ${A} \
    --data-dir /data/imagenet \
    --output ./runs/$(basename ${C%.yaml})-${A} \
    --log-wandb
```

CIFAR-100 (``configs/cifar100.yaml``) is a separate ablation and is
documented alongside the γ-initialization sweep in
``configs/ablation/gamma_init.yaml``.

## Expected deltas

See the paper's Table 1 for the complete set of deltas. Reproductions
within ±0.1 Top-1 of the reported numbers are within the seed-to-seed
noise of the recipes.
