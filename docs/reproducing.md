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

## Robustness

Robustness numbers in the paper are produced by ``eval/imagenet_robustness.py``
(ImageNet-C / A / R / O) and ``eval/cifar_robustness.py`` (CIFAR-100-C).
Download the evaluation datasets with

```bash
bash scripts/prepare_data.sh /data
```

which populates ``/data`` with the Hendrycks public tarballs
(`CIFAR-100-C`, `ImageNet-C`, `imagenet-a`, `imagenet-r`, `imagenet-o`)
and `cifar-100-python`. ImageNet-1k train/val itself must be obtained
from image-net.org separately.

Each checkpoint is then evaluated with::

    python -m eval.imagenet_robustness \
        --model ${MODEL} --activation ${ACT} \
        --checkpoint ${CKPT} --data-root /data \
        --output results/robustness/${MODEL}-${ACT}.json

    python -m eval.cifar_robustness \
        --model ${CIFAR_MODEL} --activation ${ACT} \
        --checkpoint ${CIFAR_CKPT} --data-root /data \
        --output results/robustness/cifar-${CIFAR_MODEL}-${ACT}.json

The JSON outputs are the source of truth for the robustness tables in
the paper.

## Expected deltas

See the paper's Table 1 for the complete set of deltas. Reproductions
within ±0.1 Top-1 of the reported numbers are within the seed-to-seed
noise of the recipes.
