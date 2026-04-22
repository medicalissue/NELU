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

### Recipe fidelity

The ConvNeXt / DeiT / Swin YAMLs are byte-for-byte translations of the
MMPretrain reproduced configs (after resolving their ``_base_``
inheritance). Every recipe parameter matches the upstream value:
AdamW betas and epsilon, weight decay, learning rate at the declared
batch size, cosine schedule with 20-epoch linear warmup, RandAugment
policies (``timm_increasing``, magnitude 9, std 0.5), RandomErasing
probability 0.25 in ``rand`` mode, Mixup α=0.8 / CutMix α=1.0 with
switch probability 0.5, label smoothing 0.1, drop_path per model,
gradient clipping (Swin only, max-norm 5.0), and EMA where the upstream
config enables it (ConvNeXt decay 0.9999, DeiT-Base decay 0.99996).
The MMPretrain pipeline contains no ColorJitter so neither do we.

timm's ``create_optimizer_v2`` already exposes ``filter_bias_and_bn=True``
as its default, which groups all 1-D parameters (biases, LayerNorm
weights) and the model's own ``no_weight_decay()`` set into a wd-free
parameter group. This matches MMPretrain's
``norm_decay_mult=bias_decay_mult=flat_decay_mult=0`` plus the
``custom_keys`` for ``cls_token``, ``pos_embed``,
``absolute_pos_embed``, and ``relative_position_bias_table``.

The EfficientNet-B0 and EfficientNet-B2 recipes are taken directly from
timm's ``hfdocs/source/training_script.mdx``; they produce the
``efficientnet_b0.ra_in1k`` and ``efficientnet_b2.ra_in1k`` checkpoints
published on HuggingFace.

The only intentional deviation from MMPretrain is the random seed. The
upstream default is ``randomness=dict(seed=None)`` — a fresh random
integer per run. We fix ``seed=42`` in every config so the baseline and
Gate-Normalization arms of our experiment matrix are directly
comparable. ``tests/test_configs.py`` guards every parameter mentioned
above against regressions.

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
