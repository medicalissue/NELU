# Training

Two training paths are supported: a **local** path for single-node
workstations and a **SkyPilot** path for managed spot-instance runs with
auto-recovery from preemption.

## Local

Install the training extras and ensure your dataset is on disk:

```bash
pip install -e '.[train]'
```

ImageNet-1k should be laid out in the ``torchvision.ImageFolder`` style
(one directory per class). A single run uses ``torchrun``:

```bash
torchrun --nproc_per_node=8 -m train.imagenet \
    --config configs/imagenet/convnext_tiny.yaml \
    --activation nelu \
    --data-dir /data/imagenet \
    --output ./runs/convnext_tiny-nelu
```

For CIFAR-100, use the lighter-weight ``train.cifar`` entrypoint:

```bash
python -m train.cifar \
    --config configs/cifar100.yaml \
    --activation nelu \
    --data-dir /data
```

Useful flags:

* ``--log-wandb`` — stream metrics to Weights & Biases. Requires
  ``WANDB_API_KEY`` in the environment (or ``.env``).
* ``--resume PATH`` — resume from a checkpoint. Paired with the W&B
  run-id sidecar, this keeps all logs on one run.
* ``--norm-axes sample|channel|[<ints>]`` — override the auto-detected
  reduction axes. Accepts the ``"channel"`` / ``"sample"`` aliases or an
  explicit axis list such as ``[-1]`` or ``[2, 3]``.
* ``--torchcompile [backend]`` (alias: ``inductor`` when no arg), plus
  ``--torchcompile-mode {default,reduce-overhead,max-autotune,…}`` — wrap
  the training task in ``torch.compile``. Gate-Normalization's fused
  CUDA kernels are registered via ``torch.library.custom_op`` with
  ``register_fake`` tensors, so Dynamo traces through them without graph
  breaks. The baseline (GELU/SiLU) arm compiles unconditionally; the
  Gate-Normalization arm falls through to the Python path on the first
  iteration and the CUDA op on the rest.
* ``--amp`` + ``--amp-dtype {float16,bfloat16}`` — bfloat16 is the
  preferred choice on Hopper (no GradScaler needed, wider dynamic range).
  Leave at ``float16`` for bitwise fidelity to MMPretrain/timm baselines.

``torchcompile``/``torchcompile_mode`` are also exposed in every YAML
(null by default) so ``--config`` alone is enough to turn compile on
for a production sweep.

## EMA per architecture

We follow each architecture's reference recipe verbatim for the EMA
hyperparameter (this is uniform across activation variants, so the
controlled comparison is unaffected):

| Model | EMA | Decay (timm) | Reference momentum |
|-------|-----|--------------|--------------------|
| DeiT-Small        | off | —        | MMPretrain drops EMA for the small variant |
| DeiT-Base         | on  | 0.99996  | MMPretrain ``momentum=4e-5`` |
| Swin-Tiny / Small | off | —        | MMPretrain swin recipe omits EMAHook |
| ConvNeXt-Tiny / Small | on | 0.9999 | MMPretrain ``momentum=1e-4`` |
| EfficientNet-B0 / B2  | on | 0.9999 | timm training script ``--model-ema-decay 0.9999`` |

When EMA is on, ``eval_top1`` / ``eval_top5`` in W&B are the **EMA**
metrics (and best-checkpoint tracking uses them). The non-EMA branch
is logged in parallel as ``raw_eval_top1`` / ``raw_eval_top5`` for
debugging and ablation. When EMA is off, ``eval_*`` is the only branch
and ``raw_*`` keys are absent.

For the paper, report EMA ``eval_top1`` whenever EMA is on — that
matches every reported reference number from MMPretrain / timm.

## SkyPilot (managed jobs)

Install SkyPilot once with whatever cloud plugin your credentials target:

```bash
pip install 'skypilot-nightly[aws]'
sky check
```

Launch a single managed job:

```bash
sky jobs launch -n convnext-tiny-nelu sky/train.yaml \
    --env CONFIG=configs/imagenet/convnext_tiny.yaml \
    --env ACTIVATION=nelu
```

Spot preemptions trigger a retry on a fresh VM. Checkpoints are
persisted to the S3 bucket mounted at ``/output``; the trainer detects a
pre-existing ``last.pth.tar`` and resumes from it, and re-uses the same
W&B run id because it is stored alongside the checkpoints. No extra CLI
flag is needed on resume.

A convenience ``scripts/launch_all.sh`` script (not included; write your
own) can loop over every config-activation pair to dispatch the full
experiment matrix.
