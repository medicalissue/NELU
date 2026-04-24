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
