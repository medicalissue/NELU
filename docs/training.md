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
* ``--rms-mode per_sample|per_token`` — override the auto-detected
  reduction axes.

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
