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
  ``--torchcompile-mode {default,reduce-overhead,max-autotune,…}`` —
  wrap the training task in ``torch.compile``. Gate-Normalization's
  forward pass is plain PyTorch, so Inductor traces and fuses it
  end-to-end without help.
* ``--amp`` + ``--amp-dtype {float16,bfloat16}`` — bfloat16 is the
  preferred choice on Hopper (no GradScaler needed, wider dynamic range).
  Leave at ``float16`` for bitwise fidelity to MMPretrain/timm baselines.

``torchcompile``/``torchcompile_mode`` are also exposed in every YAML
(null by default) so ``--config`` alone is enough to turn compile on
for a production sweep.

## Fused CUDA kernel

NELU / NiLU forward and backward have a fused CUDA implementation under
``gate_norm/csrc/``. It is built on first call via
``torch.utils.cpp_extension.load`` and cached under
``~/.cache/torch_extensions/`` (re-import is essentially free
afterwards). The kernel implements the RMS-only form

    rsigma = 1 / sqrt(mean(z²) + eps)
    y      = z · g(γ · z · rsigma)

with three dispatch tiers selected at launch time:

* **Vectorized register-cached** for medium ``N`` (≤ ~4 K bf16 / fp16,
  ≤ ~2 K fp32 backward): row stays in per-thread registers across both
  reduction and emit passes, smem holds only warp-staging scratch.
* **Smem-cached** for larger rows that still fit in dynamic shared
  memory (Hopper opt-in cap ~228 KB, Ampere ~164 KB).
* **Streaming two-pass** for ``N`` beyond the smem cap: stream ``z``
  twice, share ``rsigma`` / ``S`` via global scratch.

The dgamma reduction uses one ``atomicAdd`` per block into a single
fp32 scalar — contention is ``O(M)`` and never the bottleneck for
typical FFN row counts.

The pure-PyTorch path remains the reference; the kernel is taken only
when the input is on CUDA and dtype is fp32 / fp16 / bf16. To bisect
numerical mismatches between paths, set
``GATE_NORM_FORCE_PYTHON=1`` to force the PyTorch path everywhere
without rebuilding.

Build troubleshooting:

* The kernel needs the CUDA SDK (``nvcc``) on the system PATH and
  ``ninja`` (``pip install ninja``). Both are present in the standard
  PyTorch ``nvidia/cuda:*-devel`` images.
* Build artefacts live under
  ``~/.cache/torch_extensions/<py_ver>/gate_norm_fused/``. Delete this
  directory to force a rebuild after CUDA SDK changes.
* If you see ``cudaErrorInvalidValue`` on Hopper / Ampere with very
  long rows, the dynamic-smem opt-in is failing — check
  ``cudaDeviceProp.sharedMemPerBlockOptin`` returns the expected
  ~228 KB / ~164 KB.

## EMA

All ImageNet runs use a single unified EMA recipe (``decay = 0.9999``).
This is a small departure from the per-arch reference recipes — DeiT-S
and Swin-T/S originally ship without EMA, and DeiT-B uses
``decay = 0.99996`` — but a uniform decay makes ``eval_top1`` directly
comparable across the whole sweep without per-arch footnotes.

| Model                 | EMA | Decay  | Reference                              |
|-----------------------|-----|--------|----------------------------------------|
| DeiT-Small / Base     | on  | 0.9999 | MMPretrain drops EMA (S) / 0.99996 (B) |
| Swin-Tiny / Small     | on  | 0.9999 | MMPretrain swin recipe omits EMAHook   |
| ConvNeXt-Tiny / Small | on  | 0.9999 | MMPretrain ``momentum=1e-4``           |
| EfficientNet-B0 / B2  | on  | 0.9999 | timm ``--model-ema-decay 0.9999``      |

``eval_top1`` / ``eval_top5`` in W&B are the **EMA** metrics (and
best-checkpoint tracking uses them). The non-EMA branch is logged in
parallel as ``raw_eval_top1`` / ``raw_eval_top5`` for debugging and
ablation — keep these around to diagnose EMA-vs-raw divergence.

For the paper, report EMA ``eval_top1`` in the main table and list
``raw_eval_top1`` in the appendix.

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
