# SkyPilot recipes

This directory contains the SkyPilot task specs used to run the Gate
Normalization experiments on commodity cloud GPUs with automatic spot
preemption recovery.

## Files

- `train.yaml` — single ImageNet-1k training run. Runs `torchrun` on one
  8-GPU node. Use `sky jobs launch` to opt into managed (auto-recovering)
  mode; use `sky launch` for a one-off debug run.
- `setup.sh` — invoked once per VM boot to install the package and warm up
  the CUDA extensions.

## Environment variables

`sky launch --env KEY=VAL` sets a task-level environment variable. The
variables recognised by `train.yaml` are:

| Name            | Purpose                                            |
|-----------------|----------------------------------------------------|
| `CONFIG`        | Path to a config under `configs/imagenet/`.        |
| `ACTIVATION`    | `gelu`, `silu`, `nelu`, `nilu`, or `relu`.         |
| `EXP_NAME`      | Output subdirectory name; defaults to model-act.   |
| `DATA_BUCKET`   | S3 bucket mounted read-only at `/data/imagenet`.   |
| `CKPT_BUCKET`   | S3 bucket mounted at `/output` for checkpoints.    |
| `WANDB_API_KEY` | Forwarded to the worker for `wandb login`.         |
| `WANDB_PROJECT` | W&B project (default `gate-normalization`).        |
| `WANDB_ENTITY`  | W&B team / user (optional).                        |

## Resuming a preempted job

The trainer writes `last.pth.tar` on every epoch-end to `$OUTPUT_DIR`, which
is mounted to the `CKPT_BUCKET` and therefore survives VM termination. It
also writes a `wandb_run_id.json` sidecar so a resumed run reconnects to
the same W&B run without any extra CLI flag. A fresh `sky jobs launch` on
the same output directory resumes cleanly.
