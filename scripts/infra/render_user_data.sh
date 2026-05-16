#!/usr/bin/env bash
# Render scripts/infra/user-data.sh with env vars substituted, output to
# stdout. Python handles the substitution so arbitrary characters in
# values (slashes, pipes, newlines) pass through safely — unlike sed,
# whose delimiter rules differ between GNU and BSD.
#
# Required env (usually sourced from .env):
#   REPO_URL, REPO_REF, VENV_S3_URL, CKPT_BUCKET,
#   WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY,
#   AWS_DEFAULT_REGION
# Optional:
#   JOB_ORDER   (empty → orchestrate.sh uses its built-in default)

set -euo pipefail

: "${REPO_URL:=https://github.com/medicalissue/NELU.git}"
: "${REPO_REF:=public-release}"
: "${VENV_S3_URL:=s3://nelu-datasets/env/nelu-venv-py310-cu130.tar.gz}"
: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"
: "${WANDB_PROJECT:=gate-normalization}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${JOB_ORDER:=}"
: "${ENTRY_SCRIPT:=}"
: "${EVAL_EPOCH:=}"
: "${EVAL_MODELS:=}"
: "${EVAL_RESULT_PREFIX:=}"
: "${NUM_MEDMNIST_SLOTS:=}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template="$SCRIPT_DIR/user-data.sh"

export REPO_URL REPO_REF VENV_S3_URL CKPT_BUCKET WANDB_API_KEY \
       WANDB_PROJECT WANDB_ENTITY AWS_DEFAULT_REGION JOB_ORDER \
       ENTRY_SCRIPT EVAL_EPOCH EVAL_MODELS EVAL_RESULT_PREFIX \
       NUM_MEDMNIST_SLOTS

python3 - "$template" <<'PY'
import os, sys, pathlib
tpl = pathlib.Path(sys.argv[1]).read_text()
keys = [
    "REPO_URL", "REPO_REF", "VENV_S3_URL", "CKPT_BUCKET",
    "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY",
    "AWS_DEFAULT_REGION", "JOB_ORDER",
    "ENTRY_SCRIPT", "EVAL_EPOCH", "EVAL_MODELS", "EVAL_RESULT_PREFIX",
    "NUM_MEDMNIST_SLOTS",
]
for k in keys:
    tpl = tpl.replace(f"@@{k}@@", os.environ.get(k, ""))
sys.stdout.write(tpl)
PY
