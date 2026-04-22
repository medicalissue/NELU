#!/usr/bin/env bash
# Render scripts/infra/user-data.sh with env vars substituted, output to
# stdout (base64-encoded so `aws ec2 run-instances --user-data` accepts
# it inline without further escaping).
#
# Required env (usually sourced from .env):
#   REPO_URL, REPO_REF
#   VENV_S3_URL
#   CKPT_BUCKET
#   WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
#   AWS_DEFAULT_REGION
#
# Optional:
#   JOB_ORDER   (defaults to orchestrate.sh's built-in list if empty)

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template="$SCRIPT_DIR/user-data.sh"

sed \
  -e "s|@@REPO_URL@@|$REPO_URL|g" \
  -e "s|@@REPO_REF@@|$REPO_REF|g" \
  -e "s|@@VENV_S3_URL@@|$VENV_S3_URL|g" \
  -e "s|@@CKPT_BUCKET@@|$CKPT_BUCKET|g" \
  -e "s|@@WANDB_API_KEY@@|$WANDB_API_KEY|g" \
  -e "s|@@WANDB_PROJECT@@|$WANDB_PROJECT|g" \
  -e "s|@@WANDB_ENTITY@@|$WANDB_ENTITY|g" \
  -e "s|@@AWS_DEFAULT_REGION@@|$AWS_DEFAULT_REGION|g" \
  -e "s|@@JOB_ORDER@@|$JOB_ORDER|g" \
  "$template"
