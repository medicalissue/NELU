#!/usr/bin/env bash
# VM user-data: runs as root at first boot. Hands off to bootstrap.sh.
# Intentionally minimal — anything substantive belongs in bootstrap.sh.
#
# Env vars are populated by render_user_data.sh which substitutes the
# @@VAR@@ placeholders before the launch template is updated.

set -euo pipefail
exec > /var/log/user-data.log 2>&1
echo "[user-data] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"

export REPO_URL="@@REPO_URL@@"
export REPO_REF="@@REPO_REF@@"
export VENV_S3_URL="@@VENV_S3_URL@@"
export CKPT_BUCKET="@@CKPT_BUCKET@@"
export WANDB_PROJECT="@@WANDB_PROJECT@@"
export WANDB_ENTITY="@@WANDB_ENTITY@@"
export WANDB_API_KEY="@@WANDB_API_KEY@@"
export AWS_DEFAULT_REGION="@@AWS_DEFAULT_REGION@@"
export JOB_ORDER="@@JOB_ORDER@@"
# Eval-mode overrides (empty for training workers).
export ENTRY_SCRIPT="@@ENTRY_SCRIPT@@"
export EVAL_EPOCH="@@EVAL_EPOCH@@"
export EVAL_MODELS="@@EVAL_MODELS@@"
export EVAL_RESULT_PREFIX="@@EVAL_RESULT_PREFIX@@"

# Minimal repo clone just to access scripts/bootstrap.sh.
# The real workspace at /workspace is set up by bootstrap.sh itself,
# but we need the script now so we clone to a throwaway path.
mkdir -p /opt/bootstrap
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" /opt/bootstrap/nelu
exec bash /opt/bootstrap/nelu/scripts/bootstrap.sh
