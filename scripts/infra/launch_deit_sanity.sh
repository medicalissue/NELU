#!/usr/bin/env bash
# DeiT-Base + new NELU sanity launch.
#
# Same machinery as launch_campaign.sh but:
#   - 1 worker (TARGET_WORKERS=1)
#   - p5.48xlarge spot (8× H100)
#   - 1 job: deit_base + nelu (new LN + per-channel γ_c, β_c, token-pool)
#   - separate W&B project so we don't pollute the main imagenet workspace
#
# Goal: verify the new NELU default survives a real ImageNet training
# step. We're not chasing convergence here — just want to see that
# forward/backward works, gradient stats are sane, and no NaN/Inf.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    _caller_TARGET_WORKERS="${TARGET_WORKERS-}"
    _caller_INSTANCE_TYPE="${INSTANCE_TYPE-}"
    _caller_JOB_ORDER="${JOB_ORDER-}"
    _caller_WANDB_PROJECT="${WANDB_PROJECT-}"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    [[ -n "$_caller_TARGET_WORKERS" ]] && TARGET_WORKERS="$_caller_TARGET_WORKERS"
    [[ -n "$_caller_INSTANCE_TYPE"  ]] && INSTANCE_TYPE="$_caller_INSTANCE_TYPE"
    [[ -n "$_caller_JOB_ORDER"      ]] && JOB_ORDER="$_caller_JOB_ORDER"
    [[ -n "$_caller_WANDB_PROJECT"  ]] && WANDB_PROJECT="$_caller_WANDB_PROJECT"
else
    echo "FATAL: $ENV_FILE not found." >&2
    exit 2
fi

: "${CAMPAIGN_AZS:?CAMPAIGN_AZS missing from .env}"
: "${TARGET_WORKERS:=1}"
: "${INSTANCE_TYPE:=p5.48xlarge}"
export TARGET_WORKERS INSTANCE_TYPE

export CAMPAIGN="deit_sanity"

if [[ -z "${JOB_ORDER:-}" ]]; then
    JOB_ORDER=$(grep -v '^\s*#' "$SCRIPT_DIR/default_job_order_deit_sanity.txt" \
        | grep -v '^\s*$' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ //;s/ $//')
    echo "JOB_ORDER not set — parsed default_job_order_deit_sanity.txt"
fi
export JOB_ORDER

# Force separate W&B project — sanity runs shouldn't mix with the main
# imagenet workspace.
export WANDB_PROJECT="deit-sanity"

echo "DeiT-Base + new NELU sanity launching:"
echo "  TARGET_WORKERS = $TARGET_WORKERS"
echo "  INSTANCE_TYPE  = $INSTANCE_TYPE"
echo "  CAMPAIGN_AZS   = $CAMPAIGN_AZS"
echo "  CKPT_BUCKET    = $CKPT_BUCKET"
echo "  WANDB_PROJECT  = $WANDB_PROJECT"
echo "  JOB_ORDER      = $JOB_ORDER"
echo

exec bash "$SCRIPT_DIR/watchdog.sh"
