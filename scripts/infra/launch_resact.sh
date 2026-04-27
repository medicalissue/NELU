#!/usr/bin/env bash
# ResAct ablation launcher.
#
# Same machinery as launch_beta.sh — watchdog keeps spot workers alive
# until every <exp>/complete sentinel is on S3 — but with the 4-job
# ResAct queue and a separate WANDB_PROJECT/CAMPAIGN tag.
#
# 4 jobs (resact_gelu_a5×{42,43} + resact_gelu_a0×{42,43}) on a single
# g5.12xlarge (4 GPU) instance → all run in parallel via the per-GPU
# slot fanout in orchestrate_beta.sh.
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
: "${INSTANCE_TYPE:=g5.12xlarge}"
export TARGET_WORKERS INSTANCE_TYPE

# Reuse β-pipeline fanout — per-GPU slot orchestrator + cls/ae split.
# All ResAct jobs are cls only.
export ENTRY_SCRIPT="scripts/orchestrate_beta.sh"
export CAMPAIGN="resact"

if [[ -z "${JOB_ORDER:-}" ]]; then
    JOB_ORDER=$(grep -v '^\s*#' "$SCRIPT_DIR/default_job_order_resact.txt" \
        | grep -v '^\s*$' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ //;s/ $//')
    echo "JOB_ORDER not set — parsed default_job_order_resact.txt"
fi
export JOB_ORDER

# Force ResAct W&B project regardless of what .env has — the imagenet-side
# default would lump these runs together with the main CIFAR/ImageNet
# campaign workspace. (Same pattern as launch_beta.sh.)
export WANDB_PROJECT="resact-ablation"

echo "ResAct campaign starting:"
echo "  TARGET_WORKERS = $TARGET_WORKERS"
echo "  INSTANCE_TYPE  = $INSTANCE_TYPE"
echo "  ENTRY_SCRIPT   = $ENTRY_SCRIPT"
echo "  CAMPAIGN_AZS   = $CAMPAIGN_AZS"
echo "  CKPT_BUCKET    = $CKPT_BUCKET"
echo "  WANDB_PROJECT  = $WANDB_PROJECT"
echo "  #jobs          = $(echo "$JOB_ORDER" | wc -w)"
echo "  jobs:"
for j in $JOB_ORDER; do echo "    $j"; done
echo

exec bash "$SCRIPT_DIR/watchdog.sh"
