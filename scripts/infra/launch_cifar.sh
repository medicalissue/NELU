#!/usr/bin/env bash
# CIFAR-100 campaign launcher.
#
# Same machinery as launch_campaign.sh (watchdog keeps a spot-worker pool
# alive until every <exp>/complete sentinel is on S3), but with three
# CIFAR-specific overrides baked in:
#
#   * ENTRY_SCRIPT  — scripts/orchestrate_cifar.sh (GPU-slot fanout)
#   * JOB_ORDER     — parsed from default_job_order_cifar.txt
#   * INSTANCE_TYPE — defaults to g5.12xlarge (4×A10G) because CIFAR
#                     models are single-GPU-per-job and a 4-GPU VM pulls
#                     four jobs in parallel via the slot fanout.
#
# Override any of these on the command line or in .env before invoking.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    # Load .env but give *caller-exported* values precedence, so
    # ``TARGET_WORKERS=1 INSTANCE_TYPE=g5.12xlarge bash launch_cifar.sh``
    # isn't silently clobbered by whatever the .env file contains.
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
    echo "FATAL: $ENV_FILE not found. Copy .env.example → .env and fill it in." >&2
    exit 2
fi

: "${CAMPAIGN_AZS:?CAMPAIGN_AZS missing from .env (e.g. \"us-west-2d us-west-2c\")}"
: "${TARGET_WORKERS:=1}"                     # one 4-GPU VM drains 105 runs in ~15h
: "${INSTANCE_TYPE:=g5.12xlarge}"             # 4×A10G, cheapest CIFAR-friendly spot
export TARGET_WORKERS INSTANCE_TYPE

# Drive the worker via the CIFAR orchestrator rather than the default
# ImageNet one. render_user_data.sh already propagates ENTRY_SCRIPT to
# the worker's bootstrap.
export ENTRY_SCRIPT="scripts/orchestrate_cifar.sh"

# Tag our workers with this campaign so the watchdog and the ImageNet
# campaign don't see each other's instances when counting "live workers".
export CAMPAIGN="cifar"

# CIFAR jobs are <cfg>:<act>:<seed> triples; the ImageNet default_job_order
# file is <cfg>:<act> pairs. Parse the CIFAR-specific file instead.
if [[ -z "${JOB_ORDER:-}" ]]; then
    JOB_ORDER=$(grep -v '^\s*#' "$SCRIPT_DIR/default_job_order_cifar.txt" \
        | grep -v '^\s*$' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ //;s/ $//')
    echo "JOB_ORDER not set — parsed default_job_order_cifar.txt"
fi
export JOB_ORDER

: "${WANDB_PROJECT:=nelu-cifar}"
export WANDB_PROJECT

echo "CIFAR campaign starting:"
echo "  TARGET_WORKERS = $TARGET_WORKERS"
echo "  INSTANCE_TYPE  = $INSTANCE_TYPE"
echo "  ENTRY_SCRIPT   = $ENTRY_SCRIPT"
echo "  CAMPAIGN_AZS   = $CAMPAIGN_AZS"
echo "  CKPT_BUCKET    = $CKPT_BUCKET"
echo "  WANDB_PROJECT  = $WANDB_PROJECT"
echo "  #jobs          = $(echo "$JOB_ORDER" | wc -w)"
echo

# Same watchdog, different JOB_ORDER + ENTRY_SCRIPT.
# Note: the watchdog's ``exp_complete`` / ``count_incomplete`` helpers
# parse JOB_ORDER as ``<cfg>:<act>:<seed>``-aware triples via the split
# on ':'; both fields land in the exp basename.
exec bash "$SCRIPT_DIR/watchdog.sh"
