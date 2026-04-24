#!/usr/bin/env bash
# Main-campaign entry point.
#
#   1. Loads .env (AWS, W&B, bucket names, AZ allow-list, worker pool size).
#   2. Exec's scripts/infra/watchdog.sh, which keeps TARGET_WORKERS alive
#      until every experiment has a `complete` sentinel.
#
# Stop with Ctrl-C. Watchdog will exit on its own when the queue drains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    # Preserve caller-provided overrides so CLI-like invocation
    #   TARGET_WORKERS=1 INSTANCE_TYPE=g5.12xlarge bash launch_campaign.sh
    # isn't silently clobbered by the .env file.
    _caller_TARGET_WORKERS="${TARGET_WORKERS-}"
    _caller_INSTANCE_TYPE="${INSTANCE_TYPE-}"
    _caller_JOB_ORDER="${JOB_ORDER-}"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    [[ -n "$_caller_TARGET_WORKERS" ]] && TARGET_WORKERS="$_caller_TARGET_WORKERS"
    [[ -n "$_caller_INSTANCE_TYPE"  ]] && INSTANCE_TYPE="$_caller_INSTANCE_TYPE"
    [[ -n "$_caller_JOB_ORDER"      ]] && JOB_ORDER="$_caller_JOB_ORDER"
else
    echo "FATAL: $ENV_FILE not found. Copy .env.example → .env and fill it in." >&2
    exit 2
fi

: "${CAMPAIGN_AZS:?CAMPAIGN_AZS missing from .env (e.g. \"us-west-2d us-west-2c\")}"
: "${TARGET_WORKERS:=2}"

# Derive JOB_ORDER from the flat-file default unless the caller set it.
if [[ -z "${JOB_ORDER:-}" ]]; then
    JOB_ORDER=$(grep -v '^\s*#' "$SCRIPT_DIR/default_job_order.txt" \
        | grep -v '^\s*$' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ //;s/ $//')
    echo "JOB_ORDER not set — parsed default_job_order.txt"
fi
export JOB_ORDER

echo "Campaign starting:"
echo "  TARGET_WORKERS = $TARGET_WORKERS"
echo "  INSTANCE_TYPE  = ${INSTANCE_TYPE:-p5.48xlarge}"
echo "  CAMPAIGN_AZS   = $CAMPAIGN_AZS"
echo "  CKPT_BUCKET    = $CKPT_BUCKET"
echo "  JOB_ORDER      = $JOB_ORDER"
echo

exec bash "$SCRIPT_DIR/watchdog.sh"
