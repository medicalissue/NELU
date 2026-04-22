#!/usr/bin/env bash
# Spot-fleet watchdog. Keeps N workers alive across the us-west-2 AZs
# until every experiment in the queue has a completion sentinel in S3.
#
# Behavior:
#   * Maintain up to `TARGET_WORKERS` running spot instances tagged
#     Project=gate-norm,Role=worker. If fewer are running, launch more
#     via scripts/infra/run_worker.sh, rotating through AZ_LIST.
#   * Detect completion: if every experiment key in JOB_ORDER has a
#     `complete` object under $CKPT_BUCKET/<exp>/, stop the watchdog.
#   * Retry capacity errors: if run-instances fails (most common: no
#     spot capacity in the chosen AZ), move to the next AZ and keep
#     trying. Sleep POLL_INTERVAL_SEC between full passes.
#
# Required env (loaded from .env by the caller):
#   CKPT_BUCKET           e.g. s3://nelu-checkpoints or s3://…/_dryrun
#   WANDB_API_KEY         passed through to every worker
#   WANDB_PROJECT, WANDB_ENTITY (optional)
#   JOB_ORDER             space-separated <config>:<activation>; used to
#                         know when "all done"
#
# Optional env:
#   TARGET_WORKERS=2      desired live-worker count
#   INSTANCE_TYPE=p5.48xlarge
#   AZ_LIST="us-west-2d us-west-2a us-west-2b us-west-2c"
#   POLL_INTERVAL_SEC=60  seconds between watchdog passes
#   MAX_LAUNCH_RETRIES=8  max capacity-retry loops across the AZ list
#                         before giving up on a single launch attempt
#
# Usage:
#   source .env
#   CKPT_BUCKET=s3://nelu-checkpoints \
#   JOB_ORDER="..." \
#   TARGET_WORKERS=2 \
#   bash scripts/infra/watchdog.sh
set -euo pipefail

: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"
: "${JOB_ORDER:?JOB_ORDER required (space-separated cfg:act pairs)}"
: "${TARGET_WORKERS:=2}"
: "${INSTANCE_TYPE:=p5.48xlarge}"
: "${AZ_LIST:=us-west-2d us-west-2a us-west-2b us-west-2c}"
: "${POLL_INTERVAL_SEC:=60}"
: "${MAX_LAUNCH_RETRIES:=8}"
export CKPT_BUCKET WANDB_API_KEY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WORKER="$SCRIPT_DIR/run_worker.sh"

log() { printf '[watchdog %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

bucket_root="${CKPT_BUCKET%/}"
bucket_name=$(echo "$bucket_root" | sed -E 's|^s3://([^/]+).*|\1|')
bucket_prefix=$(echo "$bucket_root" | sed -E 's|^s3://[^/]+/?||')
[[ -n "$bucket_prefix" ]] && bucket_prefix="${bucket_prefix}/"

all_done() {
    # Every experiment in JOB_ORDER must have a `complete` object.
    for pair in $JOB_ORDER; do
        local cfg act exp key
        IFS=: read -r cfg act <<<"$pair"
        local base
        base=$(basename "${cfg%.yaml}")
        exp="${base}-${act}"
        key="${bucket_prefix}${exp}/complete"
        if ! aws s3api head-object --bucket "$bucket_name" --key "$key" \
                >/dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

count_live_workers() {
    # Running/pending spot instances tagged as our workers.
    aws ec2 describe-instances \
        --region us-west-2 \
        --filters \
            "Name=tag:Project,Values=gate-norm" \
            "Name=tag:Role,Values=worker" \
            "Name=instance-state-name,Values=pending,running" \
        --query 'length(Reservations[].Instances[])' \
        --output text
}

launch_one() {
    # Try each AZ until one accepts us. MAX_LAUNCH_RETRIES caps the
    # outer loop so we don't spin forever on a totally-dead region.
    local attempt=0 az_cycle=($AZ_LIST)
    while (( attempt < MAX_LAUNCH_RETRIES )); do
        for az in "${az_cycle[@]}"; do
            local suffix
            suffix="$(date -u +%Y%m%dT%H%M%S)-${az##*-}"
            log "launching worker in $az ($INSTANCE_TYPE)"
            if bash "$RUN_WORKER" "$az" "$INSTANCE_TYPE" "$suffix" \
                    > /tmp/launch-$$.log 2>&1; then
                log "launch OK"
                cat /tmp/launch-$$.log | head -10
                rm -f /tmp/launch-$$.log
                return 0
            fi
            local err
            err=$(tail -5 /tmp/launch-$$.log | tr '\n' ' ')
            log "launch in $az failed: $err"
        done
        attempt=$((attempt + 1))
        log "AZ cycle exhausted (attempt $attempt/$MAX_LAUNCH_RETRIES) — sleeping 30s"
        sleep 30
    done
    log "give-up: $MAX_LAUNCH_RETRIES AZ-cycles all failed"
    return 1
}

log "starting. target=$TARGET_WORKERS type=$INSTANCE_TYPE AZs=($AZ_LIST)"
while :; do
    if all_done; then
        log "queue drained — every experiment has a complete sentinel. exiting."
        break
    fi

    live=$(count_live_workers)
    log "live workers: $live / $TARGET_WORKERS"

    while (( live < TARGET_WORKERS )); do
        if ! launch_one; then
            log "skipping this pass, will retry in ${POLL_INTERVAL_SEC}s"
            break
        fi
        live=$(count_live_workers)
    done

    sleep "$POLL_INTERVAL_SEC"
done
