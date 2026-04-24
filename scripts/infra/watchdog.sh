#!/usr/bin/env bash
# Spot-fleet watchdog. Keeps TARGET_WORKERS live in the AZs listed in
# CAMPAIGN_AZS until every experiment has a completion sentinel in S3.
#
# Behavior:
#   * Maintain TARGET_WORKERS spot instances tagged Project=gate-norm,
#     Role=worker. If fewer are live, launch one by rotating through
#     CAMPAIGN_AZS in user-specified order.
#   * Completion: when every experiment key in JOB_ORDER has a `complete`
#     object under $CKPT_BUCKET/<exp>/, exit 0.
#   * Capacity retry: if run-instances fails in all CAMPAIGN_AZS, sleep
#     POLL_INTERVAL_SEC and try again — forever. Never escalates to
#     on-demand, never gives up.
#
# Required env (usually from .env):
#   CKPT_BUCKET         s3://nelu-checkpoints[/prefix]
#   WANDB_API_KEY
#   JOB_ORDER           space-separated <cfg>:<act> pairs
#   CAMPAIGN_AZS        space-separated AZs, e.g. "us-west-2d us-west-2c"
#                       — you pick the order. Launch tries each in turn;
#                       after the last AZ fails it sleeps and starts
#                       over from the first.
#
# Optional env:
#   TARGET_WORKERS=2
#   INSTANCE_TYPE=p5.48xlarge
#   POLL_INTERVAL_SEC=60  idle sleep between full passes
#   CAPACITY_SLEEP_SEC=60 sleep between AZ-cycle retries on capacity fails
#   WANDB_PROJECT, WANDB_ENTITY   passed through to workers
#
# Usage:
#   source .env
#   bash scripts/infra/watchdog.sh
set -euo pipefail

: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"
: "${JOB_ORDER:?JOB_ORDER required (space-separated cfg:act pairs)}"
: "${CAMPAIGN_AZS:?CAMPAIGN_AZS required (e.g. \"us-west-2d us-west-2c\")}"
: "${TARGET_WORKERS:=2}"
: "${INSTANCE_TYPE:=p5.48xlarge}"
: "${POLL_INTERVAL_SEC:=60}"
: "${CAPACITY_SLEEP_SEC:=60}"
export CKPT_BUCKET WANDB_API_KEY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WORKER="$SCRIPT_DIR/run_worker.sh"

log() { printf '[watchdog %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

bucket_root="${CKPT_BUCKET%/}"
bucket_name=$(echo "$bucket_root" | sed -E 's|^s3://([^/]+).*|\1|')
bucket_prefix=$(echo "$bucket_root" | sed -E 's|^s3://[^/]+/?||')
[[ -n "$bucket_prefix" ]] && bucket_prefix="${bucket_prefix}/"

exp_complete() {
    # True iff $1 (an experiment basename) has a ``complete`` sentinel on S3.
    # timm writes ``<output>/<experiment>/complete`` — so the sync from
    # ``<outdir>`` to ``<CKPT_BUCKET>/<exp>`` mirrors that nested structure.
    # Older / flat layouts would land at ``<exp>/complete``; we tolerate both.
    local exp="$1"
    local key_nested="${bucket_prefix}${exp}/${exp}/complete"
    local key_flat="${bucket_prefix}${exp}/complete"
    if aws s3api head-object --bucket "$bucket_name" --key "$key_nested" \
            >/dev/null 2>&1; then
        return 0
    fi
    if aws s3api head-object --bucket "$bucket_name" --key "$key_flat" \
            >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Parse a queue entry into its experiment basename. Accepts either the
# ImageNet "<cfg>:<act>" pair or the CIFAR "<cfg>:<act>:<seed>" triple;
# the resulting exp mirrors what the orchestrators compute for S3 prefixes.
_exp_from_entry() {
    local entry="$1"
    local cfg act seed
    IFS=: read -r cfg act seed <<<"$entry"
    local base
    base=$(basename "${cfg%.yaml}")
    if [[ -n "$seed" ]]; then
        echo "${base}-${act}-s${seed}"
    else
        echo "${base}-${act}"
    fi
}

all_done() {
    for entry in $JOB_ORDER; do
        local exp
        exp=$(_exp_from_entry "$entry")
        if ! exp_complete "$exp"; then
            return 1
        fi
    done
    return 0
}

count_incomplete() {
    local n=0
    for entry in $JOB_ORDER; do
        local exp
        exp=$(_exp_from_entry "$entry")
        if ! exp_complete "$exp"; then
            n=$((n + 1))
        fi
    done
    echo "$n"
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
    # Try each AZ in user-specified order. If all of them reject us
    # (capacity / price / throttle), sleep CAPACITY_SLEEP_SEC and loop
    # forever — never escalates to on-demand, never gives up.
    local az_cycle=($CAMPAIGN_AZS)
    local cycle=0
    while :; do
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
        cycle=$((cycle + 1))
        log "AZ cycle $cycle exhausted — sleeping ${CAPACITY_SLEEP_SEC}s and retrying"
        sleep "$CAPACITY_SLEEP_SEC"
    done
}

log "starting. target=$TARGET_WORKERS type=$INSTANCE_TYPE AZs=($CAMPAIGN_AZS)"
while :; do
    if all_done; then
        log "queue drained — every experiment has a complete sentinel. exiting."
        break
    fi

    live=$(count_live_workers)
    remaining=$(count_incomplete)
    # Cap effective target by the number of incomplete experiments.
    # Without this cap, if TARGET_WORKERS=3 but only 1 exp is left in the
    # queue, a freshly-launched extra worker finds nothing to do, self-
    # terminates, and the watchdog re-launches → launch/terminate loop
    # burning spot $.
    effective_target=$TARGET_WORKERS
    if (( remaining < effective_target )); then
        effective_target=$remaining
    fi
    log "live workers: $live / $effective_target  (TARGET=$TARGET_WORKERS, remaining_jobs=$remaining)"

    while (( live < effective_target )); do
        # launch_one retries indefinitely; no fall-through path needed.
        launch_one
        live=$(count_live_workers)
    done

    sleep "$POLL_INTERVAL_SEC"
done
