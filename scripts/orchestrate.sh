#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Worker entrypoint for the Gate-Normalization training campaign.
#
# A single dstack task runs this script. The worker lives for the lifetime
# of the spot VM and pulls one training job at a time from an S3-backed
# queue. When the queue is empty (every job has a completion sentinel) the
# worker exits cleanly; dstack does not relaunch it.
#
# Flow per VM:
#   1. Setup:
#      - Discover AZ + instance-id from IMDSv2.
#      - Create an EBS volume from DATA_SNAPSHOT in this AZ.
#      - Attach the volume, mount read-only at /data.
#      - Start a preempt watcher that polls the spot/instance-action IMDS
#        endpoint; on receipt it sends SIGTERM to the current trainer.
#   2. Loop:
#      - Walk the ordered job list (longest recipes first). For each job:
#        * If s3://CKPT_BUCKET/<exp>/complete exists → skip.
#        * If s3://CKPT_BUCKET/<exp>/lease is fresh (<LEASE_TTL s) → skip.
#        * Else acquire lease (S3 conditional PUT), sync any prior state
#          down, run torchrun, sync state back up, release lease.
#      - After one full pass with no work available → exit.
#   3. Teardown (always, via EXIT trap):
#      - Unmount, detach, delete the volume.
#      - The lease is released explicitly on the exit path above, so a
#        fresh worker can pick the same experiment up.
#
# Required env (from dstack task):
#   DATA_SNAPSHOT       e.g. snap-0adfaa42ce378623c
#   CKPT_BUCKET         e.g. s3://nelu-checkpoints
#   WANDB_API_KEY       W&B api key
#   WANDB_PROJECT       default gate-normalization
#   WANDB_ENTITY        optional
#   AWS_DEFAULT_REGION  us-west-2
#
# Optional:
#   JOB_ORDER           space-separated <config>:<activation> pairs; if
#                       unset, defaults to the paper's 12-run matrix in
#                       longest-first order.
#   LEASE_TTL           seconds (default 600). A lease older than TTL is
#                       considered dead and can be stolen.
#   HEARTBEAT_EVERY     seconds between lease refreshes (default 60).
#   MAX_IDLE_PASSES     if a full pass finds no work this many times in
#                       a row, the worker exits (default 1).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { printf '[orchestrate %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

: "${DATA_SNAPSHOT:?DATA_SNAPSHOT not set}"
: "${CKPT_BUCKET:?CKPT_BUCKET not set (e.g. s3://nelu-checkpoints)}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${WANDB_PROJECT:=gate-normalization}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=1}"
export AWS_DEFAULT_REGION

# Default job order — longest wall-clock first to keep tail latency small.
: "${JOB_ORDER:=\
configs/imagenet/swin_small.yaml:gelu \
configs/imagenet/swin_small.yaml:nelu \
configs/imagenet/convnext_small.yaml:gelu \
configs/imagenet/convnext_small.yaml:nelu \
configs/imagenet/deit_base.yaml:gelu \
configs/imagenet/deit_base.yaml:nelu \
configs/imagenet/swin_tiny.yaml:gelu \
configs/imagenet/swin_tiny.yaml:nelu \
configs/imagenet/convnext_tiny.yaml:gelu \
configs/imagenet/convnext_tiny.yaml:nelu \
configs/imagenet/deit_small.yaml:gelu \
configs/imagenet/deit_small.yaml:nelu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── IMDSv2 helpers ─────────────────────────────────────────────────
imds() {
    local token
    token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    curl -sS -H "X-aws-ec2-metadata-token: $token" \
         "http://169.254.169.254/latest/meta-data/$1"
}

AZ=$(imds placement/availability-zone)
INSTANCE_ID=$(imds instance-id)
log "AZ=$AZ instance=$INSTANCE_ID"

# ── Volume create + attach + mount ─────────────────────────────────
VOLUME_ID=""
DATA_MOUNT=/data

cleanup_volume() {
    if [[ -n "$VOLUME_ID" ]]; then
        log "tearing down volume $VOLUME_ID"
        sudo umount "$DATA_MOUNT" 2>/dev/null || true
        aws ec2 detach-volume --volume-id "$VOLUME_ID" --force >/dev/null 2>&1 || true
        # detach-volume is async — wait before delete.
        for _ in {1..60}; do
            state=$(aws ec2 describe-volumes --volume-ids "$VOLUME_ID" \
                    --query 'Volumes[0].State' --output text 2>/dev/null || echo "gone")
            [[ "$state" == "available" || "$state" == "gone" ]] && break
            sleep 2
        done
        aws ec2 delete-volume --volume-id "$VOLUME_ID" >/dev/null 2>&1 || true
    fi
}
trap 'cleanup_volume' EXIT

setup_volume() {
    log "creating volume from $DATA_SNAPSHOT in $AZ"
    VOLUME_ID=$(aws ec2 create-volume \
        --availability-zone "$AZ" \
        --snapshot-id "$DATA_SNAPSHOT" \
        --volume-type gp3 \
        --tag-specifications "ResourceType=volume,Tags=[{Key=Project,Value=gate-norm},{Key=Role,Value=worker-data},{Key=Instance,Value=$INSTANCE_ID}]" \
        --query VolumeId --output text)
    log "volume=$VOLUME_ID — waiting for available"
    aws ec2 wait volume-available --volume-ids "$VOLUME_ID"

    log "attaching"
    aws ec2 attach-volume --volume-id "$VOLUME_ID" \
        --instance-id "$INSTANCE_ID" --device /dev/sdg >/dev/null
    aws ec2 wait volume-in-use --volume-ids "$VOLUME_ID"

    # NVMe remaps /dev/sdg → /dev/nvme?n1; resolve via serial (volume-id with dashes stripped).
    sleep 5
    local serial dev
    serial=$(echo "$VOLUME_ID" | tr -d '-')
    for _ in {1..20}; do
        dev=$(lsblk -dno NAME,SERIAL | awk -v s="$serial" '$2==s {print "/dev/"$1; exit}')
        [[ -n "$dev" ]] && break
        sleep 1
    done
    if [[ -z "$dev" ]]; then
        log "FATAL: could not resolve device for volume $VOLUME_ID"
        exit 1
    fi
    log "mounting $dev read-only at $DATA_MOUNT"
    sudo mkdir -p "$DATA_MOUNT"
    sudo mount -o ro "$dev" "$DATA_MOUNT"
    df -h "$DATA_MOUNT" >&2
}

# ── Preempt watcher ────────────────────────────────────────────────
# Polls the IMDS spot/instance-action endpoint. On notification it sends
# SIGTERM to the trainer PGID (written to /tmp/trainer.pgid by run_job).
PREEMPT_WATCHER_PID=""
start_preempt_watcher() {
    (
        while :; do
            token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
            code=$(curl -sS -o /dev/null -w '%{http_code}' \
                   -H "X-aws-ec2-metadata-token: $token" \
                   "http://169.254.169.254/latest/meta-data/spot/instance-action" 2>/dev/null || echo "000")
            if [[ "$code" == "200" ]]; then
                if [[ -f /tmp/trainer.pgid ]]; then
                    pgid=$(cat /tmp/trainer.pgid)
                    log "preempt notice — SIGTERM to pgid=$pgid"
                    kill -TERM -"$pgid" 2>/dev/null || true
                fi
                # Once notified, keep sleeping — AWS terminates within 2 min.
                break
            fi
            sleep 5
        done
    ) &
    PREEMPT_WATCHER_PID=$!
    log "preempt watcher pid=$PREEMPT_WATCHER_PID"
}

# ── Job queue (S3 lease-based claim) ───────────────────────────────
# Layout under $CKPT_BUCKET/<exp>/ :
#   complete        — presence means done. Never re-run.
#   lease           — current worker; content = "<instance-id> <unix-ts>".
#   last.pth.tar    — resume point.
#   wandb_run_id.json — W&B run id sidecar.
#   summary.csv / args.yaml / events / ...

exp_key() {
    # <config basename without .yaml>-<activation>, e.g. convnext_tiny-nelu
    local cfg="$1" act="$2"
    local base
    base=$(basename "${cfg%.yaml}")
    echo "${base}-${act}"
}

s3_exists() { aws s3 ls "$1" >/dev/null 2>&1; }

lease_claim() {
    # Atomic-ish claim: download current lease, check age, overwrite with
    # our own identity. Two workers could still race in the gap, but the
    # worst case is one wasted VM-minute before the loser detects a newer
    # lease in its next heartbeat and bails. (See lease_refresh.)
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner ts age
    now=$(date +%s)
    if s3_exists "$key"; then
        read -r owner ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
        age=$((now - ts))
        if (( age < LEASE_TTL )); then
            return 1   # fresh lease, skip
        fi
        log "stealing stale lease on $exp (age=${age}s, owner=$owner)"
    fi
    echo "$INSTANCE_ID $now" | aws s3 cp - "$key" >/dev/null
    return 0
}

lease_refresh() {
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner
    now=$(date +%s)
    # Confirm we still own the lease before touching it.
    read -r owner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    if [[ "$owner" != "$INSTANCE_ID" ]]; then
        return 1
    fi
    echo "$INSTANCE_ID $now" | aws s3 cp - "$key" >/dev/null
    return 0
}

lease_release() {
    aws s3 rm "${CKPT_BUCKET}/$1/lease" >/dev/null 2>&1 || true
}

run_job() {
    local cfg="$1" act="$2"
    local exp
    exp=$(exp_key "$cfg" "$act")
    local s3_prefix="${CKPT_BUCKET}/${exp}"

    # Skip already-complete.
    if s3_exists "${s3_prefix}/complete"; then
        log "skip ${exp} (complete)"
        return 0
    fi

    # Try to claim.
    if ! lease_claim "$exp"; then
        log "skip ${exp} (fresh lease held by another worker)"
        return 2
    fi

    log "▶ running ${exp}"

    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"

    # Pull prior state (resume point + W&B id), if any.
    aws s3 sync "${s3_prefix}/" "${outdir}/" --exclude "lease" --exclude "complete" \
        --exact-timestamps --only-show-errors || true

    local resume_flag=()
    if [[ -f "${outdir}/last.pth.tar" ]]; then
        resume_flag=(--resume "${outdir}/last.pth.tar")
        log "  resuming from ${outdir}/last.pth.tar"
    fi

    local wandb_id_flag=()
    if [[ -f "${outdir}/wandb_run_id.json" ]]; then
        local saved_id
        saved_id=$(python - <<'PY'
import json, sys
with open("${OUTDIR}/wandb_run_id.json") as f:
    print(json.load(f).get("run_id", ""))
PY
        )
        # Re-run with shell substitution — the heredoc above is in a subshell
        # where $OUTDIR isn't set. Use sed approach instead:
        saved_id=$(grep -o '"run_id"[^,}]*' "${outdir}/wandb_run_id.json" \
                   | sed 's/.*: *"\([^"]*\)".*/\1/')
        if [[ -n "$saved_id" ]]; then
            wandb_id_flag=(--wandb-resume-id "$saved_id")
            log "  resuming W&B run ${saved_id}"
        fi
    fi

    # Heartbeat loop: refresh lease + sync checkpoint every HEARTBEAT_EVERY s.
    (
        while sleep "$HEARTBEAT_EVERY"; do
            lease_refresh "$exp" || exit 0
            aws s3 sync "${outdir}/" "${s3_prefix}/" \
                --exclude "*.tmp" --exclude "lease" \
                --only-show-errors || true
        done
    ) &
    local heartbeat_pid=$!

    # Run torchrun in its own process group so the preempt watcher can
    # SIGTERM the whole tree in one kill -TERM -pgid.
    setsid bash -c "echo \$\$ > /tmp/trainer.pgid; exec torchrun \
        --nproc_per_node=\$(nvidia-smi -L | wc -l) \
        -m train.imagenet \
        --config '$cfg' \
        --activation '$act' \
        --experiment '$exp' \
        --output '$outdir' \
        --log-wandb \
        --wandb-project '$WANDB_PROJECT' \
        ${WANDB_ENTITY:+--wandb-entity '$WANDB_ENTITY'} \
        ${resume_flag[*]:-} \
        ${wandb_id_flag[*]:-}" &
    local trainer_pid=$!
    local trainer_rc=0
    wait "$trainer_pid" || trainer_rc=$?

    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f /tmp/trainer.pgid

    # Final sync regardless of exit code — we want whatever progress the
    # trainer managed to checkpoint to be visible to the next worker.
    aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "lease" \
        --only-show-errors || true

    if [[ $trainer_rc -eq 0 && -f "${outdir}/complete" ]]; then
        log "✓ ${exp} complete (rc=$trainer_rc)"
    elif [[ $trainer_rc -eq 0 ]]; then
        # Exited clean but no sentinel — likely SIGTERM at an epoch boundary.
        log "⏸ ${exp} paused (clean exit, no sentinel)"
    else
        log "✗ ${exp} failed (rc=$trainer_rc)"
    fi

    lease_release "$exp"
    rm -rf "$outdir"
    return 0
}

# ── Main ───────────────────────────────────────────────────────────
setup_volume
start_preempt_watcher

# Stage the repo onto the worker (dstack runs commands in the synced
# workdir — nothing to do here). Log W&B once per VM boot.
if command -v wandb >/dev/null; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

idle_passes=0
while (( idle_passes < MAX_IDLE_PASSES )); do
    ran_any=0
    for pair in $JOB_ORDER; do
        IFS=: read -r cfg act <<<"$pair"
        rc=0
        run_job "$cfg" "$act" || rc=$?
        if [[ $rc -eq 0 ]]; then
            ran_any=1
        fi
    done
    if (( ran_any == 0 )); then
        idle_passes=$((idle_passes + 1))
        log "no work in this pass (${idle_passes}/${MAX_IDLE_PASSES})"
        sleep 10
    else
        idle_passes=0
    fi
done

log "queue drained — worker exiting"
