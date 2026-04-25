#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# CIFAR-100 slot runner (one process per GPU).
#
# Pulls jobs from the shared S3 lease queue and runs them one at a time
# on the GPU selected by ``CUDA_VISIBLE_DEVICES`` (set by the fanout
# script). The queue logic — lease claim, heartbeat, preempt forwarding,
# final sync — is the same as the ImageNet orchestrator, only driving
# train.cifar with a seed dimension.
#
# Queue layout under $CKPT_BUCKET :
#
#     <CKPT_BUCKET>/<exp>/             # <exp> = <cfg>-<act>-s<seed>
#         complete        sentinel
#         lease           "<instance-id>-g<gpu> <unix-ts>"
#         checkpoint.pt   train.cifar CheckpointSaver output
#
# The lease owner string includes the GPU slot suffix so a single VM
# running N slots can hold N distinct leases without the slot races
# clobbering each other. Trainer pgid files are also per-slot.
#
# Required env (inherited from the fanout script):
#   CUDA_VISIBLE_DEVICES, GATE_NORM_GPU_SLOT
#   CKPT_BUCKET, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY,
#   AWS_DEFAULT_REGION
#
# Optional:
#   JOB_ORDER            whitespace-separated <cfg>:<act>:<seed> triples
#   LEASE_TTL, HEARTBEAT_EVERY, MAX_IDLE_PASSES
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${GATE_NORM_GPU_SLOT:?GATE_NORM_GPU_SLOT must be set by the fanout script}"
SLOT="$GATE_NORM_GPU_SLOT"

log() {
    printf '[slot-%s %s] %s\n' "$SLOT" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

: "${CKPT_BUCKET:?CKPT_BUCKET not set (e.g. s3://nelu-checkpoints)}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${WANDB_PROJECT:=nelu-cifar}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=1}"
export AWS_DEFAULT_REGION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${JOB_ORDER:-}" ]]; then
    _job_file="$REPO_ROOT/scripts/infra/default_job_order_cifar.txt"
    if [[ -f "$_job_file" ]]; then
        JOB_ORDER=$(grep -v '^\s*#' "$_job_file" | grep -v '^\s*$' | tr '\n' ' ')
    fi
fi
: "${JOB_ORDER:?JOB_ORDER is empty and default_job_order_cifar.txt not found}"

# ── IMDSv2 helpers ─────────────────────────────────────────────────
imds() {
    local token
    token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    curl -sS -H "X-aws-ec2-metadata-token: $token" \
         "http://169.254.169.254/latest/meta-data/$1"
}

AZ=$(imds placement/availability-zone 2>/dev/null || echo "unknown")
INSTANCE_ID=$(imds instance-id 2>/dev/null || echo "unknown")
# Lease owner must be unique per (VM, GPU) so multiple slots on the
# same VM don't clobber each other.
OWNER="${INSTANCE_ID}-g${SLOT}"
PGID_FILE="/tmp/trainer-g${SLOT}.pgid"
log "AZ=$AZ instance=$INSTANCE_ID owner=$OWNER"

DATA_MOUNT=/data
check_data_mount() {
    local cifar_dir="$DATA_MOUNT/cifar-100-python"
    if [[ ! -f "$cifar_dir/meta" || ! -f "$cifar_dir/train" || ! -f "$cifar_dir/test" ]]; then
        log "FATAL: $cifar_dir is missing the CIFAR-100 pickle files."
        log "Snapshot likely omits CIFAR-100; re-bake with scripts/prepare_data.sh."
        log "contents of $DATA_MOUNT:"; ls -l "$DATA_MOUNT" 2>&1 | head -20 >&2 || true
        exit 1
    fi
    log "data volume OK"
}

# ── Preempt watcher ────────────────────────────────────────────────
start_preempt_watcher() {
    (
        while :; do
            token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
            code=$(curl -sS -o /dev/null -w '%{http_code}' \
                   -H "X-aws-ec2-metadata-token: $token" \
                   "http://169.254.169.254/latest/meta-data/spot/instance-action" 2>/dev/null || echo "000")
            if [[ "$code" == "200" ]]; then
                if [[ -f "$PGID_FILE" ]]; then
                    pgid=$(cat "$PGID_FILE")
                    log "preempt notice — SIGTERM to pgid=$pgid"
                    kill -TERM -"$pgid" 2>/dev/null || true
                fi
                break
            fi
            sleep 5
        done
    ) &
}

# ── Queue plumbing ─────────────────────────────────────────────────

exp_key() {
    local cfg="$1" act="$2" seed="$3"
    local base
    base=$(basename "${cfg%.yaml}")
    echo "${base}-${act}-s${seed}"
}

s3_exists() { aws s3 ls "$1" >/dev/null 2>&1; }

lease_claim() {
    # Optimistic claim with a read-back confirmation step:
    #   1. Reject if a fresh lease already exists.
    #   2. PUT our own (owner, ts).
    #   3. Sleep a small jitter, then GET the lease back. If another slot
    #      PUT after us, that slot's owner wins; we lose and must skip.
    # This catches the common N-slot-at-boot concurrent-claim race that S3
    # strong read-after-write alone does not prevent (both slots see
    # "no lease" before either PUTs), without requiring conditional-write
    # support from the bucket.
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner ts age
    now=$(date +%s)
    if s3_exists "$key"; then
        read -r owner ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
        age=$((now - ts))
        if (( age < LEASE_TTL )); then
            return 1
        fi
        log "stealing stale lease on $exp (age=${age}s, owner=$owner)"
    fi
    echo "$OWNER $now" | aws s3 cp - "$key" >/dev/null
    # Jitter [0.5s, 1.5s) so concurrent PUTs from sibling slots settle.
    sleep "$(awk "BEGIN{print 0.5+rand()}")"
    local winner
    read -r winner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    if [[ "$winner" != "$OWNER" ]]; then
        log "lost lease race on $exp (winner=$winner, we=$OWNER)"
        return 1
    fi
    return 0
}

lease_refresh() {
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner
    now=$(date +%s)
    read -r owner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    if [[ "$owner" != "$OWNER" ]]; then
        return 1
    fi
    echo "$OWNER $now" | aws s3 cp - "$key" >/dev/null
    return 0
}

lease_release() {
    aws s3 rm "${CKPT_BUCKET}/$1/lease" >/dev/null 2>&1 || true
}

run_job() {
    local cfg="$1" act="$2" seed="$3"
    local exp
    exp=$(exp_key "$cfg" "$act" "$seed")
    local s3_prefix="${CKPT_BUCKET}/${exp}"

    if s3_exists "${s3_prefix}/complete"; then
        log "skip ${exp} (complete)"
        return 3
    fi
    if ! lease_claim "$exp"; then
        log "skip ${exp} (fresh lease held)"
        return 2
    fi

    log "▶ running ${exp}"
    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"

    aws s3 sync "${s3_prefix}/" "${outdir}/" \
        --exclude "lease" --exclude "complete" \
        --exact-timestamps --only-show-errors || true

    # Defense-in-depth: if S3 has a checkpoint for this exp but we didn't
    # land one locally, the sync silently failed. Don't smash the partial
    # run by restarting from scratch — bail loudly and let another worker
    # try on a fresh lease.
    local s3_has_ckpt=0
    if aws s3 ls "${s3_prefix}/checkpoint.pt" >/dev/null 2>&1; then
        s3_has_ckpt=1
    fi
    if (( s3_has_ckpt == 1 )) && [[ ! -f "${outdir}/checkpoint.pt" ]]; then
        log "FATAL: S3 has checkpoint.pt for ${exp} but local copy missing after sync"
        log "  s3_prefix=${s3_prefix}  outdir=${outdir}"
        log "  outdir contents:"; ls -lR "$outdir" >&2 || true
        lease_release "$exp"
        rm -rf "$outdir"
        return 1
    fi

    local resume_flag=()
    if [[ -f "${outdir}/checkpoint.pt" ]]; then
        # Spot preempts can leave a partial checkpoint.pt in S3 (sync
        # interrupted mid-transfer). A real CIFAR ckpt is 2-15 MB; anything
        # below 1 MB is partial, so wipe it locally + on S3 and start fresh.
        local sz
        sz=$(stat -c %s "${outdir}/checkpoint.pt" 2>/dev/null || echo 0)
        if (( sz < 1000000 )); then
            log "  WARN: ${outdir}/checkpoint.pt is only ${sz} bytes (corrupted partial); discarding + restarting from scratch"
            rm -f "${outdir}/checkpoint.pt"
            aws s3 rm "${s3_prefix}/checkpoint.pt" --only-show-errors || true
        else
            resume_flag=(--resume "${outdir}/checkpoint.pt")
            log "  resuming from ${outdir}/checkpoint.pt (${sz} bytes)"
        fi
    fi

    local entity_env=()
    if [[ -n "$WANDB_ENTITY" ]]; then
        entity_env=(WANDB_ENTITY="$WANDB_ENTITY")
    fi

    # Heartbeat: lease_refresh fails (lease stolen) → exit so the
    # slot's run_job teardown notices and bails.
    (
        while sleep "$HEARTBEAT_EVERY"; do
            if ! lease_refresh "$exp"; then
                log "  lease lost — stopping heartbeat"
                # Signal the trainer pgid so we don't leave an orphan.
                if [[ -f "$PGID_FILE" ]]; then
                    pgid=$(cat "$PGID_FILE")
                    kill -TERM -"$pgid" 2>/dev/null || true
                fi
                exit 0
            fi
            aws s3 sync "${outdir}/" "${s3_prefix}/" \
                --exclude "*.tmp" --exclude "lease" \
                --only-show-errors || true
        done
    ) &
    local heartbeat_pid=$!

    # setsid → trainer becomes its own session leader, so the preempt
    # watcher can SIGTERM the whole process tree by pgid (== pid).
    # GATE_NORM_FORCE_PYTHON=1: bench on A10G shows the native PyTorch path
    # under torch.compile is faster than the fused CUDA kernel for CIFAR's
    # large reduction axis (sample axes, N=CHW=16384), where the kernel falls
    # to its Tier-3 global-atomicAdd fallback. ImageNet keeps the fused path.
    setsid env "${entity_env[@]}" GATE_NORM_FORCE_PYTHON=1 python -m train.cifar \
        --config "$cfg" \
        --activation "$act" \
        --seed "$seed" \
        --output_dir "$outdir" \
        --wandb \
        --wandb_project "$WANDB_PROJECT" \
        "${resume_flag[@]}" &
    local trainer_pid=$!
    echo "$trainer_pid" > "$PGID_FILE"
    local trainer_rc=0
    wait "$trainer_pid" || trainer_rc=$?

    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f "$PGID_FILE"

    aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "lease" \
        --only-show-errors || true

    if [[ $trainer_rc -eq 0 && -f "${outdir}/complete" ]]; then
        log "✓ ${exp} complete (rc=$trainer_rc)"
    elif [[ $trainer_rc -eq 0 ]]; then
        log "⏸ ${exp} paused (clean exit, no sentinel)"
    else
        log "✗ ${exp} failed (rc=$trainer_rc)"
    fi

    lease_release "$exp"
    rm -rf "$outdir"
    return 0
}

# ── Main ───────────────────────────────────────────────────────────
check_data_mount
start_preempt_watcher

if command -v wandb >/dev/null; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

idle_passes=0
while (( idle_passes < MAX_IDLE_PASSES )); do
    ran_any=0
    for triple in $JOB_ORDER; do
        IFS=: read -r cfg act seed <<<"$triple"
        if [[ -z "$cfg" || -z "$act" || -z "$seed" ]]; then
            log "skipping malformed triple: $triple"
            continue
        fi
        rc=0
        run_job "$cfg" "$act" "$seed" || rc=$?
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

log "queue drained — slot exiting"
