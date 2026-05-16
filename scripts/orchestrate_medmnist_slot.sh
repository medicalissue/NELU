#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# MedMNIST v2 slot runner (one process per GPU).
#
# Mirrors scripts/orchestrate_cifar_slot.sh exactly — same S3 lease /
# heartbeat / preempt logic — with three MedMNIST-specific changes:
#
#   * Job entries are <dataset>:<model>:<act>:<seed> 4-tuples
#     (CIFAR was <cfg>:<act>:<seed> triples).
#   * Trainer is ``python -m train.medmnist`` (CIFAR drove train.cifar).
#   * No /data CIFAR-pickle mount check: the medmnist package downloads
#     its (tiny, ≤≈60 MB total) datasets on demand into --data_dir.
#
# Queue layout under $CKPT_BUCKET :
#
#     <CKPT_BUCKET>/<exp>/             # <exp> = <ds>-<model>-<act>-s<seed>
#         complete        sentinel (train.medmnist writes it)
#         lease           "<instance-id>-g<gpu> <unix-ts>"
#         checkpoint.pt   train.medmnist output
#
# Required env (inherited from the fanout script):
#   CUDA_VISIBLE_DEVICES, GATE_NORM_GPU_SLOT
#   CKPT_BUCKET, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY,
#   AWS_DEFAULT_REGION
#
# Optional:
#   JOB_ORDER            whitespace-separated <ds>:<model>:<act>:<seed>
#   LEASE_TTL, HEARTBEAT_EVERY, MAX_IDLE_PASSES, MEDMNIST_DATA_DIR
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${GATE_NORM_GPU_SLOT:?GATE_NORM_GPU_SLOT must be set by the fanout script}"
SLOT="$GATE_NORM_GPU_SLOT"

log() {
    printf '[mm-slot-%s %s] %s\n' "$SLOT" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

: "${CKPT_BUCKET:?CKPT_BUCKET not set (e.g. s3://nelu-checkpoints)}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${WANDB_PROJECT:=medmnist-gate-normalization}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=1}"
: "${MEDMNIST_DATA_DIR:=/tmp/medmnist_data}"
export AWS_DEFAULT_REGION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${JOB_ORDER:-}" ]]; then
    _job_file="$REPO_ROOT/scripts/infra/default_job_order_medmnist.txt"
    if [[ -f "$_job_file" ]]; then
        JOB_ORDER=$(grep -v '^\s*#' "$_job_file" | grep -v '^\s*$' | tr '\n' ' ')
    fi
fi
: "${JOB_ORDER:?JOB_ORDER is empty and default_job_order_medmnist.txt not found}"

# medmnist downloads into a shared dir; create it once up front so the
# package's "root must exist" guard passes for every slot.
mkdir -p "$MEDMNIST_DATA_DIR"

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
OWNER="${INSTANCE_ID}-g${SLOT}"
PGID_FILE="/tmp/mm-trainer-g${SLOT}.pgid"
log "AZ=$AZ instance=$INSTANCE_ID owner=$OWNER data_dir=$MEDMNIST_DATA_DIR"

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
    local ds="$1" model="$2" act="$3" seed="$4"
    echo "${ds}-${model}-${act}-s${seed}"
}

s3_exists() { aws s3 ls "$1" >/dev/null 2>&1; }

lease_claim() {
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
    local ds="$1" model="$2" act="$3" seed="$4"
    local exp
    exp=$(exp_key "$ds" "$model" "$act" "$seed")
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

    local s3_has_ckpt=0
    if aws s3 ls "${s3_prefix}/checkpoint.pt" >/dev/null 2>&1; then
        s3_has_ckpt=1
    fi
    if (( s3_has_ckpt == 1 )) && [[ ! -f "${outdir}/checkpoint.pt" ]]; then
        log "FATAL: S3 has checkpoint.pt for ${exp} but local copy missing after sync"
        lease_release "$exp"
        rm -rf "$outdir"
        return 1
    fi

    local resume_flag=()
    if [[ -f "${outdir}/checkpoint.pt" ]]; then
        local sz
        sz=$(stat -c %s "${outdir}/checkpoint.pt" 2>/dev/null || echo 0)
        if (( sz < 1000000 )); then
            log "  WARN: ${outdir}/checkpoint.pt is only ${sz} bytes (corrupted partial); discarding + restarting"
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

    (
        while sleep "$HEARTBEAT_EVERY"; do
            if ! lease_refresh "$exp"; then
                log "  lease lost — stopping heartbeat"
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

    setsid env "${entity_env[@]}" python -m train.medmnist \
        --dataset "$ds" \
        --model "$model" \
        --activation "$act" \
        --seed "$seed" \
        --output_dir "$outdir" \
        --data_dir "$MEDMNIST_DATA_DIR" \
        --amp \
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
start_preempt_watcher

if command -v wandb >/dev/null; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

idle_passes=0
while (( idle_passes < MAX_IDLE_PASSES )); do
    ran_any=0
    for entry in $JOB_ORDER; do
        IFS=: read -r ds model act seed <<<"$entry"
        if [[ -z "$ds" || -z "$model" || -z "$act" || -z "$seed" ]]; then
            log "skipping malformed entry: $entry"
            continue
        fi
        rc=0
        run_job "$ds" "$model" "$act" "$seed" || rc=$?
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
