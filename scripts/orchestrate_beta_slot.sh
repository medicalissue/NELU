#!/usr/bin/env bash
# β-pipeline CIFAR slot runner — mirrors orchestrate_cifar_slot.sh but
# accepts a 4-tuple <cfg>:<act>:<seed>:<mode> where mode ∈ {cls, ae}.
#
# cls jobs run train.cifar exactly like the standard CIFAR campaign.
# ae jobs run train.cifar_ae and depend on the matching cls job's
# `complete` sentinel — they're skipped (returned to the queue) until
# the cls run finishes and uploads its ckpt.
#
# Queue layout under $CKPT_BUCKET :
#   <exp>/                  cls run    (exp = <model>-<act>-s<seed>)
#       complete, lease, checkpoint.pt
#   <exp>-ae/               ae run     (exp-ae = <model>-<act>-s<seed>-ae)
#       complete, lease, checkpoint.pt, ae_result.json
set -euo pipefail

: "${GATE_NORM_GPU_SLOT:?GATE_NORM_GPU_SLOT must be set by the fanout script}"
SLOT="$GATE_NORM_GPU_SLOT"

log() { printf '[slot-%s %s] %s\n' "$SLOT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

: "${CKPT_BUCKET:?CKPT_BUCKET not set}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${WANDB_PROJECT:=beta-adaptive-nelu}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=3}"
export AWS_DEFAULT_REGION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${JOB_ORDER:-}" ]]; then
    _job_file="$REPO_ROOT/scripts/infra/default_job_order_beta.txt"
    if [[ -f "$_job_file" ]]; then
        JOB_ORDER=$(grep -v '^\s*#' "$_job_file" | grep -v '^\s*$' | tr '\n' ' ')
    fi
fi
: "${JOB_ORDER:?JOB_ORDER empty and default_job_order_beta.txt not found}"

# ── IMDSv2 ─────────────────────────────────────────────────────────
imds() {
    local token
    token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    curl -sS -H "X-aws-ec2-metadata-token: $token" \
         "http://169.254.169.254/latest/meta-data/$1"
}
INSTANCE_ID=$(imds instance-id 2>/dev/null || echo "unknown")
OWNER="${INSTANCE_ID}-g${SLOT}"
PGID_FILE="/tmp/trainer-g${SLOT}.pgid"
log "owner=$OWNER"

DATA_MOUNT=/data
check_data_mount() {
    local cifar_dir="$DATA_MOUNT/cifar-100-python"
    if [[ ! -f "$cifar_dir/meta" ]]; then
        log "FATAL: $cifar_dir missing"; exit 1
    fi
}

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

# ── Queue plumbing ────────────────────────────────────────────────

s3_exists() { aws s3 ls "$1" >/dev/null 2>&1; }

base_from_cfg() { basename "${1%.yaml}"; }

exp_key() {
    # mode=cls → <model>-<act>-s<seed>
    # mode=ae  → <model>-<act>-s<seed>-ae
    local cfg="$1" act="$2" seed="$3" mode="$4"
    local base; base=$(base_from_cfg "$cfg")
    local key="${base}-${act}-s${seed}"
    [[ "$mode" == "ae" ]] && key="${key}-ae"
    echo "$key"
}

cls_exp_key() {
    local cfg="$1" act="$2" seed="$3"
    local base; base=$(base_from_cfg "$cfg")
    echo "${base}-${act}-s${seed}"
}

lease_claim() {
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now=$(date +%s)
    if s3_exists "$key"; then
        local owner ts age
        read -r owner ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
        age=$((now - ts))
        if (( age < LEASE_TTL )); then return 1; fi
        log "stealing stale lease on $exp (age=${age}s)"
    fi
    echo "$OWNER $now" | aws s3 cp - "$key" >/dev/null
    sleep "$(awk "BEGIN{print 0.5+rand()}")"
    local winner
    read -r winner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    [[ "$winner" != "$OWNER" ]] && return 1
    return 0
}

lease_refresh() {
    local exp="$1" key="${CKPT_BUCKET}/$1/lease" now=$(date +%s) owner
    read -r owner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    [[ "$owner" != "$OWNER" ]] && return 1
    echo "$OWNER $now" | aws s3 cp - "$key" >/dev/null
    return 0
}

lease_release() {
    aws s3 rm "${CKPT_BUCKET}/$1/lease" >/dev/null 2>&1 || true
}

run_cls_job() {
    local cfg="$1" act="$2" seed="$3" exp="$4"
    local s3_prefix="${CKPT_BUCKET}/${exp}"
    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"
    aws s3 sync "${s3_prefix}/" "${outdir}/" \
        --exclude "lease" --exclude "complete" \
        --exact-timestamps --only-show-errors || true

    local resume_flag=()
    if [[ -f "${outdir}/checkpoint.pt" ]]; then
        local sz; sz=$(stat -c %s "${outdir}/checkpoint.pt" 2>/dev/null || echo 0)
        if (( sz < 1000000 )); then
            log "  WARN: partial ckpt ${sz}B; discarding"
            rm -f "${outdir}/checkpoint.pt"
            aws s3 rm "${s3_prefix}/checkpoint.pt" --only-show-errors || true
        else
            resume_flag=(--resume "${outdir}/checkpoint.pt")
            log "  resuming from ckpt (${sz}B)"
        fi
    fi

    (
        while sleep "$HEARTBEAT_EVERY"; do
            if ! lease_refresh "$exp"; then
                log "  lease lost — stopping heartbeat"
                if [[ -f "$PGID_FILE" ]]; then
                    pgid=$(cat "$PGID_FILE"); kill -TERM -"$pgid" 2>/dev/null || true
                fi
                exit 0
            fi
            aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "*.tmp" --exclude "lease" --only-show-errors || true
        done
    ) &
    local heartbeat_pid=$!

    setsid python -m train.cifar \
        --config "$cfg" --activation "$act" --seed "$seed" \
        --output_dir "$outdir" \
        --wandb --wandb_project "$WANDB_PROJECT" \
        "${resume_flag[@]}" &
    local trainer_pid=$!
    echo "$trainer_pid" > "$PGID_FILE"
    local rc=0; wait "$trainer_pid" || rc=$?
    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f "$PGID_FILE"

    aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "lease" --only-show-errors || true
    rm -rf "$outdir"
    return $rc
}

run_ae_job() {
    local cfg="$1" act="$2" seed="$3" exp="$4"
    local s3_prefix="${CKPT_BUCKET}/${exp}"
    local cls_exp; cls_exp=$(cls_exp_key "$cfg" "$act" "$seed")
    local cls_s3="${CKPT_BUCKET}/${cls_exp}"

    # Sanity: cls must be complete before AE
    if ! s3_exists "${cls_s3}/complete"; then
        log "  cls dependency ${cls_exp}/complete not found"
        return 4
    fi

    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"
    # Pull AE-side ckpt (if resumed previously)
    aws s3 sync "${s3_prefix}/" "${outdir}/" \
        --exclude "lease" --exclude "complete" \
        --exact-timestamps --only-show-errors || true
    # Pull the cls ckpt to seed AE
    aws s3 cp "${cls_s3}/checkpoint.pt" "${outdir}/cls_checkpoint.pt" --only-show-errors || {
        log "  failed to fetch cls ckpt"; rm -rf "$outdir"; return 1
    }

    local resume_flag=()
    if [[ -f "${outdir}/checkpoint.pt" ]]; then
        local sz; sz=$(stat -c %s "${outdir}/checkpoint.pt" 2>/dev/null || echo 0)
        if (( sz < 100000 )); then
            log "  WARN: AE partial ckpt ${sz}B; discarding"
            rm -f "${outdir}/checkpoint.pt"
            aws s3 rm "${s3_prefix}/checkpoint.pt" --only-show-errors || true
        else
            resume_flag=(--resume "${outdir}/checkpoint.pt")
            log "  resuming AE from ckpt (${sz}B)"
        fi
    fi

    (
        while sleep "$HEARTBEAT_EVERY"; do
            if ! lease_refresh "$exp"; then
                log "  lease lost — stopping heartbeat"
                if [[ -f "$PGID_FILE" ]]; then
                    pgid=$(cat "$PGID_FILE"); kill -TERM -"$pgid" 2>/dev/null || true
                fi
                exit 0
            fi
            aws s3 sync "${outdir}/" "${s3_prefix}/" \
                --exclude "*.tmp" --exclude "lease" --exclude "cls_checkpoint.pt" \
                --only-show-errors || true
        done
    ) &
    local heartbeat_pid=$!

    setsid python -m train.cifar_ae \
        --config "$cfg" --activation "$act" --seed "$seed" \
        --output_dir "$outdir" \
        --cls_ckpt "${outdir}/cls_checkpoint.pt" \
        --data_root /data \
        --ae_mode full \
        --wandb --wandb_project "$WANDB_PROJECT" \
        "${resume_flag[@]}" &
    local trainer_pid=$!
    echo "$trainer_pid" > "$PGID_FILE"
    local rc=0; wait "$trainer_pid" || rc=$?
    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f "$PGID_FILE"

    aws s3 sync "${outdir}/" "${s3_prefix}/" \
        --exclude "lease" --exclude "cls_checkpoint.pt" --only-show-errors || true
    rm -rf "$outdir"
    return $rc
}

run_job() {
    local cfg="$1" act="$2" seed="$3" mode="$4"
    [[ -z "$mode" ]] && mode="cls"
    local exp; exp=$(exp_key "$cfg" "$act" "$seed" "$mode")
    local s3_prefix="${CKPT_BUCKET}/${exp}"

    if s3_exists "${s3_prefix}/complete"; then
        log "skip ${exp} (complete)"
        return 3
    fi
    if ! lease_claim "$exp"; then
        log "skip ${exp} (fresh lease held)"
        return 2
    fi

    log "▶ running ${exp} (mode=$mode)"
    local rc=0
    case "$mode" in
        cls) run_cls_job "$cfg" "$act" "$seed" "$exp" || rc=$? ;;
        ae)  run_ae_job  "$cfg" "$act" "$seed" "$exp" || rc=$? ;;
        *)   log "unknown mode '$mode'"; rc=99 ;;
    esac

    if [[ $rc -eq 0 && -s "/tmp/runs/${exp}/complete" ]] || \
       aws s3 ls "${s3_prefix}/complete" >/dev/null 2>&1; then
        log "✓ ${exp} complete (rc=$rc)"
    elif [[ $rc -eq 4 ]]; then
        log "⏳ ${exp} cls dep not ready"
    elif [[ $rc -eq 0 ]]; then
        log "⏸ ${exp} paused (clean exit, no sentinel)"
    else
        log "✗ ${exp} failed (rc=$rc)"
    fi
    lease_release "$exp"
    return 0
}

# ── Main ───────────────────────────────────────────────────────────
check_data_mount
start_preempt_watcher
wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true

idle_passes=0
while (( idle_passes < MAX_IDLE_PASSES )); do
    ran_any=0
    for entry in $JOB_ORDER; do
        IFS=: read -r cfg act seed mode <<<"$entry"
        if [[ -z "$cfg" || -z "$act" || -z "$seed" ]]; then
            log "skipping malformed entry: $entry"; continue
        fi
        rc=0; run_job "$cfg" "$act" "$seed" "$mode" || rc=$?
        [[ $rc -eq 0 ]] && ran_any=1
    done
    if (( ran_any == 0 )); then
        idle_passes=$((idle_passes + 1))
        log "no work in this pass (${idle_passes}/${MAX_IDLE_PASSES})"
        sleep 30
    else
        idle_passes=0
    fi
done
log "queue drained — slot exiting"
