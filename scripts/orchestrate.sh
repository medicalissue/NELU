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

: "${CKPT_BUCKET:?CKPT_BUCKET not set (e.g. s3://nelu-checkpoints)}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set}"
: "${WANDB_PROJECT:=gate-normalization}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=1}"
export AWS_DEFAULT_REGION

# Default job order lives in scripts/infra/default_job_order.txt — one
# "<config>:<activation>" pair per line, '#' comments ignored. Making it
# a standalone file keeps the launcher's awk parsing trivial (grep out
# comments/blanks and flatten) and lets users edit the queue without
# touching shell quoting.
if [[ -z "${JOB_ORDER:-}" ]]; then
    _job_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/infra/default_job_order.txt"
    if [[ -f "$_job_file" ]]; then
        JOB_ORDER=$(grep -v '^\s*#' "$_job_file" | grep -v '^\s*$' | tr '\n' ' ')
    fi
fi
: "${JOB_ORDER:?JOB_ORDER is empty and default_job_order.txt not found}"

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

AZ=$(imds placement/availability-zone 2>/dev/null || echo "unknown")
INSTANCE_ID=$(imds instance-id 2>/dev/null || echo "unknown")
log "AZ=$AZ instance=$INSTANCE_ID"

# ── Data volume sanity check ───────────────────────────────────────
# The dataset volume is mounted by dstack itself (see `volumes:` in the
# task YAML). We only verify it looks right before the job loop starts.
DATA_MOUNT=/data

check_data_mount() {
    if [[ ! -d "$DATA_MOUNT/imagenet/val" ]]; then
        log "FATAL: expected $DATA_MOUNT/imagenet/val — dstack volume misconfigured?"
        log "contents of $DATA_MOUNT:"; ls "$DATA_MOUNT" 2>&1 | head -20 >&2 || true
        exit 1
    fi
    log "data volume OK: $(df -h "$DATA_MOUNT" | tail -1)"
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
    # Optimistic claim with read-back confirmation:
    #   1. Skip if a fresh lease exists.
    #   2. PUT our own (owner, ts).
    #   3. Sleep a small jitter, GET the lease. If another worker PUT
    #      after us that worker wins; we lose and skip. Catches the
    #      concurrent-claim race S3 read-after-write cannot prevent
    #      (both workers see "no lease" before either PUTs).
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
    sleep "$(awk "BEGIN{print 0.5+rand()}")"
    local winner
    read -r winner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    if [[ "$winner" != "$INSTANCE_ID" ]]; then
        log "lost lease race on $exp (winner=$winner, we=$INSTANCE_ID)"
        return 1
    fi
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

    # Skip already-complete. The sentinel lands at one of two S3 paths
    # depending on timm's output layout (nested under $exp/, or flat at
    # the prefix root). Check both to stay robust.
    if s3_exists "${s3_prefix}/${exp}/complete" || s3_exists "${s3_prefix}/complete"; then
        log "skip ${exp} (complete)"
        return 3
    fi

    # Try to claim.
    if ! lease_claim "$exp"; then
        log "skip ${exp} (fresh lease held by another worker)"
        return 2
    fi

    log "▶ running ${exp}"

    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"

    # Pull prior state (resume point + W&B id), if any. Skip the rolling
    # checkpoint-*.pth.tar history — resume only needs last.pth.tar, and
    # S3 accumulates these without a --delete mirror policy.
    aws s3 sync "${s3_prefix}/" "${outdir}/" \
        --exclude "lease" --exclude "complete" --exclude "checkpoint-*.pth.tar" \
        --exact-timestamps --only-show-errors || true

    # Locate the last.pth.tar. timm's CheckpointSaver writes into
    # "${args.output}/${args.experiment}/" so it lives at $outdir/$exp/
    # in practice, but we also probe $outdir/ directly as a safety net
    # against future timm layout changes. find(1) is our truth: if there
    # is exactly one last.pth.tar anywhere under $outdir we use it, no
    # matter where timm chose to put it.
    local last_ckpt=""
    if [[ -f "${outdir}/${exp}/last.pth.tar" ]]; then
        last_ckpt="${outdir}/${exp}/last.pth.tar"
    elif [[ -f "${outdir}/last.pth.tar" ]]; then
        last_ckpt="${outdir}/last.pth.tar"
    else
        # Fallback: anywhere under outdir.
        last_ckpt=$(find "$outdir" -maxdepth 3 -name 'last.pth.tar' 2>/dev/null | head -1)
    fi

    # Defence-in-depth: if S3 prefix is non-empty but we still didn't
    # find last.pth.tar, something is wrong (layout mismatch, download
    # failure). Refuse to silently train from scratch over a live exp.
    local s3_has_ckpt=0
    if aws s3 ls "${s3_prefix}/" --recursive 2>/dev/null | grep -q 'last.pth.tar'; then
        s3_has_ckpt=1
    fi
    if (( s3_has_ckpt == 1 )) && [[ -z "$last_ckpt" ]]; then
        log "FATAL: S3 has last.pth.tar for ${exp} but local copy missing after sync — refusing to restart from scratch"
        log "  s3_prefix=${s3_prefix}  outdir=${outdir}"
        log "  outdir contents:"; ls -lR "$outdir" >&2 || true
        lease_release "$exp"
        rm -rf "$outdir"
        return 1
    fi

    local resume_flag=()
    local ckptdir=""
    if [[ -n "$last_ckpt" ]]; then
        resume_flag=(--resume "$last_ckpt")
        ckptdir="$(dirname "$last_ckpt")"
        log "  resuming from ${last_ckpt}"
    fi

    local wandb_id_flag=()
    # W&B sidecar lives alongside the checkpoint (same dir).
    local sidecar=""
    if [[ -n "$ckptdir" && -f "${ckptdir}/wandb_run_id.json" ]]; then
        sidecar="${ckptdir}/wandb_run_id.json"
    elif [[ -f "${outdir}/wandb_run_id.json" ]]; then
        sidecar="${outdir}/wandb_run_id.json"
    else
        sidecar=$(find "$outdir" -maxdepth 3 -name 'wandb_run_id.json' 2>/dev/null | head -1)
    fi
    if [[ -n "$sidecar" ]]; then
        local saved_id
        saved_id=$(python -c "import json,sys; print(json.load(open('${sidecar}')).get('run_id',''))" 2>/dev/null || true)
        if [[ -n "$saved_id" ]]; then
            wandb_id_flag=(--wandb-resume-id "$saved_id")
            log "  resuming W&B run ${saved_id} (sidecar=${sidecar})"
        fi
    fi

    local entity_flag=()
    if [[ -n "$WANDB_ENTITY" ]]; then
        entity_flag=(--wandb-entity "$WANDB_ENTITY")
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

    # Launch torchrun in its own session so the preempt watcher can
    # SIGTERM the whole tree by pgid (= pid of the setsid leader).
    local gpus
    gpus=$(nvidia-smi -L | wc -l)
    setsid torchrun \
        --nproc_per_node="$gpus" \
        -m train.imagenet \
        --config "$cfg" \
        --activation "$act" \
        --experiment "$exp" \
        --output "$outdir" \
        --log-wandb \
        --wandb-project "$WANDB_PROJECT" \
        "${entity_flag[@]}" \
        "${resume_flag[@]}" \
        "${wandb_id_flag[@]}" &
    local trainer_pid=$!
    # setsid makes the child a session/group leader — its pgid == its pid.
    echo "$trainer_pid" > /tmp/trainer.pgid
    local trainer_rc=0
    wait "$trainer_pid" || trainer_rc=$?

    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f /tmp/trainer.pgid

    # Final sync regardless of exit code — we want whatever progress the
    # trainer managed to checkpoint to be visible to the next worker.
    aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "lease" \
        --only-show-errors || true

    # timm writes into ``<outdir>/<experiment>/`` because ``--experiment
    # $exp`` is forwarded to ``utils.get_outdir``. The completion sentinel
    # lives there, not at the top of $outdir. Probe both so a future timm
    # layout change doesn't silently regress us to "every run is paused".
    local complete_path=""
    if [[ -f "${outdir}/${exp}/complete" ]]; then
        complete_path="${outdir}/${exp}/complete"
    elif [[ -f "${outdir}/complete" ]]; then
        complete_path="${outdir}/complete"
    fi
    if [[ $trainer_rc -eq 0 && -n "$complete_path" ]]; then
        log "✓ ${exp} complete (rc=$trainer_rc)  sentinel=${complete_path}"
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
check_data_mount
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
        # rc=0 means this worker actually ran training (trainer executed).
        # rc=2 = lease held by another worker, rc=3 = already complete.
        # Only rc=0 counts as "real work" — skip-paths must not reset the
        # idle counter, otherwise a worker finding a drained queue loops
        # forever burning spot $.
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
