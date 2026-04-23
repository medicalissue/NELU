#!/usr/bin/env bash
# Bare-EC2 worker bootstrap. Invoked by the VM's user-data at boot.
#
# Pre-conditions (handled by the launch template):
#   * /dev/sdg is an EBS volume cloned from the dataset snapshot (NVMe
#     remaps this to /dev/nvme*n1; we resolve by serial).
#   * IAM instance profile `nelu-worker-profile` is attached.
#   * Ubuntu 22.04 base with NVIDIA driver pre-installed (AWS DL base AMI
#     or the AMI baked from our data instance).
#
# Responsibilities:
#   1. Mount the dataset volume at /data (read-only).
#   2. Fetch the pre-built venv tarball from S3, extract to /opt/nelu-venv.
#   3. Clone/update the project repo at /workspace.
#   4. Activate the venv, install the repo in editable mode (no deps).
#   5. Log everything to /var/log/nelu/bootstrap.log AND the EC2 console
#      (via `exec > /dev/console`) so `aws ec2 get-console-output` can be
#      used when SSH is not reachable.
#   6. hand off to scripts/orchestrate.sh.
#
# Required env (exported by user-data wrapper):
#   VENV_S3_URL       s3://nelu-datasets/env/nelu-venv-py310-cu130.tar.gz
#   REPO_URL          https://github.com/<user>/NELU.git  (or commit sha)
#   REPO_REF          branch/tag/commit
#   CKPT_BUCKET       s3://nelu-checkpoints
#   WANDB_API_KEY     (value)
#   WANDB_PROJECT     gate-normalization
#   WANDB_ENTITY      (optional)
#   AWS_DEFAULT_REGION us-west-2
#   JOB_ORDER         (optional; defaults set in orchestrate.sh)
#
# The script does NOT `set -e`: we want to survive individual failures
# long enough to leave diagnostic breadcrumbs before exiting non-zero so
# the watchdog can relaunch.

LOGDIR=/var/log/nelu
mkdir -p "$LOGDIR"
LOG="$LOGDIR/bootstrap.log"
# Mirror everything to /dev/console so `aws ec2 get-console-output` shows
# progress when the VM is un-SSH-able.
exec > >(tee -a "$LOG" | tee /dev/console) 2>&1

echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting on $(hostname)"

: "${VENV_S3_URL:?VENV_S3_URL is required}"
: "${REPO_URL:?REPO_URL is required}"
: "${REPO_REF:=public-release}"
: "${CKPT_BUCKET:?CKPT_BUCKET is required}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${VENV_ROOT:=/opt/nelu-venv}"
: "${WORKSPACE:=/workspace}"
export AWS_DEFAULT_REGION

die() {
    local msg="$1" rc="${2:-99}"
    echo "[bootstrap] FATAL: $msg (rc=$rc)"
    exit "$rc"
}

step() {
    local name="$1"; shift
    local logfile="${LOGDIR}/${name}.log"
    echo "[bootstrap] ▶ $name"
    if ! "$@" > "$logfile" 2>&1; then
        echo "[bootstrap] ✗ $name failed (rc=$?). Last 40 lines of $logfile:"
        tail -40 "$logfile"
        return 1
    fi
    echo "[bootstrap] ✓ $name ok"
}

# ── 1. Mount /data ────────────────────────────────────────────────
mount_data() {
    if mountpoint -q /data; then
        echo "[bootstrap] /data already mounted"
        return 0
    fi
    # Launch template attached /dev/sdg — NVMe remaps; resolve by volume-id.
    local imds_token token_flag vol_id
    token_flag=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
                 -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    # The volume-id for the sdg mapping comes from block-device-mapping metadata.
    vol_id=$(curl -sS -H "X-aws-ec2-metadata-token: $token_flag" \
             "http://169.254.169.254/latest/meta-data/block-device-mapping/ebs1" 2>/dev/null)
    if [[ -z "$vol_id" ]]; then
        vol_id=$(curl -sS -H "X-aws-ec2-metadata-token: $token_flag" \
                 "http://169.254.169.254/latest/meta-data/block-device-mapping/ebs2" 2>/dev/null)
    fi
    # Fallback: enumerate all nvme disks, skip root (has partitions), pick the 500GB one.
    local dev=""
    local serial_nodash="${vol_id//-/}"
    local serial_dash="$vol_id"
    for _ in {1..30}; do
        if [[ -n "$vol_id" ]]; then
            dev=$(lsblk -dno NAME,SERIAL | awk -v a="$serial_nodash" -v b="$serial_dash" \
                  '$2==a || $2==b {print "/dev/"$1; exit}')
        fi
        if [[ -z "$dev" ]]; then
            # Heuristic: any nvme disk that is 500GB-class and not the root
            dev=$(lsblk -dno NAME,SIZE,TYPE | awk '$2=="500G" && $3=="disk" {print "/dev/"$1; exit}')
        fi
        [[ -n "$dev" ]] && break
        sleep 1
    done
    [[ -z "$dev" ]] && die "could not resolve /data device" 2
    echo "[bootstrap] mounting $dev at /data (ro)"
    mkdir -p /data
    mount -o ro "$dev" /data || die "mount $dev /data failed" 3
    df -h /data
}
mount_data

# ── 2. Fetch venv ─────────────────────────────────────────────────
if [[ -x "${VENV_ROOT}/bin/python" ]]; then
    echo "[bootstrap] reusing existing ${VENV_ROOT}"
else
    step fetch-venv aws s3 cp "$VENV_S3_URL" /tmp/nelu-venv.tar.gz \
        || die "fetch-venv failed" 4
    step extract-venv bash -c "mkdir -p /opt && tar xzf /tmp/nelu-venv.tar.gz -C /opt && rm -f /tmp/nelu-venv.tar.gz" \
        || die "extract-venv failed" 5
fi

# ── 3. Clone repo ─────────────────────────────────────────────────
if [[ -d "$WORKSPACE/.git" ]]; then
    step repo-update bash -c "cd $WORKSPACE && git fetch --all -q && git checkout $REPO_REF -q && git pull -q" \
        || die "repo-update failed" 6
else
    step repo-clone git clone -q --branch "$REPO_REF" "$REPO_URL" "$WORKSPACE" \
        || die "repo-clone failed" 6
fi

# ── 4. Editable install ──────────────────────────────────────────
# shellcheck disable=SC1090
source "${VENV_ROOT}/bin/activate"
step editable-install python -m pip install -e "$WORKSPACE" --no-deps -q \
    || die "editable-install failed" 7

# ── 5. Orchestrate.sh ────────────────────────────────────────────
# ENTRY_SCRIPT selects which orchestrator runs after bootstrap. Default
# is the training worker; eval VMs set ENTRY_SCRIPT=scripts/eval_orchestrate.sh
# via user-data to reuse this same bootstrap machinery for robustness eval.
: "${ENTRY_SCRIPT:=scripts/orchestrate.sh}"
echo "[bootstrap] handing off to ${ENTRY_SCRIPT}"
cd "$WORKSPACE"
bash "$ENTRY_SCRIPT"
ORC_RC=$?
echo "[bootstrap] ${ENTRY_SCRIPT} exited rc=$ORC_RC"

# ── 6. Self-terminate ───────────────────────────────────────────
# orchestrate.sh returns when the queue is drained or the worker has
# nothing left to do. Without this step the spot VM would sit idle,
# billing us $14–21/h and confusing the watchdog's live-worker count.
# Primary path: ec2:TerminateInstances via IAM. Fallback: OS-level halt
# (scheduled in 2 min so logs flush first) — AWS reaps halted spot VMs
# shortly after. If both fail, the watchdog's effective_target cap
# prevents launch/terminate storms, but the idle VM still bills until
# the campaign ends.
echo "[bootstrap] self-terminating spot instance"
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
IID=$(curl -sSH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
terminated=0
if [[ -n "$IID" ]]; then
    if aws ec2 terminate-instances --instance-ids "$IID" \
            --region "${AWS_DEFAULT_REGION:-us-west-2}" \
            >>"$LOG" 2>&1; then
        echo "[bootstrap] terminate-instances OK for $IID"
        terminated=1
    else
        echo "[bootstrap] terminate-instances FAILED — scheduling OS halt"
    fi
fi
if (( terminated == 0 )); then
    # Last-resort: halt in 2 min so bootstrap log + final S3 sync finish.
    # `shutdown -h +2` schedules halt; AWS then cleans up the spot VM.
    shutdown -h +2 "nelu worker self-terminate fallback" >/dev/null 2>&1 || \
        (sleep 120 && halt -p) &
fi
exit $ORC_RC
