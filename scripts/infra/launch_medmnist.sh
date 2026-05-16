#!/usr/bin/env bash
# MedMNIST v2 data-scarce campaign launcher.
#
# Same machinery as launch_cifar.sh (watchdog keeps a spot-worker pool
# alive until every <exp>/complete sentinel is on S3), with MedMNIST
# overrides:
#
#   * ENTRY_SCRIPT  — scripts/orchestrate_medmnist.sh (GPU-slot fanout)
#   * JOB_ORDER     — parsed from default_job_order_medmnist.txt
#                     (<dataset>:<model>:<act>:<seed> 4-tuples)
#   * INSTANCE_TYPE — g5.12xlarge (4×A10G); MedMNIST jobs are
#                     single-GPU and tiny, so a 4-GPU VM drains four in
#                     parallel via the slot fanout.
#   * CAMPAIGN      — "medmnist" so the watchdog's live-worker count
#                     doesn't collide with the CIFAR/ImageNet fleets.
#
# The watchdog's _exp_from_entry splits JOB_ORDER on ':' — for a 4-tuple
# <ds>:<model>:<act>:<seed> it would mis-derive the exp basename, so we
# also override JOB_EXP_FMT-awareness by pointing the watchdog at the
# MedMNIST exp scheme. We do this by exporting MEDMNIST=1 and letting the
# watchdog's generic key match the slot's <ds>-<model>-<act>-s<seed>
# layout via the medmnist-aware override below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
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
: "${TARGET_WORKERS:=1}"
: "${INSTANCE_TYPE:=g5.12xlarge}"
export TARGET_WORKERS INSTANCE_TYPE

export ENTRY_SCRIPT="scripts/orchestrate_medmnist.sh"
export CAMPAIGN="medmnist"

# Tell the fanout exactly how many GPU slots to spawn instead of letting
# it guess from nvidia-smi (which mis-counted 1 GPU as 2 on g5.2xlarge).
# Map the known instance types; leave unset for anything else so the
# orchestrator's hardened GPU detector decides. Caller can override with
# NUM_MEDMNIST_SLOTS.
if [[ -z "${NUM_MEDMNIST_SLOTS:-}" ]]; then
    case "$INSTANCE_TYPE" in
        g5.2xlarge|g5.xlarge|g5.4xlarge|g5.8xlarge|g5.16xlarge|\
        g6.2xlarge|g6.xlarge|g6.4xlarge|g4dn.xlarge|g4dn.2xlarge)
            export NUM_MEDMNIST_SLOTS=1 ;;
        g5.12xlarge|g6.12xlarge|g5.24xlarge)
            export NUM_MEDMNIST_SLOTS=4 ;;
        g5.48xlarge|g6.48xlarge|p4d.24xlarge|p5.48xlarge)
            export NUM_MEDMNIST_SLOTS=8 ;;
        *) : ;;  # unknown → orchestrator auto-detects
    esac
fi

if [[ -z "${JOB_ORDER:-}" ]]; then
    JOB_ORDER=$(grep -v '^\s*#' "$SCRIPT_DIR/default_job_order_medmnist.txt" \
        | grep -v '^\s*$' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ //;s/ $//')
    echo "JOB_ORDER not set — parsed default_job_order_medmnist.txt"
fi
export JOB_ORDER

# The stock watchdog's _exp_from_entry derives the exp basename from a
# <cfg>:<act>:<seed> triple. MedMNIST entries are <ds>:<model>:<act>:<seed>
# and the slot writes exp = <ds>-<model>-<act>-s<seed>. Tell the watchdog
# to use the MedMNIST scheme.
export JOB_EXP_SCHEME="medmnist"

: "${WANDB_PROJECT:=medmnist-gate-normalization}"
export WANDB_PROJECT

echo "MedMNIST campaign starting:"
echo "  TARGET_WORKERS = $TARGET_WORKERS"
echo "  INSTANCE_TYPE  = $INSTANCE_TYPE"
echo "  ENTRY_SCRIPT   = $ENTRY_SCRIPT"
echo "  CAMPAIGN_AZS   = $CAMPAIGN_AZS"
echo "  CKPT_BUCKET    = $CKPT_BUCKET"
echo "  WANDB_PROJECT  = $WANDB_PROJECT"
echo "  #jobs          = $(echo "$JOB_ORDER" | wc -w)"
echo

exec bash "$SCRIPT_DIR/watchdog.sh"
