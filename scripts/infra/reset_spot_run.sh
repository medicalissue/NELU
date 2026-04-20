#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Reset all NELU spot-run state so a fresh launch starts cleanly.
#
#  This script:
#    1. Stops local launch/monitor processes on this machine
#    2. Terminates NELU spot instances in the configured region
#    3. Removes S3 run state (results/orchestrator/jobs/logs)
#    4. Clears local tracker/results/logs directories
#
#  Usage:
#    ./scripts/infra/reset_spot_run.sh
#    ./scripts/infra/reset_spot_run.sh --env-file .env
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/aws_common.sh"

ENV_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --env-file)
            if [ $# -lt 2 ]; then
                echo "ERROR: --env-file requires a path" >&2
                exit 1
            fi
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--env-file FILE]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

load_env_file "$ENV_FILE" "$REPO_ROOT"

REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
TRACKER_FILE="${INSTANCE_IDS_FILE:-${REPO_ROOT}/.nelu_instance_ids.txt}"

echo "═══════════════════════════════════════════════════════════"
echo "  Reset NELU Spot Run"
echo "═══════════════════════════════════════════════════════════"
echo "  Region:        $REGION"
echo "  S3 bucket:     $S3_BUCKET"
echo "  Tracker file:  $TRACKER_FILE"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "Stopping local orchestrators on this machine..."
pkill -f 'scripts/infra/launch_spot.sh' 2>/dev/null || true
pkill -f 'scripts/infra/monitor_spot.sh' 2>/dev/null || true

echo "Discovering active NELU spot instances..."
INSTANCE_IDS=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters \
        Name=instance-lifecycle,Values=spot \
        Name=tag:Project,Values=nelu \
        Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text)

SPOT_REQUEST_IDS=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters \
        Name=instance-lifecycle,Values=spot \
        Name=tag:Project,Values=nelu \
        Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
    --query 'Reservations[].Instances[].SpotInstanceRequestId' \
    --output text)

if [ -n "${INSTANCE_IDS:-}" ] && [ "${INSTANCE_IDS}" != "None" ]; then
    echo "Terminating spot instances..."
    aws ec2 terminate-instances \
        --region "$REGION" \
        --instance-ids $INSTANCE_IDS \
        >/dev/null
else
    echo "No active NELU spot instances found."
fi

if [ -n "${SPOT_REQUEST_IDS:-}" ] && [ "${SPOT_REQUEST_IDS}" != "None" ]; then
    echo "Cancelling associated spot requests..."
    aws ec2 cancel-spot-instance-requests \
        --region "$REGION" \
        --spot-instance-request-ids $SPOT_REQUEST_IDS \
        >/dev/null || true
fi

echo "Removing S3 run state..."
aws s3 rm "${S3_BUCKET}/results/" --recursive >/dev/null
aws s3 rm "${S3_BUCKET}/orchestrator/" --recursive >/dev/null
aws s3 rm "${S3_BUCKET}/jobs/" --recursive >/dev/null
aws s3 rm "${S3_BUCKET}/logs/" --recursive >/dev/null

echo "Clearing local run state..."
rm -f "$TRACKER_FILE"
rm -rf "${REPO_ROOT}/results" "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/results" "${REPO_ROOT}/logs"

echo ""
echo "Reset complete."
echo "  Note: if another machine is still running launch_spot.sh or monitor_spot.sh,"
echo "  it can relaunch nodes. Stop those processes there before starting fresh."
