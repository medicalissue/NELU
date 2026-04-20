#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Resume snapshot preparation on an existing instance, finish
#  dataset sync via SSM, create an EBS snapshot, update
#  DATA_SNAPSHOT in the local .env, and optionally terminate the
#  instance.
#
#  Usage:
#    ./scripts/infra/resume_snapshot_setup.sh --instance-id i-xxxx
#    ./scripts/infra/resume_snapshot_setup.sh --instance-id i-xxxx --keep-instance
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/aws_common.sh"

ENV_FILE=""
INSTANCE_ID=""
KEEP_INSTANCE=0
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
        --instance-id)
            if [ $# -lt 2 ]; then
                echo "ERROR: --instance-id requires a value" >&2
                exit 1
            fi
            INSTANCE_ID="$2"
            shift 2
            ;;
        --keep-instance)
            KEEP_INSTANCE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --instance-id INSTANCE_ID [--env-file FILE] [--keep-instance]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: --instance-id is required" >&2
    exit 1
fi

load_env_file "$ENV_FILE" "$REPO_ROOT"
TARGET_ENV_FILE="$(resolve_env_file_path "$ENV_FILE" "$REPO_ROOT")"

REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
DATA_SOURCE_S3="${DATA_SOURCE_S3:-$S3_BUCKET}"
SSM_WAIT_TIMEOUT="${SNAPSHOT_SSM_WAIT_TIMEOUT:-900}"
SETUP_WAIT_TIMEOUT="${SNAPSHOT_SETUP_WAIT_TIMEOUT:-14400}"
KEEP_ON_FAILURE="${SNAPSHOT_KEEP_ON_FAILURE:-1}"
SNAP_DESC="${SNAPSHOT_DESCRIPTION:-nelu-training-env-$(date +%Y%m%d)}"
SSM_COMMAND_MAX_TIMEOUT=172800

case "$SETUP_WAIT_TIMEOUT" in
    ''|*[!0-9]*)
        echo "ERROR: SNAPSHOT_SETUP_WAIT_TIMEOUT must be a positive integer number of seconds" >&2
        exit 1
        ;;
esac

if [ "$SETUP_WAIT_TIMEOUT" -le 0 ] || [ "$SETUP_WAIT_TIMEOUT" -gt "$SSM_COMMAND_MAX_TIMEOUT" ]; then
    echo "ERROR: SNAPSHOT_SETUP_WAIT_TIMEOUT must be between 1 and ${SSM_COMMAND_MAX_TIMEOUT} seconds" >&2
    exit 1
fi

wait_for_ssm_online() {
    local instance_id="$1"
    local timeout="$2"
    local start_ts
    local ping_status

    start_ts=$(date +%s)
    while true; do
        ping_status=$(aws ssm describe-instance-information \
            --region "$REGION" \
            --filters "Key=InstanceIds,Values=${instance_id}" \
            --query 'InstanceInformationList[0].PingStatus' \
            --output text 2>/dev/null || true)
        if [ "$ping_status" = "Online" ]; then
            return 0
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "$timeout" ]; then
            echo "ERROR: instance $instance_id did not come online in SSM within ${timeout}s" >&2
            return 1
        fi
        sleep 10
    done
}

wait_for_ssm_command() {
    local command_id="$1"
    local instance_id="$2"
    local timeout="$3"
    local start_ts
    local status

    start_ts=$(date +%s)
    while true; do
        status=$(aws ssm get-command-invocation \
            --region "$REGION" \
            --command-id "$command_id" \
            --instance-id "$instance_id" \
            --query 'Status' \
            --output text 2>/dev/null || true)
        case "$status" in
            Success)
                return 0
                ;;
            Failed|TimedOut|Cancelled|Cancelling)
                return 1
                ;;
            Pending|InProgress|Delayed|"")
                ;;
            *)
                ;;
        esac
        if [ $(( $(date +%s) - start_ts )) -ge "$timeout" ]; then
            echo "ERROR: SSM command $command_id timed out after ${timeout}s" >&2
            return 1
        fi
        sleep 15
    done
}

dump_ssm_command_logs() {
    local command_id="$1"
    local instance_id="$2"

    echo "SSM stdout:"
    aws ssm get-command-invocation \
        --region "$REGION" \
        --command-id "$command_id" \
        --instance-id "$instance_id" \
        --query 'StandardOutputContent' \
        --output text 2>/dev/null || true
    echo ""
    echo "SSM stderr:"
    aws ssm get-command-invocation \
        --region "$REGION" \
        --command-id "$command_id" \
        --instance-id "$instance_id" \
        --query 'StandardErrorContent' \
        --output text 2>/dev/null || true
}

INSTANCE_STATE=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || true)

if [ -z "$INSTANCE_STATE" ] || [ "$INSTANCE_STATE" = "None" ]; then
    echo "ERROR: instance not found: $INSTANCE_ID" >&2
    exit 1
fi

if [ "$INSTANCE_STATE" != "running" ]; then
    echo "ERROR: instance $INSTANCE_ID is not running (state: $INSTANCE_STATE)" >&2
    exit 1
fi

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text 2>/dev/null || true)

echo "Resuming snapshot setup on existing instance"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo "  Region:      $REGION"

echo ""
echo "Uploading refreshed snapshot setup bundle to S3..."
upload_repo_bundle "$REPO_ROOT" "${S3_BUCKET}/code/nelu-snapshot-setup.tar.gz"
echo "  Bundle uploaded to ${S3_BUCKET}/code/nelu-snapshot-setup.tar.gz"

echo ""
echo "Waiting for SSM agent..."
if ! wait_for_ssm_online "$INSTANCE_ID" "$SSM_WAIT_TIMEOUT"; then
    echo "Instance left running for inspection: $INSTANCE_ID" >&2
    exit 1
fi

COMMANDS_FILE="/tmp/nelu-snapshot-resume-commands-$$.json"
python - "$COMMANDS_FILE" "$S3_BUCKET" "$REGION" "$DATA_SOURCE_S3" "$SETUP_WAIT_TIMEOUT" <<'PY'
import json
import shlex
import sys

out_path, s3_bucket, region, data_source_s3, setup_wait_timeout = sys.argv[1:6]
bundle_uri = f"{s3_bucket}/code/nelu-snapshot-setup.tar.gz"
workdir = "/opt/nelu-snapshot-setup"
repo_dir = f"{workdir}/repo"

resume_script = """set -euo pipefail
mkdir -p /data/imagenet
if command -v s5cmd >/dev/null 2>&1; then
  s5cmd sync "${DATA_SOURCE_S3%/}/imagenet/*" /data/imagenet/
else
  aws s3 sync "${DATA_SOURCE_S3%/}/imagenet/" /data/imagenet/ --no-progress
fi
cd /opt/nelu-snapshot-setup/repo
bash scripts/download_data.sh /data
"""

commands = [
    "set -euo pipefail",
    f"mkdir -p {shlex.quote(workdir)}",
    f"aws s3 cp {shlex.quote(bundle_uri)} {shlex.quote(workdir + '/nelu-code.tar.gz')} --region {shlex.quote(region)}",
    f"rm -rf {shlex.quote(repo_dir)}",
    f"mkdir -p {shlex.quote(repo_dir)}",
    f"tar xzf {shlex.quote(workdir + '/nelu-code.tar.gz')} -C {shlex.quote(repo_dir)}",
    "cat >/tmp/nelu-resume.sh <<'EOF'\n" + resume_script + "\nEOF",
    "chmod +x /tmp/nelu-resume.sh",
    (
        "sudo env "
        f"AWS_REGION={shlex.quote(region)} "
        f"S3_BUCKET={shlex.quote(s3_bucket)} "
        f"DATA_SOURCE_S3={shlex.quote(data_source_s3)} "
        "bash /tmp/nelu-resume.sh"
    ),
]

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump({
        "commands": commands,
        "executionTimeout": [setup_wait_timeout],
    }, fh)
PY

echo "Resuming dataset sync via SSM..."
COMMAND_ID=$(aws ssm send-command \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "NELU snapshot resume" \
    --parameters "file://${COMMANDS_FILE}" \
    --query 'Command.CommandId' \
    --output text)
rm -f "$COMMANDS_FILE"

echo "  SSM Command ID: $COMMAND_ID"
echo "  SSM execution timeout: ${SETUP_WAIT_TIMEOUT}s"

if ! wait_for_ssm_command "$COMMAND_ID" "$INSTANCE_ID" "$SETUP_WAIT_TIMEOUT"; then
    echo "ERROR: snapshot resume failed on instance $INSTANCE_ID" >&2
    dump_ssm_command_logs "$COMMAND_ID" "$INSTANCE_ID"
    if [ "$KEEP_ON_FAILURE" -eq 1 ]; then
        echo "Instance left running for inspection: $INSTANCE_ID" >&2
    else
        aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
    fi
    exit 1
fi

VOL_ID=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName==`/dev/sdf`].Ebs.VolumeId' \
    --output text)

if [ -z "$VOL_ID" ] || [ "$VOL_ID" = "None" ]; then
    echo "ERROR: failed to resolve /dev/sdf volume for instance $INSTANCE_ID" >&2
    echo "Instance left running for inspection: $INSTANCE_ID" >&2
    exit 1
fi

echo "Creating snapshot from volume $VOL_ID..."
SNAP_ID=$(aws ec2 create-snapshot \
    --region "$REGION" \
    --volume-id "$VOL_ID" \
    --description "$SNAP_DESC" \
    --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=${SNAP_DESC}},{Key=Project,Value=nelu}]" \
    --query 'SnapshotId' \
    --output text)

aws ec2 wait --region "$REGION" snapshot-completed --snapshot-ids "$SNAP_ID"
set_env_value "$TARGET_ENV_FILE" "DATA_SNAPSHOT" "$SNAP_ID"

echo "Snapshot ready: $SNAP_ID"
echo "Updated ${TARGET_ENV_FILE} with DATA_SNAPSHOT=$SNAP_ID"

if [ "$KEEP_INSTANCE" -eq 1 ]; then
    echo "Keeping snapshot setup instance running: $INSTANCE_ID"
else
    echo "Terminating snapshot setup instance..."
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
fi

echo ""
echo "Done."
echo "  DATA_SNAPSHOT=$SNAP_ID"
echo "  launch_spot.sh will now use the new snapshot automatically."
