#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Launch a temporary on-demand instance for snapshot preparation,
#  run setup_snapshot.sh via SSM, create an EBS snapshot, update
#  DATA_SNAPSHOT in the local .env, and terminate the instance.
#
#  Usage:
#    ./scripts/infra/launch_snapshot_setup.sh
#    ./scripts/infra/launch_snapshot_setup.sh --launch-only
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/aws_common.sh"

ENV_FILE=""
LAUNCH_ONLY=0
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
        --launch-only)
            LAUNCH_ONLY=1
            shift
            ;;
        --keep-instance)
            KEEP_INSTANCE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--env-file FILE] [--launch-only] [--keep-instance]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

load_env_file "$ENV_FILE" "$REPO_ROOT"
TARGET_ENV_FILE="$(resolve_env_file_path "$ENV_FILE" "$REPO_ROOT")"

REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
DATA_SOURCE_S3="${DATA_SOURCE_S3:-$S3_BUCKET}"
KEY_NAME="${KEY_NAME:-}"
SECURITY_GROUP="${SECURITY_GROUP:-}"
SUBNET_CANDIDATES="$(normalize_subnet_candidates || true)"
IAM_ROLE="${IAM_INSTANCE_PROFILE:-}"
INSTANCE_TYPE="${SNAPSHOT_INSTANCE_TYPE:-p5.48xlarge}"
ROOT_VOL_SIZE="${SNAPSHOT_ROOT_VOLUME_SIZE:-200}"
DATA_VOL_SIZE="${SNAPSHOT_DATA_VOLUME_SIZE:-500}"
INSTANCE_NAME="${SNAPSHOT_SETUP_NAME:-nelu-snapshot-setup}"
SSM_WAIT_TIMEOUT="${SNAPSHOT_SSM_WAIT_TIMEOUT:-900}"
SETUP_WAIT_TIMEOUT="${SNAPSHOT_SETUP_WAIT_TIMEOUT:-14400}"
KEEP_ON_FAILURE="${SNAPSHOT_KEEP_ON_FAILURE:-1}"

if [ -z "$KEY_NAME" ] || [ -z "$SECURITY_GROUP" ] || [ -z "$SUBNET_CANDIDATES" ] || [ -z "$IAM_ROLE" ]; then
    echo "ERROR: KEY_NAME, SECURITY_GROUP, SUBNETS (or SUBNET), and IAM_INSTANCE_PROFILE must be set in .env" >&2
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

AMI_ID="$(resolve_ami "$REGION")"

echo "Using AMI: $AMI_ID"
echo "Subnet candidates: $SUBNET_CANDIDATES"
echo "Uploading snapshot setup bundle to S3..."
upload_repo_bundle "$REPO_ROOT" "${S3_BUCKET}/code/nelu-snapshot-setup.tar.gz"
echo "  Bundle uploaded to ${S3_BUCKET}/code/nelu-snapshot-setup.tar.gz"

INSTANCE_ID=""
LAUNCHED_SUBNET=""
ERR_FILE="$(mktemp)"
for SUBNET in $SUBNET_CANDIDATES; do
    echo "Trying snapshot setup subnet: $SUBNET"
    if INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --subnet-id "$SUBNET" \
        --iam-instance-profile "Arn=$IAM_ROLE" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'"$ROOT_VOL_SIZE"',"VolumeType":"gp3","Iops":10000,"Throughput":500}},{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":'"$DATA_VOL_SIZE"',"VolumeType":"gp3","Iops":10000,"Throughput":500}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Project,Value=nelu},{Key=Role,Value=snapshot-setup}]" \
        --query 'Instances[0].InstanceId' \
        --output text 2>"$ERR_FILE"); then
        LAUNCHED_SUBNET="$SUBNET"
        break
    fi
    sed 's/^/    /' "$ERR_FILE" >&2
done

rm -f "$ERR_FILE"
if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: failed to launch snapshot setup instance in any subnet" >&2
    exit 1
fi

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "Instance ready"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo "  Subnet:      $LAUNCHED_SUBNET"
echo "  AMI:         $AMI_ID"

if [ "$LAUNCH_ONLY" -eq 1 ]; then
    echo ""
    echo "Launch-only mode enabled."
    echo "  Use SSM or SSH to run setup manually:"
    echo "    sudo bash scripts/infra/setup_snapshot.sh"
    exit 0
fi

echo ""
echo "Waiting for SSM agent..."
if ! wait_for_ssm_online "$INSTANCE_ID" "$SSM_WAIT_TIMEOUT"; then
    echo "Instance left running for inspection: $INSTANCE_ID" >&2
    exit 1
fi

COMMANDS_FILE="/tmp/nelu-snapshot-setup-commands-$$.json"
python - "$COMMANDS_FILE" "$S3_BUCKET" "$REGION" "$DATA_SOURCE_S3" <<'PY'
import json
import shlex
import sys

out_path, s3_bucket, region, data_source_s3 = sys.argv[1:5]
bundle_uri = f"{s3_bucket}/code/nelu-snapshot-setup.tar.gz"
workdir = "/opt/nelu-snapshot-setup"
repo_dir = f"{workdir}/repo"

commands = [
    "set -euo pipefail",
    f"mkdir -p {shlex.quote(workdir)}",
    f"aws s3 cp {shlex.quote(bundle_uri)} {shlex.quote(workdir + '/nelu-code.tar.gz')} --region {shlex.quote(region)}",
    f"rm -rf {shlex.quote(repo_dir)}",
    f"mkdir -p {shlex.quote(repo_dir)}",
    f"tar xzf {shlex.quote(workdir + '/nelu-code.tar.gz')} -C {shlex.quote(repo_dir)}",
    f"cd {shlex.quote(repo_dir)}",
    (
        "sudo env "
        f"AWS_REGION={shlex.quote(region)} "
        f"S3_BUCKET={shlex.quote(s3_bucket)} "
        f"DATA_SOURCE_S3={shlex.quote(data_source_s3)} "
        "bash scripts/infra/setup_snapshot.sh"
    ),
]

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump({"commands": commands}, fh)
PY

echo "Running setup_snapshot.sh via SSM..."
COMMAND_ID=$(aws ssm send-command \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "NELU snapshot setup" \
    --parameters "file://${COMMANDS_FILE}" \
    --query 'Command.CommandId' \
    --output text)
rm -f "$COMMANDS_FILE"

if ! wait_for_ssm_command "$COMMAND_ID" "$INSTANCE_ID" "$SETUP_WAIT_TIMEOUT"; then
    echo "ERROR: snapshot setup failed on instance $INSTANCE_ID" >&2
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

SNAP_DESC="${SNAPSHOT_DESCRIPTION:-nelu-training-env-$(date +%Y%m%d)}"
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
