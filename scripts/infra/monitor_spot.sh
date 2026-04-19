#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Monitor spot instances and re-launch if terminated.
#  Runs on the LOCAL machine (Mac Mini), not on the EC2 instance.
#
#  Usage: ./scripts/infra/monitor_spot.sh <instance_ids_file>
#
#  The instance_ids_file is created by launch_spot.sh and contains
#  one line per node: INSTANCE_ID NODE_ID JOB_FILE
#
#  The script polls every 60 seconds. When a terminated instance
#  is detected, it re-launches a new spot instance with the same
#  job file. Exits when all jobs show DONE markers on S3.
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
            echo "Usage: $0 [--env-file FILE] <instance_ids_file>"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

load_env_file "$ENV_FILE" "$REPO_ROOT"

POLL_INTERVAL="${POLL_INTERVAL:-60}"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
REGION="${AWS_REGION:-us-east-1}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <instance_ids_file>"
    echo ""
    echo "  Monitors spot instances and re-launches terminated ones."
    echo "  The instance_ids_file is created by launch_spot.sh."
    echo "  Reads settings from ${REPO_ROOT}/.env when present."
    exit 1
fi

IDS_FILE="$1"

if [ ! -f "$IDS_FILE" ]; then
    echo "ERROR: Instance IDs file not found: $IDS_FILE"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  NELU Spot Instance Monitor"
echo "═══════════════════════════════════════════════════════════"
echo "  Instance file: $IDS_FILE"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "  S3 bucket:     $S3_BUCKET"
echo "═══════════════════════════════════════════════════════════"
echo ""

log_err() {
    echo "$*" >&2
}

relaunch_instance() {
    local NODE_ID="$1"
    local JOB_FILE="$2"

    log_err "[$(date -u '+%H:%M:%S')] Re-launching node $NODE_ID..."

    local WANDB_KEY="${WANDB_API_KEY:-}"
    local KEY_NAME="${KEY_NAME:-nelu-training}"
    local SECURITY_GROUP="${SECURITY_GROUP:-sg-CHANGEME}"
    local SUBNET="${SUBNET:-subnet-CHANGEME}"
    local IAM_ROLE="${IAM_INSTANCE_PROFILE:-}"
    local INSTANCE_TYPE="${INSTANCE_TYPE:-p5.48xlarge}"
    local MAX_SPOT_PRICE="${MAX_SPOT_PRICE:-30.00}"
    local DATA_SNAPSHOT="${DATA_SNAPSHOT:-}"
    local USER_DATA_FILE
    local AMI_ID
    local NEW_ID

    if [ "$SECURITY_GROUP" = "sg-CHANGEME" ] || [ "$SUBNET" = "subnet-CHANGEME" ] || [ -z "$IAM_ROLE" ]; then
        log_err "ERROR: SECURITY_GROUP, SUBNET, and IAM_INSTANCE_PROFILE must be set for re-launch"
        return 1
    fi
    if [ -z "$DATA_SNAPSHOT" ]; then
        log_err "ERROR: DATA_SNAPSHOT must be set for re-launch (EBS snapshot with training env)"
        return 1
    fi

    AMI_ID="$(resolve_ami "$REGION")" || return 1
    USER_DATA_FILE="$(render_user_data_file "$SCRIPT_DIR/user_data.sh" "$S3_BUCKET" "$NODE_ID" "$WANDB_KEY")"

    if ! NEW_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --subnet-id "$SUBNET" \
        --iam-instance-profile "Arn=$IAM_ROLE" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Iops":10000,"Throughput":500}},{"DeviceName":"/dev/sdf","Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":500,"VolumeType":"gp3","Iops":10000,"Throughput":500,"DeleteOnTermination":true}}]' \
        --user-data "file://${USER_DATA_FILE}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nelu-node-${NODE_ID}},{Key=Project,Value=nelu}]" \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' \
        --output text); then
        rm -f "$USER_DATA_FILE"
        log_err "[$(date -u '+%H:%M:%S')] Re-launch failed for node $NODE_ID"
        return 1
    fi
    rm -f "$USER_DATA_FILE"

    log_err "[$(date -u '+%H:%M:%S')] New instance for node $NODE_ID: $NEW_ID"
    printf '%s\n' "$NEW_ID"
}

# Main monitoring loop
while true; do
    ALL_DONE=true
    TEMP_FILE=$(mktemp)

    while IFS= read -r line; do
        # Skip comments and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && { echo "$line" >> "$TEMP_FILE"; continue; }
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue

        INST_ID=$(echo "$line" | awk '{print $1}')
        NODE_ID=$(echo "$line" | awk '{print $2}')
        JOB_FILE=$(echo "$line" | awk '{print $3}')

        # Check if all jobs for this node are done on S3
        # Read the job file and check each job's DONE marker
        NODE_DONE=true
        if [ -f "$JOB_FILE" ]; then
            while IFS= read -r job_line; do
                [[ "$job_line" =~ ^[[:space:]]*# ]] && continue
                [[ "$job_line" =~ ^[[:space:]]*$ ]] && continue

                # Parse job line to build RUN_NAME
                # shellcheck disable=SC2086
                set -- $job_line
                J_PHASE="$1"; J_MODEL="$2"; J_ACT="$3"
                shift 3
                J_RUN="${J_PHASE}_${J_MODEL}_${J_ACT}"
                for jarg in "$@"; do
                    jclean=$(echo "$jarg" | sed 's/^--//; s/=/_/g')
                    J_RUN="${J_RUN}_${jclean}"
                done

                if ! aws s3 ls "${S3_BUCKET}/results/${J_RUN}/DONE" >/dev/null 2>&1; then
                    NODE_DONE=false
                    break
                fi
            done < "$JOB_FILE"
        else
            echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: local job file missing: $JOB_FILE"
            NODE_DONE=false
        fi

        if [ "$NODE_DONE" = true ]; then
            echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: ALL DONE"
            echo "# DONE $line" >> "$TEMP_FILE"
            continue
        fi

        ALL_DONE=false

        # Check instance state
        STATE=$(aws ec2 describe-instances \
            --instance-ids "$INST_ID" \
            --region "$REGION" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text 2>/dev/null || echo "unknown")

        case "$STATE" in
            running|pending)
                echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID ($INST_ID): $STATE"
                echo "$line" >> "$TEMP_FILE"
                ;;
            terminated|stopped|shutting-down)
                echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID ($INST_ID): $STATE — re-launching..."
                if NEW_ID=$(relaunch_instance "$NODE_ID" "$JOB_FILE"); then
                    echo "$NEW_ID $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
                else
                    echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: re-launch failed, keeping old instance record"
                    echo "$line" >> "$TEMP_FILE"
                fi
                ;;
            *)
                echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID ($INST_ID): unknown state '$STATE'"
                echo "$line" >> "$TEMP_FILE"
                ;;
        esac
    done < "$IDS_FILE"

    # Update the instance IDs file
    mv "$TEMP_FILE" "$IDS_FILE"

    if [ "$ALL_DONE" = true ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  All jobs complete. Monitor exiting."
        echo "═══════════════════════════════════════════════════════════"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
