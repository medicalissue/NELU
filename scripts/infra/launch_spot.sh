#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Launch N spot instances, each running a slice of the job queue.
#
#  Each instance boots from a Deep Learning AMI, runs user_data.sh
#  to install deps + clone the repo, then starts run_all.sh with
#  its assigned job file. Instances auto-shutdown when done.
#
#  Usage:
#    ./scripts/infra/launch_spot.sh <n_nodes> <job_queue_dir>
#    ./scripts/infra/launch_spot.sh 3 scripts/
#
#  Prerequisites:
#    - AWS CLI configured with appropriate permissions
#    - Job files: scripts/jobs_node1.txt, scripts/jobs_node2.txt, ...
#    - S3 bucket with code tarball uploaded (run setup first)
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
            echo "Usage: $0 [--env-file FILE] <n_nodes> <job_queue_dir>"
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

# ── Config (override via environment) ───────────────────────────

S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-nelu-training}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-CHANGEME}"
SUBNET="${SUBNET:-subnet-CHANGEME}"
IAM_ROLE="${IAM_INSTANCE_PROFILE:-}"

# Validate required settings
if [ "$SECURITY_GROUP" = "sg-CHANGEME" ] || [ "$SUBNET" = "subnet-CHANGEME" ]; then
    echo "ERROR: Set SECURITY_GROUP and SUBNET environment variables"
    exit 1
fi

if [ -z "$IAM_ROLE" ]; then
    echo "ERROR: Set IAM_INSTANCE_PROFILE to the ARN of the instance profile with S3 access"
    exit 1
fi

DATA_SNAPSHOT="${DATA_SNAPSHOT:-}"
if [ -z "$DATA_SNAPSHOT" ]; then
    echo "ERROR: Set DATA_SNAPSHOT to the EBS snapshot ID containing the training environment"
    echo "  Create one with: ./scripts/infra/setup_snapshot.sh"
    exit 1
fi

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. wandb logging will be disabled on instances."
    echo "  To enable: set WANDB_API_KEY in .env"
fi

# Instance type preference (H100 > A100 > A10G)
INSTANCE_TYPE="${INSTANCE_TYPE:-p5.48xlarge}"
MAX_SPOT_PRICE="${MAX_SPOT_PRICE:-30.00}"  # p5.48xlarge spot ~$16-25, cap at $30

# ── Parse arguments ─────────────────────────────────────────────

if [ $# -lt 2 ]; then
    echo "Usage: $0 <n_nodes> <job_queue_dir>"
    echo ""
    echo "  n_nodes:       Number of spot instances to launch"
    echo "  job_queue_dir: Directory containing jobs_node1.txt, jobs_node2.txt, ..."
    echo ""
    echo "  Environment variables:"
    echo "    S3_BUCKET              S3 bucket for code/data/results (default: s3://nelu-datasets)"
    echo "    AWS_REGION             AWS region (default: us-east-1)"
    echo "    KEY_NAME               EC2 key pair name"
    echo "    SECURITY_GROUP         Security group ID (REQUIRED)"
    echo "    SUBNET                 Subnet ID (REQUIRED)"
    echo "    IAM_INSTANCE_PROFILE   ARN of instance profile with S3 access (REQUIRED)"
    echo "    INSTANCE_TYPE          EC2 instance type (default: p5.48xlarge)"
    echo ""
    echo "  These can be set in ${REPO_ROOT}/.env instead of exporting them."
    exit 1
fi

N_NODES="$1"
JOB_DIR="$2"
AMI_ID="$(resolve_ami "$REGION")"

echo "Using AMI: $AMI_ID"

# ── Upload code to S3 ──────────────────────────────────────────

echo "Uploading code to S3..."
cd "$REPO_ROOT"
tar czf /tmp/nelu-code.tar.gz \
    --exclude='data' --exclude='results' --exclude='wandb' \
    --exclude='.env' --exclude='.env.*' \
    --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' .
aws s3 cp /tmp/nelu-code.tar.gz "${S3_BUCKET}/code/nelu-code.tar.gz" --quiet
rm /tmp/nelu-code.tar.gz
echo "  Code uploaded to ${S3_BUCKET}/code/"

# ── Upload job files ────────────────────────────────────────────

for i in $(seq 1 "$N_NODES"); do
    JOB_FILE="${JOB_DIR}/jobs_node${i}.txt"
    if [ ! -f "$JOB_FILE" ]; then
        echo "ERROR: Job file not found: $JOB_FILE"
        exit 1
    fi
    aws s3 cp "$JOB_FILE" "${S3_BUCKET}/jobs/jobs_node${i}.txt" --quiet
    echo "  Uploaded $JOB_FILE"
done

# ── Launch instances ────────────────────────────────────────────

INSTANCE_IDS_FILE="/tmp/nelu_instance_ids.txt"
echo "# NELU spot instance tracker — $(date -u)" > "$INSTANCE_IDS_FILE"
echo "# Format: INSTANCE_ID NODE_ID JOB_FILE" >> "$INSTANCE_IDS_FILE"

echo ""
echo "Launching $N_NODES spot instances..."

for i in $(seq 1 "$N_NODES"); do
    echo ""
    echo "── Node $i ──"

    USER_DATA_FILE="$(render_user_data_file "$SCRIPT_DIR/user_data.sh" "$S3_BUCKET" "$i" "$WANDB_API_KEY")"

    # Request a spot instance (one-time / terminate on interruption)
    if ! INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --subnet-id "$SUBNET" \
        --iam-instance-profile "Arn=$IAM_ROLE" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Iops":10000,"Throughput":500}},{"DeviceName":"/dev/sdf","Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":500,"VolumeType":"gp3","Iops":10000,"Throughput":500,"DeleteOnTermination":true}}]' \
        --user-data "file://${USER_DATA_FILE}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nelu-node-${i}},{Key=Project,Value=nelu}]" \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' \
        --output text); then
        rm -f "$USER_DATA_FILE"
        echo "ERROR: failed to launch node $i" >&2
        exit 1
    fi
    rm -f "$USER_DATA_FILE"

    echo "  Instance: $INSTANCE_ID"
    echo "  Job file: jobs_node${i}.txt"
    echo "$INSTANCE_ID $i ${JOB_DIR}/jobs_node${i}.txt" >> "$INSTANCE_IDS_FILE"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All $N_NODES instances launched."
echo "  Instance IDs saved to: $INSTANCE_IDS_FILE"
echo ""
echo "  Monitor with:"
echo "    aws ec2 describe-instances --filters 'Name=tag:Project,Values=nelu' \\"
echo "      --query 'Reservations[].Instances[].{ID:InstanceId,State:State.Name,Type:InstanceType}' \\"
echo "      --output table --region $REGION"
echo ""
echo "  Auto-relaunch terminated instances with:"
echo "    ./scripts/infra/monitor_spot.sh $INSTANCE_IDS_FILE"
echo ""
echo "  Results will appear in: ${S3_BUCKET}/results/"
echo "═══════════════════════════════════════════════════════════"
