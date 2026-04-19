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

# ── Config (override via environment) ───────────────────────────

S3_BUCKET="${S3_BUCKET:-s3://nelu-experiments}"
REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-nelu-training}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-CHANGEME}"
SUBNET="${SUBNET:-subnet-CHANGEME}"
IAM_ROLE="${IAM_ROLE:-arn:aws:iam::instance-profile/S3Access}"

# Deep Learning AMI (Ubuntu 22.04) with CUDA pre-installed
AMI="${AMI:-ami-0c02fb55956c7d316}"

# Instance type preference (H100 > A100 > A10G)
INSTANCE_TYPE="${INSTANCE_TYPE:-p5.48xlarge}"
MAX_SPOT_PRICE="${MAX_SPOT_PRICE:-50.00}"

# ── Parse arguments ─────────────────────────────────────────────

if [ $# -lt 2 ]; then
    echo "Usage: $0 <n_nodes> <job_queue_dir>"
    echo ""
    echo "  n_nodes:       Number of spot instances to launch"
    echo "  job_queue_dir: Directory containing jobs_node1.txt, jobs_node2.txt, ..."
    echo ""
    echo "  Environment variables:"
    echo "    S3_BUCKET       S3 bucket for code/data/results (default: s3://nelu-experiments)"
    echo "    AWS_REGION      AWS region (default: us-east-1)"
    echo "    KEY_NAME        EC2 key pair name"
    echo "    SECURITY_GROUP  Security group ID"
    echo "    SUBNET          Subnet ID"
    echo "    INSTANCE_TYPE   EC2 instance type (default: p5.48xlarge)"
    exit 1
fi

N_NODES="$1"
JOB_DIR="$2"

# ── Upload code to S3 ──────────────────────────────────────────

echo "Uploading code to S3..."
cd "$REPO_ROOT"
tar czf /tmp/nelu-code.tar.gz \
    --exclude='data' --exclude='results' --exclude='wandb' \
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

echo ""
echo "Launching $N_NODES spot instances..."

for i in $(seq 1 "$N_NODES"); do
    echo ""
    echo "── Node $i ──"

    # Generate user data with the node number baked in
    USER_DATA=$(cat "$SCRIPT_DIR/user_data.sh" | \
        sed "s|__S3_BUCKET__|${S3_BUCKET}|g" | \
        sed "s|__NODE_ID__|${i}|g" | \
        base64 | tr -d '\n')

    # Request a spot instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --subnet-id "$SUBNET" \
        --iam-instance-profile "Arn=$IAM_ROLE" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":10000,"Throughput":500}}]' \
        --user-data "$USER_DATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nelu-node-${i}},{Key=Project,Value=nelu}]" \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' \
        --output text 2>&1) || true

    echo "  Instance: $INSTANCE_ID"
    echo "  Job file: jobs_node${i}.txt"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All $N_NODES instances launched."
echo ""
echo "  Monitor with:"
echo "    aws ec2 describe-instances --filters 'Name=tag:Project,Values=nelu' \\"
echo "      --query 'Reservations[].Instances[].{ID:InstanceId,State:State.Name,Type:InstanceType}' \\"
echo "      --output table --region $REGION"
echo ""
echo "  Results will appear in: ${S3_BUCKET}/results/"
echo "═══════════════════════════════════════════════════════════"
