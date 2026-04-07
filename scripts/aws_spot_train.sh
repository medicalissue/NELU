#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  AWS Spot Fleet training — auto-retry with S3 checkpointing.
#
#  1. Launches spot request (tries H100 → A100 fallback)
#  2. Syncs code + data from S3
#  3. Runs training with --resume from latest checkpoint
#  4. Periodically syncs checkpoints to S3
#  5. On spot interruption: auto-retries
#
#  Prerequisites:
#    - AWS CLI configured (aws configure)
#    - S3 bucket created
#    - AMI with CUDA + conda ready (or use Deep Learning AMI)
#
#  Usage:
#    # One-time setup
#    bash scripts/aws_spot_train.sh setup
#
#    # Launch training
#    bash scripts/aws_spot_train.sh launch
#
#    # Check status
#    bash scripts/aws_spot_train.sh status
#
#    # Download results
#    bash scripts/aws_spot_train.sh download
# ═══════════════════════════════════════════════════════════════

set -e

# ── Config ───────────────────────────────────────────────────
S3_BUCKET="s3://nelu-experiments"           # ← change this
REGION="us-east-1"
KEY_NAME="your-key-pair"                    # ← change this
SECURITY_GROUP="sg-xxxxxxxx"               # ← change this
SUBNET="subnet-xxxxxxxx"                   # ← change this
AMI="ami-0c02fb55956c7d316"                # Deep Learning AMI (Ubuntu)

# Instance preference order (tries H100 first, falls back)
INSTANCE_TYPES="p5.48xlarge,p4d.24xlarge,p4de.24xlarge"
MAX_SPOT_PRICE="50.00"  # $/hr cap

S3_CODE="${S3_BUCKET}/code"
S3_DATA="${S3_BUCKET}/data"
S3_CKPT="${S3_BUCKET}/checkpoints"
S3_RESULTS="${S3_BUCKET}/results"
# ─────────────────────────────────────────────────────────────

case "${1:-help}" in

# ─── Setup: upload code + data to S3 ────────────────────────
setup)
    echo "Uploading code to S3..."
    cd "$(dirname "$0")/.."
    tar czf /tmp/nelu-code.tar.gz \
        --exclude='data' --exclude='results' --exclude='wandb' \
        --exclude='__pycache__' --exclude='.git' .
    aws s3 cp /tmp/nelu-code.tar.gz "${S3_CODE}/nelu-code.tar.gz"
    rm /tmp/nelu-code.tar.gz

    echo "Uploading data to S3 (CIFAR, CIFAR-100-C)..."
    aws s3 sync data/ "${S3_DATA}/" --exclude "*.tar.gz"

    echo "Done. Code + data on S3."
    echo "For ImageNet: aws s3 sync /path/to/imagenet ${S3_DATA}/imagenet/"
    ;;

# ─── Launch: create spot fleet request ──────────────────────
launch)
    TRAIN_SCRIPT="${2:-run_h100.sh}"

    # User data script — runs on instance startup
    cat > /tmp/spot_userdata.sh << 'USERDATA'
#!/bin/bash
set -ex

# Metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
echo "Instance: $INSTANCE_ID ($INSTANCE_TYPE)"

S3_BUCKET="__S3_BUCKET__"
S3_CODE="__S3_CODE__"
S3_DATA="__S3_DATA__"
S3_CKPT="__S3_CKPT__"
TRAIN_SCRIPT="__TRAIN_SCRIPT__"

# Setup spot interruption handler
cat > /usr/local/bin/spot_handler.sh << 'HANDLER'
#!/bin/bash
# Called 2 minutes before termination
echo "SPOT INTERRUPTION — saving checkpoint to S3"
aws s3 sync /workspace/results/ __S3_CKPT__/results/ || true
aws s3 sync /workspace/results/ __S3_BUCKET__/results/ || true
echo "Checkpoint saved. Goodbye."
HANDLER
chmod +x /usr/local/bin/spot_handler.sh

# Monitor for spot interruption (background)
(while true; do
    if curl -s -o /dev/null -w "%{http_code}" \
        http://169.254.169.254/latest/meta-data/spot/instance-action | grep -q 200; then
        /usr/local/bin/spot_handler.sh
        break
    fi
    sleep 5
done) &

# Setup workspace
mkdir -p /workspace && cd /workspace

# Download code
aws s3 cp "${S3_CODE}/nelu-code.tar.gz" /tmp/code.tar.gz
tar xzf /tmp/code.tar.gz -C /workspace/

# Download data
aws s3 sync "${S3_DATA}/" /workspace/data/ || true

# Download previous checkpoints (for --resume)
aws s3 sync "${S3_CKPT}/results/" /workspace/results/ || true

# Setup conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch || conda activate base
pip install timm wandb tqdm scikit-learn datasets transformers ninja 2>/dev/null

# Build CUDA kernel
cd /workspace && python -c "from nelu.cuda_kernel import NELUCUDA; print('CUDA kernel OK')"

# Start S3 checkpoint sync (every 30 min, background)
(while true; do
    sleep 1800
    aws s3 sync /workspace/results/ "${S3_CKPT}/results/" --quiet || true
done) &

# Run training
echo "Starting training: $TRAIN_SCRIPT"
bash "$TRAIN_SCRIPT" 2>&1 | tee /workspace/train.log

# Final sync
aws s3 sync /workspace/results/ "${S3_CKPT}/results/"
aws s3 cp /workspace/train.log "${S3_BUCKET}/logs/${INSTANCE_ID}.log"

echo "Training complete."
USERDATA

    # Substitute variables
    sed -i "s|__S3_BUCKET__|${S3_BUCKET}|g" /tmp/spot_userdata.sh
    sed -i "s|__S3_CODE__|${S3_CODE}|g" /tmp/spot_userdata.sh
    sed -i "s|__S3_DATA__|${S3_DATA}|g" /tmp/spot_userdata.sh
    sed -i "s|__S3_CKPT__|${S3_CKPT}|g" /tmp/spot_userdata.sh
    sed -i "s|__TRAIN_SCRIPT__|${TRAIN_SCRIPT}|g" /tmp/spot_userdata.sh

    USERDATA_B64=$(base64 -w0 /tmp/spot_userdata.sh)

    # Build fleet config
    cat > /tmp/spot_fleet.json << FLEET
{
    "SpotPrice": "${MAX_SPOT_PRICE}",
    "TargetCapacity": 1,
    "IamFleetRole": "arn:aws:iam::role/aws-ec2-spot-fleet-tagging-role",
    "LaunchSpecifications": [
        {
            "ImageId": "${AMI}",
            "InstanceType": "p5.48xlarge",
            "KeyName": "${KEY_NAME}",
            "SecurityGroups": [{"GroupId": "${SECURITY_GROUP}"}],
            "SubnetId": "${SUBNET}",
            "UserData": "${USERDATA_B64}",
            "BlockDeviceMappings": [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 500, "VolumeType": "gp3"}}
            ],
            "IamInstanceProfile": {"Arn": "arn:aws:iam::instance-profile/S3Access"}
        },
        {
            "ImageId": "${AMI}",
            "InstanceType": "p4d.24xlarge",
            "KeyName": "${KEY_NAME}",
            "SecurityGroups": [{"GroupId": "${SECURITY_GROUP}"}],
            "SubnetId": "${SUBNET}",
            "UserData": "${USERDATA_B64}",
            "BlockDeviceMappings": [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 500, "VolumeType": "gp3"}}
            ],
            "IamInstanceProfile": {"Arn": "arn:aws:iam::instance-profile/S3Access"}
        },
        {
            "ImageId": "${AMI}",
            "InstanceType": "p4de.24xlarge",
            "KeyName": "${KEY_NAME}",
            "SecurityGroups": [{"GroupId": "${SECURITY_GROUP}"}],
            "SubnetId": "${SUBNET}",
            "UserData": "${USERDATA_B64}",
            "BlockDeviceMappings": [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 500, "VolumeType": "gp3"}}
            ],
            "IamInstanceProfile": {"Arn": "arn:aws:iam::instance-profile/S3Access"}
        }
    ],
    "AllocationStrategy": "lowestPrice",
    "TerminateInstancesWithExpiration": true,
    "Type": "maintain",
    "ValidUntil": "$(date -u -d '+7 days' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+7d +%Y-%m-%dT%H:%M:%SZ)"
}
FLEET

    echo "Requesting spot fleet..."
    FLEET_ID=$(aws ec2 request-spot-fleet \
        --spot-fleet-request-config file:///tmp/spot_fleet.json \
        --region ${REGION} \
        --query 'SpotFleetRequestId' --output text)

    echo "Spot Fleet ID: ${FLEET_ID}"
    echo "${FLEET_ID}" > /tmp/nelu_fleet_id.txt
    echo ""
    echo "Monitor with: bash scripts/aws_spot_train.sh status"
    echo "Download with: bash scripts/aws_spot_train.sh download"
    ;;

# ─── Status: check fleet ────────────────────────────────────
status)
    FLEET_ID=$(cat /tmp/nelu_fleet_id.txt 2>/dev/null || echo "unknown")
    echo "Fleet ID: ${FLEET_ID}"

    if [ "$FLEET_ID" != "unknown" ]; then
        aws ec2 describe-spot-fleet-requests \
            --spot-fleet-request-ids "${FLEET_ID}" \
            --region ${REGION} \
            --query 'SpotFleetRequestConfigs[0].{State:SpotFleetRequestState,Fulfilled:FulfilledCapacity}' \
            --output table

        echo ""
        echo "Active instances:"
        aws ec2 describe-spot-fleet-instances \
            --spot-fleet-request-id "${FLEET_ID}" \
            --region ${REGION} \
            --query 'ActiveInstances[*].{ID:InstanceId,Type:InstanceType}' \
            --output table
    fi

    echo ""
    echo "S3 checkpoints:"
    aws s3 ls "${S3_CKPT}/results/" --recursive --human-readable 2>/dev/null | tail -10
    ;;

# ─── Download: get results from S3 ──────────────────────────
download)
    echo "Downloading results from S3..."
    mkdir -p results
    aws s3 sync "${S3_CKPT}/results/" results/
    aws s3 sync "${S3_BUCKET}/logs/" logs/
    echo "Done. Check results/ and logs/"
    ;;

# ─── Cancel: stop fleet ─────────────────────────────────────
cancel)
    FLEET_ID=$(cat /tmp/nelu_fleet_id.txt 2>/dev/null || echo "")
    if [ -n "$FLEET_ID" ]; then
        aws ec2 cancel-spot-fleet-requests \
            --spot-fleet-request-ids "${FLEET_ID}" \
            --terminate-instances \
            --region ${REGION}
        echo "Fleet ${FLEET_ID} cancelled."
    fi
    ;;

*)
    echo "Usage: $0 {setup|launch|status|download|cancel}"
    echo ""
    echo "  setup    - Upload code + data to S3"
    echo "  launch   - Request spot fleet (H100 → A100 fallback)"
    echo "  status   - Check fleet status + S3 checkpoints"
    echo "  download - Download results from S3"
    echo "  cancel   - Cancel fleet + terminate instances"
    ;;
esac
