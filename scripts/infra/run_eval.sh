#!/usr/bin/env bash
# Launch a single spot g5.12xlarge to run robustness evaluation on one
# or more checkpoints, reusing the standard data snapshot (ImageNet,
# ImageNet-C, ImageNet-A/R/O all pre-mounted at /data). Instance self-
# terminates when eval_orchestrate.sh finishes.
#
# Usage:
#   source .env
#   bash scripts/infra/run_eval.sh
#
# Or override defaults:
#   EVAL_EPOCH=250 INSTANCE_TYPE=g5.12xlarge \
#     EVAL_MODELS="configs/imagenet/swin_small.yaml:gelu:swin_small-gelu configs/imagenet/swin_small.yaml:nelu:swin_small-nelu" \
#     bash scripts/infra/run_eval.sh
#
# g5.12xlarge = 4× A10G / 48 vCPU / 192 GB RAM. Spot typically $1–2/h in
# us-west-2; the full four-benchmark sweep for two Swin-S checkpoints
# finishes in roughly 2–3 hours.

set -euo pipefail

# ── Defaults ───────────────────────────────────────────────────────
: "${INSTANCE_TYPE:=g5.12xlarge}"
: "${EVAL_EPOCH:=250}"
: "${EVAL_MODELS:=configs/imagenet/swin_small.yaml:gelu:swin_small-gelu configs/imagenet/swin_small.yaml:nelu:swin_small-nelu}"
: "${EVAL_RESULT_PREFIX:=eval-results}"

# AZ fallback order — g5 capacity usually fine, we still rotate.
: "${EVAL_AZS:=us-west-2a us-west-2b us-west-2c us-west-2d}"

# ── Infra constants (mirror run_worker.sh) ─────────────────────────
AMI=ami-0027d9a89a2d7f75b
SG=sg-00c0e8d6f674b0ff4
KEY=Kwonsaiserver
IAM_PROFILE=nelu-worker-profile
DATA_SNAPSHOT=snap-0adfaa42ce378623c
REGION=us-west-2

subnet_for_az() {
    case "$1" in
        us-west-2a) echo "subnet-02611feabb32d468c" ;;
        us-west-2b) echo "subnet-0db9bbaeac9d76567" ;;
        us-west-2c) echo "subnet-059b8cc477a51f6d7" ;;
        us-west-2d) echo "subnet-086f01bf8cbb32315" ;;
        *) return 2 ;;
    esac
}

: "${CKPT_BUCKET:?CKPT_BUCKET required (e.g. from .env)}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"  # render_user_data requires this; eval doesn't log to W&B
export CKPT_BUCKET WANDB_API_KEY
export EVAL_EPOCH EVAL_MODELS EVAL_RESULT_PREFIX
export ENTRY_SCRIPT="scripts/eval_orchestrate.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_B64=$(bash "$SCRIPT_DIR/render_user_data.sh" | base64)

NAME_SUFFIX="eval-$(date -u +%Y%m%dT%H%M%S)"
NAME="nelu-eval-${NAME_SUFFIX}"

echo "▶ eval launcher config"
echo "  INSTANCE_TYPE       = $INSTANCE_TYPE"
echo "  EVAL_EPOCH          = $EVAL_EPOCH"
echo "  EVAL_MODELS         = $EVAL_MODELS"
echo "  EVAL_RESULT_PREFIX  = $EVAL_RESULT_PREFIX"
echo "  AZ order            = $EVAL_AZS"
echo

launch_in_az() {
    local az="$1"
    local subnet
    subnet=$(subnet_for_az "$az") || { echo "unknown az $az" >&2; return 2; }

    local market_args=()
    local market_label="spot"
    if [[ "${EVAL_USE_SPOT:-1}" == "1" ]]; then
        market_args=(--instance-market-options 'MarketType=spot')
    else
        market_label="on-demand"
    fi
    echo "▶ launching ${NAME} in ${az} (${INSTANCE_TYPE}, ${market_label})"
    aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY" \
        --subnet-id "$subnet" \
        --security-group-ids "$SG" \
        --iam-instance-profile "Name=$IAM_PROFILE" \
        ${market_args[@]+"${market_args[@]}"} \
        --block-device-mappings '[
            {"DeviceName":"/dev/sda1",
             "Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}},
            {"DeviceName":"/dev/sdg",
             "Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":500,"VolumeType":"gp3",
                    "Iops":16000,"Throughput":1000,"DeleteOnTermination":true}}
        ]' \
        --user-data "$USER_DATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[
            {Key=Name,Value=$NAME},
            {Key=Project,Value=gate-norm},
            {Key=Role,Value=eval}
        ]" \
        --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone,State:State.Name}' \
        --output json
}

for az in $EVAL_AZS; do
    if launch_in_az "$az"; then
        echo
        echo "✓ launched in $az"
        exit 0
    fi
    echo "  ↳ $az failed, trying next"
done

echo "FATAL: no capacity in any AZ" >&2
exit 3
