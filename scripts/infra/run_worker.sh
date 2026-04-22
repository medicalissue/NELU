#!/usr/bin/env bash
# Launch ONE spot worker.
#
# Usage:
#   source .env
#   bash scripts/infra/run_worker.sh <az> [<instance-type>] [<name-suffix>]
#
# Positional:
#   az              us-west-2a|b|c|d. Picks the matching subnet.
#   instance-type   Defaults to p5.48xlarge (H100:8). Use g4dn.xlarge for dryrun.
#   name-suffix     Tag suffix, e.g. "dryrun-1"; defaults to the current ts.
#
# The /data volume is provisioned in-place from the dataset snapshot via
# a BlockDeviceMapping on run-instances — no pre-create step.
# DeleteOnTermination=true cleans the volume up on spot preemption.

set -euo pipefail

AZ="${1:?az required}"
INSTANCE_TYPE="${2:-p5.48xlarge}"
NAME_SUFFIX="${3:-$(date -u +%Y%m%dT%H%M%S)}"

AMI=ami-0027d9a89a2d7f75b                    # Ubuntu 22.04 + NVIDIA from our data instance
SG=sg-00c0e8d6f674b0ff4                      # launch-wizard-7 (ssh 22 open)
KEY=Kwonsaiserver                            # ssh key pair
IAM_PROFILE=nelu-worker-profile
DATA_SNAPSHOT=snap-0adfaa42ce378623c
REGION=us-west-2

case "$AZ" in
    us-west-2a) SUBNET=subnet-02611feabb32d468c ;;
    us-west-2b) SUBNET=subnet-0db9bbaeac9d76567 ;;
    us-west-2c) SUBNET=subnet-059b8cc477a51f6d7 ;;
    us-west-2d) SUBNET=subnet-086f01bf8cbb32315 ;;
    *) echo "unknown AZ $AZ" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_B64=$(bash "$SCRIPT_DIR/render_user_data.sh" | base64)

NAME="nelu-worker-${NAME_SUFFIX}"

echo "▶ launching ${NAME} in ${AZ} (${INSTANCE_TYPE})"
aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings '[
        {"DeviceName":"/dev/sda1",
         "Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}},
        {"DeviceName":"/dev/sdg",
         "Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}
    ]' \
    --user-data "$USER_DATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$NAME},
        {Key=Project,Value=gate-norm},
        {Key=Role,Value=worker}
    ]" \
    --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone,State:State.Name}' \
    --output json
