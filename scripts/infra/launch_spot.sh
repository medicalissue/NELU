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
REGION="${AWS_REGION:-us-west-2}"
KEY_NAME="${KEY_NAME:-nelu-training}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-CHANGEME}"
IAM_ROLE="${IAM_INSTANCE_PROFILE:-}"
SUBNET_CANDIDATES="$(normalize_subnet_candidates || true)"

# Validate required settings
if [ "$SECURITY_GROUP" = "sg-CHANGEME" ] || [ -z "$SUBNET_CANDIDATES" ]; then
    echo "ERROR: Set SECURITY_GROUP and SUBNETS (or SUBNET) environment variables"
    exit 1
fi

if [ -z "$IAM_ROLE" ]; then
    echo "ERROR: Set IAM_INSTANCE_PROFILE to the ARN of the instance profile with S3 access"
    exit 1
fi

DATA_SNAPSHOT="${DATA_SNAPSHOT:-}"
if [ -z "$DATA_SNAPSHOT" ]; then
    echo "ERROR: Set DATA_SNAPSHOT to the EBS snapshot ID containing the training environment"
    echo "  Create one with: ./scripts/infra/launch_snapshot_setup.sh"
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
SPOT_RETRY_INTERVAL="${SPOT_RETRY_INTERVAL:-60}"
POLL_INTERVAL="${POLL_INTERVAL:-$SPOT_RETRY_INTERVAL}"
FORCE_NEW_RUN="${FORCE_NEW_RUN:-0}"

# ── Parse arguments ─────────────────────────────────────────────

if [ $# -lt 2 ]; then
    echo "Usage: $0 <n_nodes> <job_queue_dir>"
    echo ""
    echo "  n_nodes:       Number of spot instances to launch"
    echo "  job_queue_dir: Directory containing jobs_node1.txt, jobs_node2.txt, ..."
    echo ""
    echo "  Environment variables:"
    echo "    S3_BUCKET              S3 bucket for code/data/results (default: s3://nelu-datasets)"
    echo "    AWS_REGION             AWS region (default: us-west-2)"
    echo "    KEY_NAME               EC2 key pair name"
    echo "    SECURITY_GROUP         Security group ID (REQUIRED)"
    echo "    SUBNETS                Comma/space-separated subnet IDs to try (REQUIRED)"
    echo "    SUBNET                 Single subnet fallback (legacy)"
    echo "    IAM_INSTANCE_PROFILE   ARN of instance profile with S3 access (REQUIRED)"
    echo "    INSTANCE_TYPE          EC2 instance type (default: p5.48xlarge)"
    echo "    SPOT_RETRY_INTERVAL    Seconds between full retry rounds (default: 60)"
    echo "    FORCE_NEW_RUN          Ignore existing tracker and start a fresh run (default: 0)"
    echo ""
    echo "  These can be set in ${REPO_ROOT}/.env instead of exporting them."
    exit 1
fi

N_NODES="$1"
JOB_DIR="$2"
AMI_ID="$(resolve_ami "$REGION")"

echo "Using AMI: $AMI_ID"
echo "Subnet candidates: $SUBNET_CANDIDATES"

# ── Upload code to S3 ──────────────────────────────────────────

echo "Uploading code to S3..."
upload_repo_bundle "$REPO_ROOT" "${S3_BUCKET}/code/nelu-code.tar.gz"
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

# ── Orchestration helpers ────────────────────────────────────────

launch_node_instance() {
    local node_id="$1"
    local user_data_file
    local instance_id=""
    local launched_subnet=""
    local err_file
    local subnet

    user_data_file="$(render_user_data_file "$SCRIPT_DIR/user_data.sh" "$S3_BUCKET" "$node_id" "$WANDB_API_KEY" "$ORCH_RUN_ID")"
    err_file="$(mktemp)"

    for subnet in $SUBNET_CANDIDATES; do
        echo "  Trying subnet: $subnet" >&2
        if instance_id=$(aws ec2 run-instances \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SECURITY_GROUP" \
            --subnet-id "$subnet" \
            --iam-instance-profile "Arn=$IAM_ROLE" \
            --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Iops":10000,"Throughput":500}},{"DeviceName":"/dev/sdf","Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":500,"VolumeType":"gp3","Iops":10000,"Throughput":500,"DeleteOnTermination":true}}]' \
            --user-data "file://${user_data_file}" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nelu-node-${node_id}},{Key=Project,Value=nelu}]" \
            --region "$REGION" \
            --query 'Instances[0].InstanceId' \
            --output text 2>"$err_file"); then
            launched_subnet="$subnet"
            break
        fi
        sed 's/^/    /' "$err_file" >&2
    done

    rm -f "$err_file" "$user_data_file"

    if [ -n "$instance_id" ]; then
        echo "  Instance: $instance_id" >&2
        echo "  Subnet:   $launched_subnet" >&2
        printf '%s\n' "$instance_id"
        return 0
    fi

    return 1
}

node_failed_on_s3() {
    local node_id="$1"
    aws s3 ls "${S3_BUCKET}/orchestrator/${ORCH_RUN_ID}/node${node_id}.FAILED" >/dev/null 2>&1
}

job_done_on_s3() {
    local job_file="$1"
    local job_line
    local phase model act run_name jarg jclean

    if [ ! -f "$job_file" ]; then
        return 1
    fi

    while IFS= read -r job_line; do
        [[ "$job_line" =~ ^[[:space:]]*# ]] && continue
        [[ "$job_line" =~ ^[[:space:]]*$ ]] && continue

        # shellcheck disable=SC2086
        set -- $job_line
        phase="$1"
        model="$2"
        act="$3"
        shift 3
        run_name="${phase}_${model}_${act}"
        for jarg in "$@"; do
            jclean=$(echo "$jarg" | sed 's/^--//; s/=/_/g')
            run_name="${run_name}_${jclean}"
        done

        if ! aws s3 ls "${S3_BUCKET}/results/${run_name}/DONE" >/dev/null 2>&1; then
            return 1
        fi
    done < "$job_file"

    return 0
}

# ── Orchestration state ──────────────────────────────────────────

INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-${REPO_ROOT}/.nelu_instance_ids.txt}"
INIT_FILE="$(mktemp)"
EXISTING_RUN_ID=""
if [ "$FORCE_NEW_RUN" != "1" ] && [ -f "$INSTANCE_IDS_FILE" ]; then
    EXISTING_RUN_ID=$(awk '/^# Run ID:/ {print $4; exit}' "$INSTANCE_IDS_FILE")
fi
if [ -n "$EXISTING_RUN_ID" ]; then
    ORCH_RUN_ID="$EXISTING_RUN_ID"
else
    ORCH_RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')"
fi
echo "# NELU spot instance tracker — $(date -u)" > "$INIT_FILE"
echo "# Run ID: $ORCH_RUN_ID" >> "$INIT_FILE"
echo "# Format: INSTANCE_ID NODE_ID JOB_FILE" >> "$INIT_FILE"

for i in $(seq 1 "$N_NODES"); do
    JOB_FILE="${JOB_DIR}/jobs_node${i}.txt"
    EXISTING_LINE=""
    if [ -n "$EXISTING_RUN_ID" ] && [ -f "$INSTANCE_IDS_FILE" ]; then
        EXISTING_LINE=$(awk -v node="$i" '$1 !~ /^#/ && $2==node {print; exit}' "$INSTANCE_IDS_FILE")
    fi

    if [ -n "$EXISTING_LINE" ]; then
        echo "$EXISTING_LINE" >> "$INIT_FILE"
    else
        echo "PENDING $i $JOB_FILE" >> "$INIT_FILE"
    fi
done

mv "$INIT_FILE" "$INSTANCE_IDS_FILE"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  NELU Spot Orchestrator"
echo "═══════════════════════════════════════════════════════════"
echo "  Nodes:          $N_NODES"
echo "  Run ID:         $ORCH_RUN_ID"
echo "  Tracker file:   $INSTANCE_IDS_FILE"
echo "  Poll interval:  ${POLL_INTERVAL}s"
echo "  Retry interval: ${SPOT_RETRY_INTERVAL}s"
echo "  Results:        ${S3_BUCKET}/results/"
echo "═══════════════════════════════════════════════════════════"
echo ""

while true; do
    ALL_TERMINAL=true
    ANY_FAILED=false
    TEMP_FILE="$(mktemp)"
    echo "# NELU spot instance tracker — $(date -u)" > "$TEMP_FILE"
    echo "# Run ID: $ORCH_RUN_ID" >> "$TEMP_FILE"
    echo "# Format: INSTANCE_ID NODE_ID JOB_FILE" >> "$TEMP_FILE"

    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue

        INST_ID=$(echo "$line" | awk '{print $1}')
        NODE_ID=$(echo "$line" | awk '{print $2}')
        JOB_FILE=$(echo "$line" | awk '{print $3}')

        if [ "$INST_ID" = "FAILED" ] || node_failed_on_s3 "$NODE_ID"; then
            echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: FAILED"
            echo "FAILED $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
            ANY_FAILED=true
            continue
        fi

        if job_done_on_s3 "$JOB_FILE"; then
            echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: ALL DONE"
            echo "# DONE $INST_ID $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
            continue
        fi

        ALL_TERMINAL=false

        if [ "$INST_ID" = "PENDING" ]; then
            STATE="pending-launch"
        else
            STATE=$(aws ec2 describe-instances \
                --instance-ids "$INST_ID" \
                --region "$REGION" \
                --query 'Reservations[0].Instances[0].State.Name' \
                --output text 2>/dev/null || echo "unknown")
        fi

        case "$STATE" in
            running|pending)
                echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID ($INST_ID): $STATE"
                echo "$INST_ID $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
                ;;
            pending-launch|terminated|stopped|shutting-down|unknown)
                if [ "$STATE" = "pending-launch" ]; then
                    echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: waiting for initial spot capacity..."
                else
                    echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID ($INST_ID): $STATE — requesting replacement..."
                fi

                echo ""
                echo "── Node $NODE_ID ──"
                if NEW_ID=$(launch_node_instance "$NODE_ID"); then
                    echo "$NEW_ID $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
                else
                    echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID: no spot capacity yet. Keeping pending."
                    echo "PENDING $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
                fi
                ;;
            *)
                echo "[$(date -u '+%H:%M:%S')] Node $NODE_ID ($INST_ID): unexpected state '$STATE'"
                echo "$INST_ID $NODE_ID $JOB_FILE" >> "$TEMP_FILE"
                ;;
        esac
    done < "$INSTANCE_IDS_FILE"

    mv "$TEMP_FILE" "$INSTANCE_IDS_FILE"

    if [ "$ALL_TERMINAL" = true ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        if [ "$ANY_FAILED" = true ]; then
            echo "  One or more nodes failed. Orchestrator exiting with error."
            echo "═══════════════════════════════════════════════════════════"
            exit 1
        fi
        echo "  All jobs complete. Orchestrator exiting."
        echo "═══════════════════════════════════════════════════════════"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
