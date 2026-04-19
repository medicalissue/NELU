#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Spot interruption handler — runs as a background daemon.
#
#  Polls the EC2 metadata endpoint every 5 seconds. When a spot
#  interruption notice is detected (2-minute warning), sends
#  SIGTERM to the training process so it can save a checkpoint
#  and sync to S3.
#
#  Usage:
#    bash scripts/infra/spot_interrupt_handler.sh &
#
#  The handler writes to /var/log/spot-handler.log for debugging.
# ═══════════════════════════════════════════════════════════════

set -uo pipefail

POLL_INTERVAL="${SPOT_POLL_INTERVAL:-5}"
METADATA_URL="http://169.254.169.254/latest/meta-data/spot/instance-action"
TOKEN_URL="http://169.254.169.254/latest/api/token"
LOG_FILE="/var/log/spot-handler.log"
S3_BUCKET="${S3_BUCKET:-s3://nelu-datasets}"
RESULTS_DIR="${RESULTS_DIR:-/data/results}"
LOG_DIR="${LOG_DIR:-/data/logs}"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Spot interrupt handler started (poll interval: ${POLL_INTERVAL}s)"

# Get IMDSv2 token
get_token() {
    curl -s -X PUT "$TOKEN_URL" \
        -H "X-aws-ec2-metadata-token-ttl-seconds:300" 2>/dev/null || echo ""
}

TOKEN=$(get_token)
TOKEN_TIME=$(date +%s)

while true; do
    # Refresh token every 4 minutes (it expires after 5)
    NOW=$(date +%s)
    if [ $((NOW - TOKEN_TIME)) -gt 240 ]; then
        TOKEN=$(get_token)
        TOKEN_TIME=$NOW
    fi

    # Check for interruption notice
    HTTP_CODE=$(curl -s -o /tmp/spot-action.json -w "%{http_code}" \
        -H "X-aws-ec2-metadata-token: $TOKEN" \
        "$METADATA_URL" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        ACTION=$(cat /tmp/spot-action.json)
        log "SPOT INTERRUPTION DETECTED: $ACTION"

        # Find and signal the training process.
        # We rely on epoch checkpoints plus an emergency sync here.
        TRAIN_PIDS=$(pgrep -f "torchrun\|train/train_imagenet_timm.py\|train/train_cifar.py\|scripts/run_all.sh\|scripts/run_single.sh" 2>/dev/null || echo "")

        if [ -n "$TRAIN_PIDS" ]; then
            log "Sending SIGTERM to training processes: $TRAIN_PIDS"
            for pid in $TRAIN_PIDS; do
                kill -TERM "$pid" 2>/dev/null || true
            done
        else
            log "No training processes found."
        fi

        log "Waiting 30 seconds before emergency sync..."
        sleep 30

        log "Emergency S3 sync..."
        [ -d "$RESULTS_DIR" ] && aws s3 sync "$RESULTS_DIR/" "${S3_BUCKET}/results/" --quiet 2>/dev/null || true
        [ -d "$LOG_DIR" ] && aws s3 sync "$LOG_DIR/" "${S3_BUCKET}/logs/" --quiet 2>/dev/null || true
        log "S3 sync complete. Spot handler exiting."
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
