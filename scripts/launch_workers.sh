#!/usr/bin/env bash
# Launch N dstack workers for the ImageNet campaign.
#
#   source .env
#   bash scripts/launch_workers.sh 4        # spawns worker-1 ... worker-4
#
# Each worker pulls jobs from the S3-backed queue until the queue is
# drained. Workers are independent — adding more (`launch_workers.sh 6`
# after the first batch) just means more parallel capacity; the S3 lease
# layer keeps them from stepping on each other.

set -euo pipefail

: "${WANDB_API_KEY:?source .env first}"
: "${WANDB_ENTITY:=}"

N="${1:-2}"

for i in $(seq 1 "$N"); do
    name="worker-${i}"
    echo "▶ submitting ${name}"
    dstack apply -f .dstack/worker.dstack.yml \
        -P . \
        -n "${name}" \
        -e WANDB_API_KEY="${WANDB_API_KEY}" \
        -e WANDB_ENTITY="${WANDB_ENTITY}" \
        -y -d
done

cat <<EOF

All ${N} workers submitted. Monitor with:
    dstack ps                   # active runs
    dstack logs worker-1 -f     # stream a worker's log
    aws s3 ls s3://nelu-checkpoints/ --recursive | grep complete   # done list
EOF
