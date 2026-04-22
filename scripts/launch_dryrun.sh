#!/usr/bin/env bash
# Submit N dstack workers against the toy 4-job queue.
#
# Usage:
#     source .env
#     bash scripts/launch_dryrun.sh 2
set -euo pipefail

: "${WANDB_API_KEY:?source .env first}"
: "${WANDB_ENTITY:=}"

N="${1:-2}"

# Wipe prior dryrun state so sentinels from a previous attempt don't
# make workers report "nothing to do" immediately.
aws s3 rm s3://nelu-checkpoints/_dryrun --recursive --region us-west-2 --quiet || true

for i in $(seq 1 "$N"); do
    name="worker-dryrun-${i}"
    echo "▶ submitting ${name}"
    dstack apply -f .dstack/worker-dryrun.dstack.yml \
        -P . \
        -n "${name}" \
        -e WANDB_API_KEY="${WANDB_API_KEY}" \
        -e WANDB_ENTITY="${WANDB_ENTITY}" \
        -y -d
done

cat <<EOF

${N} dryrun workers submitted. Observe with:
    dstack ps
    dstack logs worker-dryrun-1 -f
    aws s3 ls s3://nelu-checkpoints/_dryrun/ --recursive --region us-west-2

Queue state: 4 toy jobs (2 models × 2 activations, 2 epochs × 512 samples).
Success criteria:
  - Each worker grabs 2 jobs without overlap (check lease ownership in logs).
  - Every toy job ends with a 'complete' sentinel in its S3 prefix.
  - Workers exit cleanly when the queue drains.
EOF
