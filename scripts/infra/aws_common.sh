#!/bin/bash

DEFAULT_DLAMI_SSM_PARAM="/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"

load_env_file() {
    local explicit_path="${1:-}"
    local repo_root="${2:-.}"
    local env_file=""

    if [ -n "$explicit_path" ]; then
        env_file="$explicit_path"
    elif [ -n "${NELU_ENV_FILE:-}" ]; then
        env_file="$NELU_ENV_FILE"
    elif [ -f "$repo_root/.env" ]; then
        env_file="$repo_root/.env"
    fi

    if [ -z "$env_file" ]; then
        return 0
    fi

    if [ ! -f "$env_file" ]; then
        echo "ERROR: env file not found: $env_file" >&2
        return 1
    fi

    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
}

resolve_ami() {
    local region="$1"
    local ami_override="${AMI:-}"
    local param="${AMI_SSM_PARAM:-$DEFAULT_DLAMI_SSM_PARAM}"
    local ami_id=""

    if [ -n "$ami_override" ]; then
        printf '%s\n' "$ami_override"
        return 0
    fi

    ami_id=$(aws ssm get-parameter \
        --name "$param" \
        --region "$region" \
        --query 'Parameter.Value' \
        --output text 2>/dev/null) || {
        echo "ERROR: failed to resolve AMI from SSM parameter $param in region $region" >&2
        return 1
    }

    if [ -z "$ami_id" ] || [ "$ami_id" = "None" ]; then
        echo "ERROR: SSM parameter $param returned an empty AMI ID" >&2
        return 1
    fi

    printf '%s\n' "$ami_id"
}

render_user_data_file() {
    local template="$1"
    local s3_bucket="$2"
    local node_id="$3"
    local wandb_api_key="${4:-}"
    local out_file

    out_file=$(mktemp)
    S3_BUCKET_RENDER="$s3_bucket" \
    NODE_ID_RENDER="$node_id" \
    WANDB_API_KEY_RENDER="$wandb_api_key" \
    python - "$template" "$out_file" <<'PY'
import os
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

text = template_path.read_text()
replacements = {
    "__S3_BUCKET__": os.environ["S3_BUCKET_RENDER"],
    "__NODE_ID__": os.environ["NODE_ID_RENDER"],
    "__WANDB_API_KEY__": os.environ.get("WANDB_API_KEY_RENDER", ""),
}
for old, new in replacements.items():
    text = text.replace(old, new)

out_path.write_text(text)
PY

    printf '%s\n' "$out_file"
}
