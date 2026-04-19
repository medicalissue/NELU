#!/bin/bash

DEFAULT_DLAMI_SSM_PARAM="/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"

resolve_env_file_path() {
    local explicit_path="${1:-}"
    local repo_root="${2:-.}"

    if [ -n "$explicit_path" ]; then
        printf '%s\n' "$explicit_path"
    elif [ -n "${NELU_ENV_FILE:-}" ]; then
        printf '%s\n' "$NELU_ENV_FILE"
    else
        printf '%s/.env\n' "$repo_root"
    fi
}

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

normalize_subnet_candidates() {
    local raw="${SUBNETS:-${SUBNET:-}}"

    if [ -z "$raw" ]; then
        return 1
    fi

    python - "$raw" <<'PY'
import re
import sys

parts = [p for p in re.split(r"[\s,]+", sys.argv[1].strip()) if p]
seen = set()
ordered = []
for part in parts:
    if part not in seen:
        seen.add(part)
        ordered.append(part)

if not ordered:
    raise SystemExit(1)

print(" ".join(ordered))
PY
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

upload_repo_bundle() {
    local repo_root="$1"
    local dest_uri="$2"
    local tarball="/tmp/nelu-code-$$.tar.gz"

    (
        cd "$repo_root"
        # COPYFILE_DISABLE=1 tells macOS tar to skip Apple-specific
        # extended attributes (._* resource forks, xattr headers).
        COPYFILE_DISABLE=1 tar czf "$tarball" \
            --exclude='data' --exclude='results' --exclude='wandb' \
            --exclude='.env' --exclude='.env.*' \
            --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
            --exclude='.DS_Store' .
    )
    aws s3 cp "$tarball" "$dest_uri" --quiet
    rm -f "$tarball"
}

render_user_data_file() {
    local template="$1"
    local s3_bucket="$2"
    local node_id="$3"
    local wandb_api_key="${4:-}"
    local orch_run_id="${5:-}"
    local out_file

    out_file=$(mktemp)
    S3_BUCKET_RENDER="$s3_bucket" \
    NODE_ID_RENDER="$node_id" \
    WANDB_API_KEY_RENDER="$wandb_api_key" \
    ORCH_RUN_ID_RENDER="$orch_run_id" \
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
    "__ORCH_RUN_ID__": os.environ.get("ORCH_RUN_ID_RENDER", ""),
}
for old, new in replacements.items():
    text = text.replace(old, new)

out_path.write_text(text)
PY

    printf '%s\n' "$out_file"
}

set_env_value() {
    local env_file="$1"
    local key="$2"
    local value="$3"

    mkdir -p "$(dirname "$env_file")"

    python - "$env_file" "$key" "$value" <<'PY'
from pathlib import Path
import re
import sys

env_path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]

if env_path.exists():
    lines = env_path.read_text().splitlines()
else:
    lines = []

pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
replaced = False
out = []

for line in lines:
    if pattern.match(line):
        out.append(f"{key}={value}")
        replaced = True
    else:
        out.append(line)

if not replaced:
    if out and out[-1] != "":
        out.append("")
    out.append(f"{key}={value}")

env_path.write_text("\n".join(out) + "\n")
PY
}
