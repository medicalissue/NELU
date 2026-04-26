#!/usr/bin/env bash
# Setup script for SkyPilot workers. Idempotent; safe to re-run.
set -euxo pipefail

# SkyPilot AMIs ship PyTorch and CUDA drivers already.
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
