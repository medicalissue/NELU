#!/usr/bin/env bash
# Idle entry-script for ad-hoc spot workers. Used when we want a spot
# VM bootstrapped (venv, dataset volume, repo cloned) but no orchestrator
# running — we will SSH in and start training manually. The infinite
# sleep keeps bootstrap.sh from returning, which would otherwise trigger
# the self-terminate path.
#
# Override the bootstrap default with:
#   ENTRY_SCRIPT=scripts/sleep_forever.sh bash scripts/infra/run_worker.sh ...
exec sleep infinity
