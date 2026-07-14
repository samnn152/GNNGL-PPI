#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export QUICK_DATASET="shs148k"
export METRICS_CSV="${METRICS_CSV:-./outputs/proposal_results_148k_quick.csv}"

exec bash "$SCRIPT_DIR/run_proposal_experiments.sh" quick "$@"
