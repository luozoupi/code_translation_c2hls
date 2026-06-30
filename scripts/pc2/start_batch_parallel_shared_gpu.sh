#!/usr/bin/env bash
# Start shared GPU session for batch_parallel (wrapper).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
exec "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
