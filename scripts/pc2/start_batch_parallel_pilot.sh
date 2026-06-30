#!/usr/bin/env bash
# Start batch_parallel pilot campaign.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --config "${SCRIPT_DIR}/batch_parallel_pilot.json" \
  "$@"
