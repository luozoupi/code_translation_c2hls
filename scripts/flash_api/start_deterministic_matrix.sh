#!/usr/bin/env bash
# Run full 10-mode deterministic flash matrix via commercial LLM API.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
STAMP="${C2HLS_FLASH_API_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL="${C2HLS_MODEL:-${C2HLS_API_MODEL:-claude-sonnet-4-6}}"
echo "flash_api deterministic matrix stamp=$STAMP model=$MODEL"
exec python3 scripts/flash_api/run_matrix.py --set deterministic --stamp "$STAMP" --model "$MODEL" "$@"
