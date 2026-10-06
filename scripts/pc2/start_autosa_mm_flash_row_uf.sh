#!/usr/bin/env bash
# Flash-only autosa_mm with mandatory ROW_UF=64 (one I-tile).
# Same 90-skill flash pack as frozen mmflow. Does not chain compute/stream.
# Does not touch 20260830_mmflow or the DSP-floor campaigns.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_flash_row_uf.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_flash_row_uf.sh --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

export C2HLS_FLASH_ROW_UF="${C2HLS_FLASH_ROW_UF:-64}"
unset C2HLS_FLASH_MIN_DSP || true
export C2HLS_FLASH_ONLY=1
export C2HLS_TURNS="${C2HLS_TURNS:-7}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-7200}"
export C2HLS_POST_FLASH_DSE=0
export C2HLS_DSE_CHAIN_FLASH=0
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_STREAM_CHAIN_FLASH=0
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-mmuf64}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_flash_rowuf64}"

exec "${SCRIPT_DIR}/start_autosa_mm_flow.sh" "$@"
