#!/usr/bin/env bash
# Flash-only autosa_mm: fill U280 DSP (not a random 2000/5000 floor), stay
# strictly below 100% of DSP/BRAM/FF/LUT/URAM, keep the lowest-latency filled
# kernel. Same 90-skill flash pack as frozen mmflow. Does not chain DSE/stream.
# Does not touch 20260830_mmflow or other frozen trees.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_flash_dsp_redo.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_flash_dsp_redo.sh --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
FLOW_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    *) FLOW_ARGS+=("$1"); shift ;;
  esac
done

export C2HLS_FLASH_DSP_REDO=1
export C2HLS_FLASH_MIN_DSP="${C2HLS_FLASH_MIN_DSP:-300}"
export C2HLS_FLASH_MAX_DSP="${C2HLS_FLASH_MAX_DSP:-9024}"
export C2HLS_FLASH_DSP_FILL_PCT="${C2HLS_FLASH_DSP_FILL_PCT:-90}"
if [[ -z "${C2HLS_CANDIDATES_PER_STEP:-}" ]]; then
  export C2HLS_CANDIDATES_PER_STEP='{"flash": 3}'
fi
export C2HLS_FLASH_ONLY=1
export C2HLS_TURNS="${C2HLS_TURNS:-7}"
export C2HLS_POST_FLASH_DSE=0
export C2HLS_DSE_CHAIN_FLASH=0
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_STREAM_CHAIN_FLASH=0
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-mmfdspr}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_flash_dsp_redo}"
export BATCH_PARALLEL_STAMP="${STAMP}"

echo "=== flash_dsp_redo fill>=${C2HLS_FLASH_DSP_FILL_PCT}% of ${C2HLS_FLASH_MAX_DSP} DSP, cap <100% all resources, pick lowest latency ==="

"${SCRIPT_DIR}/start_autosa_mm_flow.sh" --stamp "${STAMP}" "${FLOW_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
if [[ -d "${CAMPAIGN_ROOT}" ]]; then
  printf '1\n' > "${CAMPAIGN_ROOT}/flash_only.txt"
  printf '1\n' > "${CAMPAIGN_ROOT}/flash_dsp_redo.txt"
  printf '%s\n' "${C2HLS_FLASH_MIN_DSP}" > "${CAMPAIGN_ROOT}/flash_min_dsp.txt"
  printf '%s\n' "${C2HLS_FLASH_MAX_DSP}" > "${CAMPAIGN_ROOT}/flash_max_dsp.txt"
  printf '%s\n' "${C2HLS_FLASH_DSP_FILL_PCT}" > "${CAMPAIGN_ROOT}/flash_dsp_fill_pct.txt"
  printf '%s\n' "${C2HLS_CANDIDATES_PER_STEP}" > "${CAMPAIGN_ROOT}/candidates_per_step.txt"
  printf '%s\n' "${C2HLS_THINKING:-api_default}" > "${CAMPAIGN_ROOT}/thinking.txt"
  cat > "${CAMPAIGN_ROOT}/NOTES.md" <<'EOF'
# flash_dsp_redo

Flash-only. Fill Alveo U280 DSP (9024) instead of accepting a random floor
(2000 vs 5000). All of DSP/BRAM/FF/LUT/URAM must stay strictly below 100%.
Three flash candidates; the pipeline keeps the filled legal kernel with the
lowest latency_cycles (quote min/max + DSP, never interval).

Does not overwrite frozen mmflow / enforcement / flash16k / dse13160 trees.
EOF
fi
