#!/usr/bin/env bash
# Flash-only autosa_mm with the distilled on-chip GEMM pack (940-class).
# Does not load the 90-skill dump. Does not chain compute/stream.
# Does not touch 20260830_mmflow.
#
# Arm C (default): knobs + 7 repair turns
#   ./scripts/pc2/start_autosa_mm_flash_onchip.sh --endpoint-url http://login5:18092/v1
# Arm B: knobs + one rewrite
#   C2HLS_TURNS=1 ./scripts/pc2/start_autosa_mm_flash_onchip.sh --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

ONCHIP_JSON="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json"
if [[ ! -f "${ONCHIP_JSON}" ]]; then
  echo "ERROR: missing ${ONCHIP_JSON}" >&2
  exit 2
fi

TURNS="${C2HLS_TURNS:-7}"
PE="${C2HLS_FLASH_PE_BLK:-16}"
case "${PE}" in
  16|32|64) ;;
  *)
    echo "ERROR: C2HLS_FLASH_PE_BLK must be 16, 32, or 64 (got '${PE}')" >&2
    exit 2
    ;;
esac

unset BATCH_PARALLEL_ARTIFACT_PREFIX || true
unset C2HLS_FLASH_ROW_UF C2HLS_FLASH_TILE_PP C2HLS_FLASH_SKILL_ENTRIES_JSON || true
unset C2HLS_PE_RECIPE || true

export C2HLS_FLASH_ONCHIP=1
# C2HLS_FLASH_ONCHIP also hard-rejects fused load_A_B / zipped A+B pipelines.
export C2HLS_FLASH_PE_BLK="${PE}"
export C2HLS_FLASH_MIN_DSP="${C2HLS_FLASH_MIN_DSP:-500}"
export C2HLS_FLASH_ONLY=1
export C2HLS_TURNS="${TURNS}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export C2HLS_POST_FLASH_DSE=0
export C2HLS_DSE_CHAIN_FLASH=0
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_STREAM_CHAIN_FLASH=0
export C2HLS_PACKAGED_SKILLS_JSON="${ONCHIP_JSON}"
export C2HLS_PACKAGED_SKILLS_ONLY=1
export BATCH_PARALLEL_VARIANT=autosa_onchip_gemm
export PC2_BATCH_JOB_PREFIX="mmonchip${TURNS}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_autosa_mm_flash_onchip_t${TURNS}"

exec "${SCRIPT_DIR}/start_autosa_mm_flow.sh" "$@"
