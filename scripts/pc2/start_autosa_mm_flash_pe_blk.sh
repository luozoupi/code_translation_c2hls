#!/usr/bin/env bash
# Flash-only autosa_mm with mandatory PE_BLK (16, 32, or 64).
# Same 90-skill flash pack + DSP floor as the compute-II campaign.
# Does not chain compute/stream. Does not touch 20260830_mmflow.
#
# Usage:
#   C2HLS_FLASH_PE_BLK=16 ./scripts/pc2/start_autosa_mm_flash_pe_blk.sh --endpoint-url http://login5:18092/v1
#   C2HLS_FLASH_PE_BLK=32 ./scripts/pc2/start_autosa_mm_flash_pe_blk.sh --endpoint-url http://login5:18092/v1
#   C2HLS_FLASH_PE_BLK=64 ./scripts/pc2/start_autosa_mm_flash_pe_blk.sh --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PE="${C2HLS_FLASH_PE_BLK:-16}"
case "${PE}" in
  16|32|64) ;;
  *)
    echo "ERROR: C2HLS_FLASH_PE_BLK must be 16, 32, or 64 (got '${PE}')" >&2
    exit 2
    ;;
esac

unset C2HLS_FLASH_ROW_UF || true
export C2HLS_FLASH_PE_BLK="${PE}"
export C2HLS_FLASH_MIN_DSP="${C2HLS_FLASH_MIN_DSP:-500}"
export C2HLS_FLASH_ONLY=1
export C2HLS_TURNS="${C2HLS_TURNS:-7}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export C2HLS_POST_FLASH_DSE=0
export C2HLS_DSE_CHAIN_FLASH=0
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_STREAM_CHAIN_FLASH=0
export PC2_BATCH_JOB_PREFIX="mmpe${PE}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_autosa_mm_flash_dsp500_pe${PE}"

exec "${SCRIPT_DIR}/start_autosa_mm_flow.sh" "$@"
