#!/usr/bin/env bash
# Full-size Vitis cosim for all top-5 flash variants (27 benches each, no LLM).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

TOP5_ARTIFACTS=(
  flash_all_skills_no_avoids_global_20260620_113247
  flash_all_new_skills_avoids_global_20260623_024548
  flash_all_new_skills_no_avoids_global_20260621_075846
  flash_noskills_20260620_004507
  flash_all_skills_avoids_global_20260620_113247
)

STAMP="${C2HLS_FLASH_COSIM_STAMP:-$(date +%Y%m%d_%H%M%S)_top5_full}"

export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-12:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-14400}"
export PC2_COSIM_ARRAY_MAX_PARALLEL="${PC2_COSIM_ARRAY_MAX_PARALLEL:-8}"

ARGS=(--stamp "${STAMP}" --full-size)
for art in "${TOP5_ARTIFACTS[@]}"; do
  ARGS+=(--artifact "${art}")
done

exec "${SCRIPT_DIR}/submit_flash_cosim_all.sh" "${ARGS[@]}" --individual "$@"
