#!/usr/bin/env bash
# Re-run Vitis cosim on existing Bn 2+2 (old) flash finals at FULL problem size
# (same N as csynth — no cosim_size_overrides). No LLM / no code regeneration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

BN22_ARTIFACT="flash_skills_20260620_004507"
STAMP="${C2HLS_FLASH_COSIM_STAMP:-$(date +%Y%m%d_%H%M%S)_bn22_full}"

export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-12:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-14400}"
export PC2_COSIM_ARRAY_MAX_PARALLEL="${PC2_COSIM_ARRAY_MAX_PARALLEL:-8}"

exec "${SCRIPT_DIR}/submit_flash_cosim_all.sh" \
  --stamp "${STAMP}" \
  --artifact "${BN22_ARTIFACT}" \
  --full-size \
  "$@"
