#!/usr/bin/env bash
# Full-size Vitis cosim for flash fixed-corpus phase_b (translator) kernels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_FLASH_COSIM_STAMP:-fixed_cosim_flash_phase_b_$(date +%Y%m%d)}"
FLASH_STAMP="${C2HLS_FLASH_FIXED_COSIM_STAMP:-}"
DATE_PREFIX="$(printf '%s' "${FLASH_STAMP}" | grep -oE '[0-9]{8}' | head -1 || true)"
if [[ -z "${DATE_PREFIX}" ]]; then
  DATE_PREFIX="$(printf '%s' "${STAMP}" | grep -oE '[0-9]{8}' | head -1 || date +%Y%m%d)"
fi
ARTIFACT_GLOB="${C2HLS_FLASH_COSIM_ARTIFACT_GLOB:-flash_fixed_cosim_*_${DATE_PREFIX}_fixed_cosim_flash}"
export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_KERNEL="phase_b"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-13:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"

exec "${SCRIPT_DIR}/submit_flash_cosim_all.sh" \
  --stamp "${STAMP}" \
  --artifact-glob "${ARTIFACT_GLOB}" \
  --kernel-source phase_b \
  --full-size \
  --individual \
  "$@"
