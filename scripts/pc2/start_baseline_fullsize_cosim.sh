#!/usr/bin/env bash
# Full-size Vitis cosim for benchmarks_cosim/hlsfactory_*/hls_baseline_cosim.cpp (no LLM).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_BASELINE_COSIM_STAMP:-$(date +%Y%m%d_%H%M%S)_fixed_cosim_benchmark}"

export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/benchmarks_cosim}"
export C2HLS_FLASH_COSIM_ROOT="${C2HLS_ROOT}/artifacts/pc2/baseline_cosim"
export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-12:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-14400}"

python3 "${SCRIPT_DIR}/build_baseline_cosim_manifest.py" \
  --stamp "${STAMP}" \
  --full-size

exec "${SCRIPT_DIR}/submit_flash_cosim_all.sh" \
  --stamp "${STAMP}" \
  --full-size \
  --individual \
  "$@"
