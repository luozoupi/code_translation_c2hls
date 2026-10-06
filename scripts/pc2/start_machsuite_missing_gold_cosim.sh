#!/usr/bin/env bash
# Option C gold fill: cosim hls_baseline.cpp for MachSuite benches missing gold cycles.
# Default trio: aes_table, aes_tableless, backprop.
#
# Usage:
#   ./scripts/pc2/start_machsuite_missing_gold_cosim.sh [--stamp STAMP] [--dry-run]
#   ./scripts/pc2/start_machsuite_missing_gold_cosim.sh --bench machsuite_viterbi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_FLASH_COSIM_STAMP:-$(date -u +%Y%m%d_%H%M%S)_machsuite_gold_fill}"
DRY_RUN=0
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --bench) shift; BENCH_ARGS+=(--bench "$1"); shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export C2HLS_FLASH_COSIM_ROOT="${C2HLS_ROOT}/artifacts/pc2/machsuite_gold_cosim_fill"
export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-12:00:00}"

MANIFEST_ARGS=(--stamp "${STAMP}" --full-size --run-root "${C2HLS_FLASH_COSIM_ROOT}")
if [[ "${#BENCH_ARGS[@]}" -gt 0 ]]; then
  MANIFEST_ARGS+=("${BENCH_ARGS[@]}")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  MANIFEST_ARGS+=(--dry-run)
fi

"${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/build_machsuite_gold_cosim_manifest.py" \
  "${MANIFEST_ARGS[@]}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok stamp=${STAMP}"
  exit 0
fi

exec "${SCRIPT_DIR}/submit_flash_cosim_all.sh" \
  --stamp "${STAMP}" \
  --full-size \
  --individual \
  --skip-verify
