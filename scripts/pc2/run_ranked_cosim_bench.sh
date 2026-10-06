#!/usr/bin/env bash
# Slurm-friendly wrapper: ranked cosim for one cell/side.
# Env: BATCH_PARALLEL_CAMPAIGN_ROOT, C2HLS_RANKED_COSIM_CELL_DIR, C2HLS_RANKED_COSIM_BENCH,
#      C2HLS_RANKED_COSIM_SIDE=flash|dataflow
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
CELL_DIR="${C2HLS_RANKED_COSIM_CELL_DIR:?set C2HLS_RANKED_COSIM_CELL_DIR}"
BENCH="${C2HLS_RANKED_COSIM_BENCH:?set C2HLS_RANKED_COSIM_BENCH}"
SIDE="${C2HLS_RANKED_COSIM_SIDE:?set C2HLS_RANKED_COSIM_SIDE}"

export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_FLASH_COSIM_FULL_SIZE="${C2HLS_FLASH_COSIM_FULL_SIZE:-1}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_COSIM_TRACE_LEVEL="${C2HLS_COSIM_TRACE_LEVEL:-none}"
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/benchmarks_cosim}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

EXTRA=()
if [[ "${C2HLS_RANKED_COSIM_FORCE:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi
if [[ "${C2HLS_RANKED_COSIM_REBUILD_RANK:-0}" == "1" ]]; then
  EXTRA+=(--rebuild-rank)
fi

pc2_log "ranked_cosim side=${SIDE} bench=${BENCH} cell=${CELL_DIR}"
"${PY}" "${SCRIPT_DIR}/run_ranked_cosim_cell.py" \
  --cell-dir "${CELL_DIR}" \
  --bench "${BENCH}" \
  --side "${SIDE}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  "${EXTRA[@]}"
pc2_log "ranked_cosim done side=${SIDE} bench=${BENCH}"
