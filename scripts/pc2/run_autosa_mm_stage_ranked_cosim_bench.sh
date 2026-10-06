#!/usr/bin/env bash
# Slurm wrapper: ranked cosim for one autosa_mm stage-box bucket or candidate.
# Env: C2HLS_STAGE_COSIM_OUT, C2HLS_STAGE_COSIM_SWEEP,
#      C2HLS_STAGE_COSIM_BUCKET or C2HLS_STAGE_COSIM_CANDIDATE
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

OUT="${C2HLS_STAGE_COSIM_OUT:?set C2HLS_STAGE_COSIM_OUT}"
SWEEP="${C2HLS_STAGE_COSIM_SWEEP:?set C2HLS_STAGE_COSIM_SWEEP}"
BUCKET="${C2HLS_STAGE_COSIM_BUCKET:-}"
CANDIDATE="${C2HLS_STAGE_COSIM_CANDIDATE:-}"
if [[ -z "${BUCKET}" && -z "${CANDIDATE}" ]]; then
  echo "set C2HLS_STAGE_COSIM_BUCKET or C2HLS_STAGE_COSIM_CANDIDATE" >&2
  exit 2
fi

export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_FLASH_COSIM_FULL_SIZE="${C2HLS_FLASH_COSIM_FULL_SIZE:-1}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_COSIM_TRACE_LEVEL="${C2HLS_COSIM_TRACE_LEVEL:-none}"
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${OUT}/bench}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

EXTRA=()
if [[ "${C2HLS_STAGE_COSIM_FORCE:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi

if [[ -n "${CANDIDATE}" ]]; then
  pc2_log "stage_ranked_cosim candidate=${CANDIDATE} out=${OUT}"
  "${PY}" "${SCRIPT_DIR}/autosa_mm_stage_ranked_cosim.py" \
    --sweep-root "${SWEEP}" \
    --out "${OUT}" \
    --candidate "${CANDIDATE}" \
    "${EXTRA[@]}"
  pc2_log "stage_ranked_cosim done candidate=${CANDIDATE}"
else
  pc2_log "stage_ranked_cosim bucket=${BUCKET} out=${OUT}"
  "${PY}" "${SCRIPT_DIR}/autosa_mm_stage_ranked_cosim.py" \
    --sweep-root "${SWEEP}" \
    --out "${OUT}" \
    --bucket "${BUCKET}" \
    "${EXTRA[@]}"
  pc2_log "stage_ranked_cosim done bucket=${BUCKET}"
fi
