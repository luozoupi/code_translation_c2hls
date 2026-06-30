#!/usr/bin/env bash
# End-to-end fixed-corpus multistep campaign (aav_n v1):
#   1) optional benchmarks_cosim gold refresh
#   2) pipelined multistep LLM+Vitis (record-flow artifacts)
#   3) full-size cosim: phase_b + each step + selected
#   4) export / validate combined JSONL
#
# Usage:
#   ./scripts/pc2/run_fixed_cosim_multistep_full_pipeline.sh --dry-run
#   ./scripts/pc2/run_fixed_cosim_multistep_full_pipeline.sh --pilot
#   ./scripts/pc2/run_fixed_cosim_multistep_full_pipeline.sh --skip-multistep --multistep-stamp 20260626_fixed_cosim_multistep_pipelined
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
PIPELINE_LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/fixed_cosim_multistep_full.log"

MULTISTEP_WALLTIME="${PC2_MULTISTEP_FULL_WALLTIME:-36:00:00}"
PILOT_WALLTIME="${PC2_MULTISTEP_PILOT_WALLTIME:-12:00:00}"
COSIM_SLURM_WALLTIME="${PC2_COSIM_PIPELINE_WALLTIME:-13:00:00}"
COSIM_TIMEOUT_SEC="${C2HLS_COSIM_PIPELINE_TIMEOUT_SEC:-43200}"
JSONL_WATCHER_WALLTIME="${PC2_MULTISTEP_JSONL_WATCHER_WALLTIME:-24:00:00}"
POLL_SEC="${PC2_PIPELINE_POLL_SEC:-60}"

MULTISTEP_STAMP="${C2HLS_MULTISTEP_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_multistep}"
DRY_RUN=0
PILOT=0
SKIP_PREPARE=0
SKIP_MULTISTEP=0
SKIP_COSIM=0
SKIP_JSONL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multistep-stamp) shift; MULTISTEP_STAMP="$1"; shift ;;
    --pilot) PILOT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --skip-multistep) SKIP_MULTISTEP=1; shift ;;
    --skip-cosim) SKIP_COSIM=1; shift ;;
    --skip-jsonl) SKIP_JSONL=1; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

DATE_PREFIX="$(printf '%s' "${MULTISTEP_STAMP}" | grep -oE '[0-9]{8}' | head -1)"
if [[ -z "${DATE_PREFIX}" ]]; then
  echo "ERROR: --multistep-stamp must contain YYYYMMDD (got: ${MULTISTEP_STAMP})" >&2
  exit 2
fi

STAMP_SUFFIX="${MULTISTEP_STAMP}"
if [[ "${STAMP_SUFFIX}" != *_pipelined ]]; then
  STAMP_SUFFIX="${MULTISTEP_STAMP}_pipelined"
fi

ARTIFACT_GLOB="multistep_fixed_cosim_aav_n_${STAMP_SUFFIX}"
COSIM_STAMP="fixed_cosim_multistep_${DATE_PREFIX}"
COSIM_ROOT="${C2HLS_ROOT}/artifacts/pc2/multistep_cosim/${COSIM_STAMP}"
JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_u280_${DATE_PREFIX}.jsonl"
BASELINE_JSONL="${C2HLS_ROOT}/misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"

mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

plog() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

plog "=== fixed cosim multistep full pipeline ==="
plog "multistep_stamp=${MULTISTEP_STAMP} artifact_glob=${ARTIFACT_GLOB}"
plog "cosim_stamp=${COSIM_STAMP} jsonl_out=${JSONL_OUT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  plog "dry-run: prepare=${SKIP_PREPARE} multistep=${SKIP_MULTISTEP} cosim=${SKIP_COSIM} jsonl=${SKIP_JSONL} pilot=${PILOT}"
  exit 0
fi

if [[ "${SKIP_MULTISTEP}" -eq 0 ]]; then
  if [[ "${PILOT}" -eq 1 ]]; then
    export PC2_FORCE_WALLTIME="${PILOT_WALLTIME}"
    "${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" --pilot --stamp "${MULTISTEP_STAMP}"
  else
    export PC2_FORCE_WALLTIME="${MULTISTEP_WALLTIME}"
    "${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" --stamp "${MULTISTEP_STAMP}"
  fi
fi

if [[ "${SKIP_COSIM}" -eq 0 ]]; then
  export C2HLS_MULTISTEP_COSIM_STAMP="${COSIM_STAMP}"
  export C2HLS_MULTISTEP_COSIM_ARTIFACT_GLOB="${ARTIFACT_GLOB}"
  export PC2_COSIM_WALLTIME="${COSIM_SLURM_WALLTIME}"
  export C2HLS_COSIM_TIMEOUT="${COSIM_TIMEOUT_SEC}"
  export C2HLS_FLASH_COSIM_FULL_SIZE=1
  export C2HLS_MULTISTEP_COSIM_FULL_SIZE=1
  "${SCRIPT_DIR}/submit_multistep_cosim_all.sh" --stamp "${COSIM_STAMP}" --artifact-glob "${ARTIFACT_GLOB}"
fi

if [[ "${SKIP_JSONL}" -eq 0 ]]; then
  plog "waiting for cosim then exporting JSONL"
  export C2HLS_MULTISTEP_JSONL_WATCHER_WALLTIME="${JSONL_WATCHER_WALLTIME}"
  "${SCRIPT_DIR}/wait_multistep_cosim_export_jsonl.sh" \
    --multistep-stamp "${STAMP_SUFFIX}" \
    --cosim-stamp "${COSIM_STAMP}" \
    --output "${JSONL_OUT}" \
    --baseline-jsonl "${BASELINE_JSONL}" \
    --cosim-walltime "${COSIM_SLURM_WALLTIME}"
fi

plog "pipeline complete"
