#!/usr/bin/env bash
# Multi-PE (DSE) on autosa_mm HLS baseline. Flash is not run.
# Seed: related_work/benchmarks/autosa_ready/autosa_mm/hls_baseline.cpp
# (same text as plain.cpp). Writes a new campaign; does not touch 20260830_mmflow.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_dse_from_baseline.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_dse_from_baseline.sh --submit --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --submit) shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

SEED="${C2HLS_ROOT}/related_work/benchmarks/autosa_ready/autosa_mm/hls_baseline.cpp"
if [[ ! -f "${SEED}" ]]; then
  echo "ERROR: missing baseline ${SEED}" >&2
  exit 2
fi

MATRIX_ROOT="${C2HLS_ROOT}/artifacts/pc2/dse_from_baseline_${STAMP}"
CELL_DIR="${MATRIX_ROOT}/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__dse__baseline"
mkdir -p "${CELL_DIR}" artifacts/pc2/post_flash_dse
cp -a "${SEED}" "${CELL_DIR}/autosa_mm_baseline.cpp"
cp -a "${SEED}" "${CELL_DIR}/autosa_mm_selected.cpp"
echo "baseline" > "${MATRIX_ROOT}/seed.txt"
echo "hls_baseline.cpp" > "${MATRIX_ROOT}/seed_file.txt"
echo "0" > "${MATRIX_ROOT}/flash.txt"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

export C2HLS_POST_FLASH_DSE=1
export C2HLS_DSE_CHAIN_FLASH=0
export C2HLS_DSE_SOURCE_ROLE=baseline
export C2HLS_DSE_FORCE=1
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_DSE_MODEL="${C2HLS_DSE_MODEL:-deepseek-v4-flash}"
export C2HLS_MODEL="${C2HLS_DSE_MODEL}"
export C2HLS_DSE_MAX_TOKENS="${C2HLS_DSE_MAX_TOKENS:-65536}"
export C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-3}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export C2HLS_POST_FLASH_MATRIX_ROOT="${MATRIX_ROOT}"
export C2HLS_POST_FLASH_BENCHES=autosa_mm

if [[ -z "${ENDPOINT_URL_ARG}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "WARNING: --endpoint-url not given; dry-run only" >&2
  else
    echo "ERROR: --endpoint-url is required (e.g. http://login5:18092/v1)." >&2
    exit 2
  fi
else
  export OPENAI_BASE_URL="${ENDPOINT_URL_ARG}"
  export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL_ARG}"
  export CHATHLS_API_BASE="${ENDPOINT_URL_ARG}"
fi

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
if [[ -f "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}}"

echo "=== autosa_mm multi-PE from HLS baseline (no flash) ==="
echo "stamp=${STAMP}"
echo "matrix_root=${MATRIX_ROOT}"
echo "seed=${SEED}"
echo "source_role=${C2HLS_DSE_SOURCE_ROLE}"
echo "model=${C2HLS_MODEL} endpoint=${OPENAI_BASE_URL:-}"
echo "skills=post_flash_dse_pe_skill_entries.json + PE recipe 16x4"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exec "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_dse.py" --pc2 \
    --matrix-root "${MATRIX_ROOT}" --benches autosa_mm --dry-run
fi

JOB="$(sbatch --parsable \
  --job-name=mmdsebl \
  --chdir="${C2HLS_ROOT}" \
  --export=ALL,C2HLS_ROOT="${C2HLS_ROOT}",C2HLS_PYTHON="${PY}",C2HLS_POST_FLASH_MATRIX_ROOT="${MATRIX_ROOT}",C2HLS_POST_FLASH_BENCHES=autosa_mm,C2HLS_DSE_FORCE=1,C2HLS_DSE_SOURCE_ROLE=baseline,C2HLS_DSE_CHAIN_FLASH=0,C2HLS_POST_FLASH_STREAM=0,C2HLS_DSE_MODEL="${C2HLS_DSE_MODEL}",C2HLS_MODEL="${C2HLS_MODEL}",C2HLS_DSE_MAX_TOKENS="${C2HLS_DSE_MAX_TOKENS}",C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES}",OPENAI_BASE_URL="${OPENAI_BASE_URL:-}",C2HLS_OPENAI_HOSTED_URL="${C2HLS_OPENAI_HOSTED_URL:-}",CHATHLS_API_BASE="${CHATHLS_API_BASE:-}",OPENAI_API_KEY="${OPENAI_API_KEY:-}",CHATHLS_API_KEY="${CHATHLS_API_KEY:-}",PYTHONPATH="${PYTHONPATH}" \
  "${SCRIPT_DIR}/post_flash_dse.sbatch.sh")"
echo "job=${JOB}"
echo "${JOB}" > "artifacts/pc2/post_flash_dse/slurm_job_dse_baseline_${STAMP}"
echo "${JOB}" > "${MATRIX_ROOT}/post_flash_dse_slurm_job_id"
