#!/usr/bin/env bash
# Serial DSE+stream retry on the existing wave-1 campaign cells.
# Seeds hcl/catapult from phase-B, then one LLM at a time on login5:18092.
#
# Usage:
#   ./scripts/pc2/start_autosa_wave1_dse_stream_retry.sh --dry-run
#   ./scripts/pc2/start_autosa_wave1_dse_stream_retry.sh --submit --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf}"
DRY_RUN=0
SUBMIT=1
ENDPOINT_URL_ARG=""
FORCE=1
BENCHES="${C2HLS_POST_FLASH_BENCHES:-autosa_mm_hcl,autosa_mm_hcl_intel,autosa_mm_intel,autosa_mm_int16,autosa_mm_catapult,autosa_mm_getting_started}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix-root) shift; MATRIX_ROOT="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --dry-run) DRY_RUN=1; SUBMIT=0; shift ;;
    --submit) SUBMIT=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --stream-only) export C2HLS_WAVE1_STREAM_ONLY=1; export C2HLS_DSE_FORCE=0; shift ;;
    --no-submit) DRY_RUN=1; SUBMIT=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

export C2HLS_POST_FLASH_DSE=1
export C2HLS_POST_FLASH_STREAM=1
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_DSE_MODEL="${C2HLS_DSE_MODEL:-deepseek-v4-flash}"
export C2HLS_STREAM_MODEL="${C2HLS_STREAM_MODEL:-deepseek-v4-flash}"
export C2HLS_MODEL="${C2HLS_DSE_MODEL}"
export C2HLS_DSE_MAX_TOKENS="${C2HLS_DSE_MAX_TOKENS:-65536}"
export C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}"
export C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-3}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export C2HLS_DSE_FORCE="${C2HLS_DSE_FORCE:-1}"
export C2HLS_STREAM_FORCE=1

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ -f "${MATRIX_ROOT}/llm_endpoint.json" ]]; then
  ENDPOINT_URL="$(
    "${PY}" - <<PY
import json
from pathlib import Path
print(json.loads(Path("${MATRIX_ROOT}/llm_endpoint.json").read_text())["url"])
PY
  )"
else
  ENDPOINT_URL="${OPENAI_BASE_URL:-}"
fi

if [[ -n "${ENDPOINT_URL}" ]]; then
  export OPENAI_BASE_URL="${ENDPOINT_URL}"
  export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL}"
  export CHATHLS_API_BASE="${ENDPOINT_URL}"
fi

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
if [[ -f "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}}"

echo "=== wave1 DSE+stream retry (serial, recipe PE) ==="
echo "matrix_root=${MATRIX_ROOT}"
echo "benches=${BENCHES}"
echo "stream_only=${C2HLS_WAVE1_STREAM_ONLY:-0}"
echo "model=${C2HLS_MODEL} endpoint=${OPENAI_BASE_URL:-}"
echo "dse_tokens=${C2HLS_DSE_MAX_TOKENS} stream_tokens=${C2HLS_STREAM_MAX_TOKENS}"

export C2HLS_POST_FLASH_MATRIX_ROOT="${MATRIX_ROOT}"
export C2HLS_POST_FLASH_BENCHES="${BENCHES}"
# Slurm --export=A=x,B=y splits on commas, so never put the CSV on the sbatch line.
BENCHES_FILE="${MATRIX_ROOT}/wave1_retry_benches.txt"
mkdir -p "${MATRIX_ROOT}"
printf '%s\n' "${BENCHES}" | tr ',' '\n' | sed '/^$/d' > "${BENCHES_FILE}"
export C2HLS_POST_FLASH_BENCHES_FILE="${BENCHES_FILE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "benches_file=${BENCHES_FILE}"
  cat "${BENCHES_FILE}"
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/seed_wave1_phase_b_selected.py" \
    --matrix-root "${MATRIX_ROOT}" --benches autosa_mm_hcl,autosa_mm_catapult
  if [[ "${C2HLS_WAVE1_STREAM_ONLY:-0}" != "1" ]]; then
    "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_dse.py" --pc2 \
      --matrix-root "${MATRIX_ROOT}" --benches "${BENCHES}" --dry-run
  fi
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_stream.py" --pc2 \
    --matrix-root "${MATRIX_ROOT}" --benches "${BENCHES}" --dry-run
  exit 0
fi

mkdir -p artifacts/pc2/post_flash_dse
OUT_STAMP="$(date -u +%Y%m%d_%H%M%S)"
JOB="$(sbatch --parsable \
  --chdir="${C2HLS_ROOT}" \
  --export=ALL \
  "${SCRIPT_DIR}/post_flash_wave1_retry.sbatch.sh")"
echo "job=${JOB}"
echo "${JOB}" > "artifacts/pc2/post_flash_dse/slurm_job_wave1_retry_${OUT_STAMP}"
echo "${JOB}" > "${MATRIX_ROOT}/post_flash_wave1_retry_slurm_job_id"
