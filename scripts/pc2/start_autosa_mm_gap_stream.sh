#!/usr/bin/env bash
# Post-DSE stream/I/O on an existing autosa_mm flash+DSE cell (DeepSeek-v4-flash stamp).
# Does not re-run flash or DSE. Uses the dedicated PE/stream skills.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_gap_stream.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_gap_stream.sh --submit
#   ./scripts/pc2/start_autosa_mm_gap_stream.sh --submit --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_mm_gap_stream.sh --submit --matrix-root artifacts/pc2/... --dependency afterok:JOBID
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-20260818_mm_gap_flash_ds_v4f}"
MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_${STAMP}}"
DRY_RUN=0
SUBMIT=1
ENDPOINT_URL_ARG=""
FORCE=0
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; MATRIX_ROOT="artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_${STAMP}"; shift ;;
    --matrix-root) shift; MATRIX_ROOT="$1"; shift ;;
    --dry-run) DRY_RUN=1; SUBMIT=0; shift ;;
    --submit) SUBMIT=1; shift ;;
    --force) FORCE=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --dependency) shift; DEPENDENCY="$1"; shift ;;
    --no-submit) DRY_RUN=1; SUBMIT=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

export C2HLS_POST_FLASH_STREAM=1
export C2HLS_STREAM_CHAIN_FLASH=1
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_STREAM_MODEL="${C2HLS_STREAM_MODEL:-deepseek-v4-flash}"
export C2HLS_MODEL="${C2HLS_STREAM_MODEL}"
export C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}"
export C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-3}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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

echo "=== autosa_mm post-DSE stream/I/O ==="
echo "matrix_root=${MATRIX_ROOT}"
echo "model=${C2HLS_MODEL} endpoint=${OPENAI_BASE_URL:-}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "skills=post_flash_stream_pe_io_skill_entries.json"
echo "max_tokens=${C2HLS_STREAM_MAX_TOKENS} force=${FORCE}"

export C2HLS_POST_FLASH_MATRIX_ROOT="${MATRIX_ROOT}"
export C2HLS_POST_FLASH_BENCHES=autosa_mm
export C2HLS_STREAM_FORCE=0
[[ "${FORCE}" -eq 1 ]] && export C2HLS_STREAM_FORCE=1

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exec "${SCRIPT_DIR}/start_post_flash_stream.sh" --dry-run --matrix-root "${MATRIX_ROOT}" --benches autosa_mm --no-borrow-gpu
fi

mkdir -p artifacts/pc2/post_flash_stream
OUT_STAMP="$(date -u +%Y%m%d_%H%M%S)"
SBATCH_EXTRA=()
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_EXTRA+=(--dependency="${DEPENDENCY}")
fi
JOB="$(sbatch --parsable \
  --chdir="${C2HLS_ROOT}" \
  "${SBATCH_EXTRA[@]}" \
  --export=ALL,C2HLS_ROOT="${C2HLS_ROOT}",C2HLS_PYTHON="${PY}",C2HLS_POST_FLASH_MATRIX_ROOT="${MATRIX_ROOT}",C2HLS_POST_FLASH_BENCHES=autosa_mm,C2HLS_STREAM_FORCE="${C2HLS_STREAM_FORCE}",C2HLS_STREAM_MODEL="${C2HLS_STREAM_MODEL}",C2HLS_MODEL="${C2HLS_MODEL}",C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}",C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-3}",OPENAI_BASE_URL="${OPENAI_BASE_URL:-}",C2HLS_OPENAI_HOSTED_URL="${C2HLS_OPENAI_HOSTED_URL:-}",CHATHLS_API_BASE="${CHATHLS_API_BASE:-}",OPENAI_API_KEY="${OPENAI_API_KEY:-}",CHATHLS_API_KEY="${CHATHLS_API_KEY:-}",PYTHONPATH="${PYTHONPATH}" \
  "${SCRIPT_DIR}/post_flash_stream.sbatch.sh")"
echo "job=${JOB}"
echo "matrix_root=${MATRIX_ROOT}"
echo "${JOB}" > "artifacts/pc2/post_flash_stream/slurm_job_${OUT_STAMP}"
echo "${JOB}" > "${MATRIX_ROOT}/post_flash_stream_slurm_job_id"
