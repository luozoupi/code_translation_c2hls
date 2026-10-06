#!/usr/bin/env bash
#SBATCH --job-name=mmmem
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=artifacts/pc2/post_flash_mem_iter/%x-%j.out
#SBATCH --error=artifacts/pc2/post_flash_mem_iter/%x-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/post_flash_mem_iter c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_MEM_ITER=1
export C2HLS_MEM_ITER_ROUNDS="${C2HLS_MEM_ITER_ROUNDS:-50}"
export C2HLS_MEM_ITER_PROMPT_TOKENS="${C2HLS_MEM_ITER_PROMPT_TOKENS:-32768}"
export C2HLS_MEM_ITER_MAX_TOKENS="${C2HLS_MEM_ITER_MAX_TOKENS:-65536}"
export C2HLS_MEM_ITER_CONTEXT_TOKENS="${C2HLS_MEM_ITER_CONTEXT_TOKENS:-131072}"
export C2HLS_LLM_TIMEOUT="${C2HLS_LLM_TIMEOUT:-3600}"
export C2HLS_LLM_TIMEOUT_RETRIES="${C2HLS_LLM_TIMEOUT_RETRIES:-8}"
export C2HLS_LLM_RETRY_BACKOFF_S="${C2HLS_LLM_RETRY_BACKOFF_S:-30}"
export C2HLS_LLM_LOCK="${C2HLS_LLM_LOCK:-1}"
export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-${C2HLS_MEM_ITER_MAX_TOKENS}}"
export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-${C2HLS_MEM_ITER_MAX_TOKENS}}"
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_MODEL="${C2HLS_MEM_MODEL:-${C2HLS_MODEL:-deepseek-v4-flash}}"
export C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-3}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:?set C2HLS_POST_FLASH_MATRIX_ROOT}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-autosa_mm}"
FORCE="${C2HLS_MEM_ITER_FORCE:-0}"

pc2_log "post-flash mem-iter matrix=${MATRIX_ROOT} benches=${BENCHES} parent_family=${C2HLS_MEM_PARENT_FAMILY:-} model=${C2HLS_MODEL:-} endpoint=${OPENAI_BASE_URL:-} rounds=${C2HLS_MEM_ITER_ROUNDS} prompt_tokens=${C2HLS_MEM_ITER_PROMPT_TOKENS} max_tokens=${C2HLS_MEM_ITER_MAX_TOKENS} context_tokens=${C2HLS_MEM_ITER_CONTEXT_TOKENS}"

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}" --benches "${BENCHES}" --turns "${C2HLS_MEM_ITER_ROUNDS}")
if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
  ARGS+=(--force)
fi
if [[ -n "${C2HLS_MODEL:-}" ]]; then
  ARGS+=(--model "${C2HLS_MODEL}")
fi

"${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_mem_iter.py" "${ARGS[@]}"
pc2_log "post-flash mem-iter done"
