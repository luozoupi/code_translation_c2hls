#!/usr/bin/env bash
# Post-flash memory parallelism on an existing flash matrix (2x and 4x).
#
# Usage:
#   ./scripts/pc2/start_post_flash_mem_parallel.sh --dry-run
#   ./scripts/pc2/start_post_flash_mem_parallel.sh --submit
#   ./scripts/pc2/start_post_flash_mem_parallel.sh --submit --benches hlsfactory_gemm
#
# Default matrix: flash_all_new_skills_avoids_global_20260623_024548
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
if [[ -z "${C2HLS_PYTHON:-}" && -n "${PC2_VLLM_VENV:-}" && -x "${PC2_VLLM_VENV}/bin/python3" ]]; then
  PY="${PC2_VLLM_VENV}/bin/python3"
fi
MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-}"
SUBMIT=0
DRY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --matrix-root) MATRIX_ROOT="$2"; shift 2 ;;
    --benches) BENCHES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}")
[[ -n "${BENCHES}" ]] && ARGS+=(--benches "${BENCHES}")
[[ "${DRY}" -eq 1 ]] && ARGS+=(--dry-run)
[[ "${FORCE}" -eq 1 ]] && ARGS+=(--force)

if [[ "${SUBMIT}" -eq 1 ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SESSION_ID="post_flash_mem_parallel_${STAMP}"
  LOG="${C2HLS_ROOT}/artifacts/pc2/sessions/${SESSION_ID}"
  mkdir -p "${LOG}"
  LLM_URL="${OPENAI_BASE_URL:-http://gpu1024:8000/v1}"
  sbatch --job-name="post-flash-mp" \
    --account="${PC2_SLURM_ACCOUNT}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task="${PC2_COMPUTE_CPUS:-16}" \
    --mem="${PC2_COMPUTE_MEM:-64G}" \
    --time="${PC2_WALLTIME}" \
    --output="${LOG}/slurm-%j.out" \
    --error="${LOG}/slurm-%j.err" \
    --wrap="cd '${C2HLS_ROOT}' && source scripts/source_local_env.sh && export C2HLS_SITE=pc2 OPENAI_BASE_URL='${LLM_URL}' OPENAI_API_KEY=EMPTY && ${PY} scripts/pc2/run_post_flash_mem_parallel.py ${ARGS[*]}"
  echo "submitted post-flash mem parallel -> ${LOG}"
else
  exec "${PY}" scripts/pc2/run_post_flash_mem_parallel.py "${ARGS[@]}"
fi
