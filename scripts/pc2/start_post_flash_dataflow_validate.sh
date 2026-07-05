#!/usr/bin/env bash
# Validate recovered post-flash DATAFLOW kernels (csim + csynth only, no LLM).
#
# Usage:
#   ./scripts/pc2/start_post_flash_dataflow_validate.sh
#   ./scripts/pc2/start_post_flash_dataflow_validate.sh --submit
#   ./scripts/pc2/start_post_flash_dataflow_validate.sh --benches hlsfactory_gemm,hlsfactory_atax
#   ./scripts/pc2/start_post_flash_dataflow_validate.sh --submit --force
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python3"
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-}"
SUBMIT=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --matrix-root) MATRIX_ROOT="$2"; shift 2 ;;
    --benches) BENCHES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}" --validate-recovered)
[[ -n "${BENCHES}" ]] && ARGS+=(--benches "${BENCHES}")
[[ "${FORCE}" -eq 1 ]] && ARGS+=(--force)

if [[ "${SUBMIT}" -eq 1 ]]; then
  account_args=()
  if [[ -n "${PC2_SLURM_ACCOUNT:-}" ]]; then
    account_args=(--account="${PC2_SLURM_ACCOUNT}")
  fi
  export POST_FLASH_MATRIX_ROOT="${MATRIX_ROOT}"
  export POST_FLASH_BENCHES="${BENCHES}"
  export POST_FLASH_FORCE="${FORCE}"
  job_id="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --export=ALL,POST_FLASH_MATRIX_ROOT,POST_FLASH_BENCHES,POST_FLASH_FORCE \
      "${account_args[@]}" \
      "${SCRIPT_DIR}/run_post_flash_dataflow_validate.sbatch.sh"
  )"
  pc2_log "submitted post-flash dataflow validate job ${job_id} matrix=${MATRIX_ROOT}"
  echo "${job_id}"
  exit 0
fi

exec "${PY}" scripts/pc2/run_post_flash_dataflow.py "${ARGS[@]}"
