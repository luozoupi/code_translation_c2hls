#!/usr/bin/env bash
#SBATCH --job-name=mmdsev2
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/pc2/post_flash_dse/%x-%j.out
#SBATCH --error=artifacts/pc2/post_flash_dse/%x-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/post_flash_dse c2hls_tmp

export C2HLS_SITE=pc2
if [[ -n "${SLURM_JOB_ID:-}" && -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
  export C2HLS_TMP_ROOT="${TMPDIR}/c2hls_tmp"
else
  export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
fi
mkdir -p "${C2HLS_TMP_ROOT}"
export C2HLS_DSE_V2=1
export C2HLS_DSE_V2_CHAIN_FLASH=1
export C2HLS_POST_FLASH_DSE=0
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_STREAM_CHAIN_FLASH=0
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_MODEL="${C2HLS_DSE_MODEL:-${C2HLS_MODEL:-deepseek-v4-flash}}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:?set C2HLS_POST_FLASH_MATRIX_ROOT}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-autosa_mm}"

pc2_log "post-flash DSE v2 matrix=${MATRIX_ROOT} benches=${BENCHES} model=${C2HLS_MODEL:-} endpoint=${OPENAI_BASE_URL:-} only_simd=${C2HLS_DSE_V2_ONLY_SIMD:-}"

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}" --benches "${BENCHES}")
if [[ "${C2HLS_DSE_FORCE:-0}" == "1" || "${C2HLS_DSE_FORCE:-}" == "true" ]]; then
  ARGS+=(--no-skip-existing)
fi

"${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_dse_v2.py" "${ARGS[@]}"
pc2_log "post-flash DSE v2 done"
