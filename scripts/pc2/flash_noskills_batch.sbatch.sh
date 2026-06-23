#!/usr/bin/env bash
#SBATCH --job-name=c2hls-flash-off
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=artifacts/pc2/slurm-flash-noskills-%j.out
#SBATCH --error=artifacts/pc2/slurm-flash-noskills-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2 c2hls_tmp

export C2HLS_SITE=pc2
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "ERROR: set OPENAI_BASE_URL to the GPU vLLM endpoint (e.g. http://gpu-node:8000/v1)" >&2
  exit 2
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

STAMP="${C2HLS_FLASH_NOSKILLS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export C2HLS_FLASH_NOSKILLS_STAMP="${STAMP}"

BENCHES="${C2HLS_FLASH_NOSKILLS_BENCHES:-hlsfactory_trmm,hlsfactory_trisolv,hlsfactory_symm,hlsfactory_3mm,hlsfactory_2mm,hlsfactory_gemm,hlsfactory_jacobi-2d}"

pc2_log "flash_noskills batch starting stamp=${STAMP} benches=${BENCHES}"
pc2_log "OPENAI_BASE_URL=${OPENAI_BASE_URL}"

"${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/run_flash_noskills_batch.py" --pc2 \
  --stamp "${STAMP}" \
  --benches "${BENCHES}" \
  "$@"

pc2_log "flash_noskills batch finished stamp=${STAMP}"
