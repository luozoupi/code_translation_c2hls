#!/usr/bin/env bash
#SBATCH --job-name=c2hls-cosim1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=artifacts/pc2/flash_cosim/slurm/cosim-one-%j.out
#SBATCH --error=artifacts/pc2/flash_cosim/slurm/cosim-one-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/flash_cosim/slurm c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_RUN_COSIM=1
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-7200}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

RUN_ROOT="${C2HLS_FLASH_COSIM_RUN_ROOT:?set C2HLS_FLASH_COSIM_RUN_ROOT}"
CELL_ID="${C2HLS_FLASH_COSIM_CELL_ID:?set C2HLS_FLASH_COSIM_CELL_ID}"

pc2_log "flash_cosim single cell_id=${CELL_ID} run_root=${RUN_ROOT}"

"${C2HLS_PYTHON:-python3}" "${_SCRIPT_DIR}/run_flash_cosim_one.py" \
  --run-root "${RUN_ROOT}" \
  --cell-id "${CELL_ID}" \
  "$@"

pc2_log "flash_cosim finished cell_id=${CELL_ID}"
