#!/usr/bin/env bash
#SBATCH --job-name=c2hls-cosim
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=artifacts/pc2/flash_cosim/slurm/cosim-%A_%a.out
#SBATCH --error=artifacts/pc2/flash_cosim/slurm/cosim-%A_%a.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# Prefer project venv (Python 3.11+) when present — login python3 may be 3.9.
if [[ -z "${C2HLS_PYTHON:-}" && -x "${_REPO_ROOT}/.venv/bin/python" ]]; then
  export C2HLS_PYTHON="${_REPO_ROOT}/.venv/bin/python"
fi
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/flash_cosim/slurm c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_RUN_COSIM=1
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-7200}"
# Large xelab/TB elaborations can SIGSEGV under the default stack limit.
ulimit -s unlimited 2>/dev/null || true
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

RUN_ROOT="${C2HLS_FLASH_COSIM_RUN_ROOT:?set C2HLS_FLASH_COSIM_RUN_ROOT}"
INDEX="${SLURM_ARRAY_TASK_ID:-${C2HLS_FLASH_COSIM_INDEX:-}}"

if [[ -z "${INDEX}" ]]; then
  echo "ERROR: set SLURM_ARRAY_TASK_ID or C2HLS_FLASH_COSIM_INDEX" >&2
  exit 2
fi

pc2_log "flash_cosim array task index=${INDEX} run_root=${RUN_ROOT}"

"${C2HLS_PYTHON:-python3}" "${_SCRIPT_DIR}/run_flash_cosim_one.py" \
  --run-root "${RUN_ROOT}" \
  --index "${INDEX}" \
  "$@"

pc2_log "flash_cosim finished index=${INDEX}"
