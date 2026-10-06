#!/usr/bin/env bash
#SBATCH --job-name=autosa-hls
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=artifacts/pc2/autosa_dse_hls_validate/%x-%j.out
#SBATCH --error=artifacts/pc2/autosa_dse_hls_validate/%x-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"

KERNEL_ID="${AUTOSA_DSE_KERNEL_ID:?set AUTOSA_DSE_KERNEL_ID}"
RUN_ROOT="${AUTOSA_DSE_HLS_RUN_ROOT:?set AUTOSA_DSE_HLS_RUN_ROOT}"
SOURCES_ROOT="${AUTOSA_DSE_SOURCES_ROOT:-${_REPO_ROOT}/AutoSA_sources}"
WALLTIME="${AUTOSA_DSE_KERNEL_WALLTIME:-}"

if [[ -n "${WALLTIME}" ]]; then
  :
fi

cd "${C2HLS_ROOT}"
mkdir -p "${RUN_ROOT}" artifacts/pc2/autosa_dse_hls_validate c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_COSIM_TIMEOUT="${AUTOSA_DSE_COSIM_TIMEOUT:-${C2HLS_COSIM_TIMEOUT:-7200}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

pc2_log "autosa_dse_hls_validate kernel=${KERNEL_ID} run_root=${RUN_ROOT}"

"${C2HLS_PYTHON:-python3}" "${_SCRIPT_DIR}/run_autosa_dse_kernel_hls.py" \
  --kernel-id "${KERNEL_ID}" \
  --sources-root "${SOURCES_ROOT}" \
  --run-root "${RUN_ROOT}/kernels/${KERNEL_ID}"

pc2_log "autosa_dse_hls_validate done kernel=${KERNEL_ID}"
