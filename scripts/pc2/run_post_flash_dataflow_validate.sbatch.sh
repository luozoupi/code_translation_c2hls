#!/usr/bin/env bash
#SBATCH --job-name=pfd-validate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=artifacts/pc2/slurm-pfd-validate-%j.out
#SBATCH --error=artifacts/pc2/slurm-pfd-validate-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"

PY="${C2HLS_PYTHON:-python3}"
if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python3"
fi

MATRIX_ROOT="${POST_FLASH_MATRIX_ROOT:-artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548}"
BENCHES="${POST_FLASH_BENCHES:-}"
FORCE="${POST_FLASH_FORCE:-0}"

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}" --validate-recovered)
[[ -n "${BENCHES}" ]] && ARGS+=(--benches "${BENCHES}")
[[ "${FORCE}" == "1" ]] && ARGS+=(--force)

exec "${PY}" "${SCRIPT_DIR}/run_post_flash_dataflow.py" "${ARGS[@]}"
