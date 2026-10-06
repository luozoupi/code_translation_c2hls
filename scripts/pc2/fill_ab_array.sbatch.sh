#!/usr/bin/env bash
#SBATCH --job-name=fill-ab
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730/slurm/fill-%A_%a.out
#SBATCH --error=artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730/slurm/fill-%A_%a.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
if [[ -z "${C2HLS_PYTHON:-}" && -x "${_REPO_ROOT}/.venv/bin/python" ]]; then
  export C2HLS_PYTHON="${_REPO_ROOT}/.venv/bin/python"
fi
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730/slurm c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_COSIM_TRACE_LEVEL="${C2HLS_COSIM_TRACE_LEVEL:-none}"
export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_COSIM_EXTRA_ARGS="${C2HLS_COSIM_EXTRA_ARGS:--disable_deadlock_detection}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-21600}"
ulimit -s unlimited 2>/dev/null || true
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

INDEX="${SLURM_ARRAY_TASK_ID:-${C2HLS_FILL_AB_INDEX:-}}"
if [[ -z "${INDEX}" ]]; then
  echo "ERROR: set SLURM_ARRAY_TASK_ID or C2HLS_FILL_AB_INDEX" >&2
  exit 2
fi

pc2_log "fill_ab task index=${INDEX}"
FORCE_ARGS=()
if [[ "${C2HLS_FILL_AB_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi
"${C2HLS_PYTHON:-python3}" "${_SCRIPT_DIR}/run_fill_ab_one.py" --index "${INDEX}" "${FORCE_ARGS[@]}"
pc2_log "fill_ab finished index=${INDEX}"
