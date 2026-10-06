#!/usr/bin/env bash
#SBATCH --job-name=mmpragdse
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=artifacts/pc2/autosa_mm_agent_pragma_dse/%x-%j.out
#SBATCH --error=artifacts/pc2/autosa_mm_agent_pragma_dse/%x-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/autosa_mm_agent_pragma_dse c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-1800}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

OUT="${AUTOSA_MM_PRAGMA_DSE_OUT:?set AUTOSA_MM_PRAGMA_DSE_OUT}"
SEED="${AUTOSA_MM_PRAGMA_DSE_SEED:-}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

pc2_log "autosa_mm pragma-only DSE out=${OUT}"

ARGS=(--out "${OUT}" --clock-ns "${C2HLS_CLOCK_NS}" --part "${C2HLS_PART}")
if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

"${PY}" "${C2HLS_ROOT}/scripts/autosa_mm_agent_pragma_dse.py" "${ARGS[@]}"
pc2_log "autosa_mm pragma-only DSE done"
