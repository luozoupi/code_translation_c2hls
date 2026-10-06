#!/usr/bin/env bash
#SBATCH --job-name=n1024fg
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/pc2/focus_group/%x-%j.out
#SBATCH --error=artifacts/pc2/focus_group/%x-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/focus_group c2hls_tmp

export C2HLS_SITE=pc2
if [[ -n "${SLURM_JOB_ID:-}" && -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
  export C2HLS_TMP_ROOT="${TMPDIR}/c2hls_tmp"
else
  export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
fi
mkdir -p "${C2HLS_TMP_ROOT}"

export C2HLS_FOCUS_GROUP=1
export C2HLS_FOCUS_GROUP_ROUNDS="${C2HLS_FOCUS_GROUP_ROUNDS:-2}"
export C2HLS_FOCUS_GROUP_REPAIR_ROUNDS="${C2HLS_FOCUS_GROUP_REPAIR_ROUNDS:-3}"
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_LLM_TIMEOUT="${C2HLS_LLM_TIMEOUT:-3600}"
export C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-1}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-86400}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-86400}"
export C2HLS_MODEL="${C2HLS_FOCUS_MODEL:-${C2HLS_MODEL:-deepseek-v4-flash}}"
case "${C2HLS_MODEL}" in
  deepseek|deepseek-*) ;;
  *) export C2HLS_MODEL=deepseek-v4-flash ;;
esac
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

KERNEL_DIR="${C2HLS_FOCUS_KERNEL_DIR:?set C2HLS_FOCUS_KERNEL_DIR}"
BENCH_DIR="${C2HLS_FOCUS_BENCH_DIR:?set C2HLS_FOCUS_BENCH_DIR}"
BENCH="${C2HLS_FOCUS_BENCH:-autosa_mm}"

pc2_log "focus-group bench=${BENCH} kernel_dir=${KERNEL_DIR} bench_dir=${BENCH_DIR} model=${C2HLS_MODEL} rounds=${C2HLS_FOCUS_GROUP_ROUNDS}"

ARGS=(--pc2 --kernel-dir "${KERNEL_DIR}" --bench "${BENCH}" --bench-dir "${BENCH_DIR}")
if [[ "${C2HLS_FOCUS_FORCE:-0}" == "1" || "${C2HLS_FOCUS_FORCE:-}" == "true" ]]; then
  ARGS+=(--force)
fi

"${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_focus_group.py" "${ARGS[@]}"
pc2_log "focus-group done"
