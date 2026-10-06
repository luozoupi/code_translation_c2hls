#!/usr/bin/env bash
#SBATCH --job-name=peovlp
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/pc2/post_flash_overlap/%x-%j.out
#SBATCH --error=artifacts/pc2/post_flash_overlap/%x-%j.err

# Serial overlap ablation: PE array + fuse + ping-pong DATAFLOW on *_dse.cpp.
set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/post_flash_overlap c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_MODEL="${C2HLS_OVERLAP_MODEL:-${C2HLS_MODEL:-deepseek-v4-flash}}"
export C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

JOBS_FILE="${C2HLS_OVERLAP_JOBS_FILE:?set C2HLS_OVERLAP_JOBS_FILE}"
rc=0
while IFS=$'\t' read -r matrix benches || [[ -n "${matrix:-}" ]]; do
  [[ -z "${matrix:-}" || "${matrix}" =~ ^# ]] && continue
  pc2_log "overlap matrix=${matrix} benches=${benches} model=${C2HLS_MODEL}"
  set +e
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_overlap.py" --pc2 \
    --matrix-root "${matrix}" --benches "${benches}" --force \
    ${C2HLS_MODEL:+--model "${C2HLS_MODEL}"}
  step_rc=$?
  set -e
  pc2_log "overlap exit=${step_rc} matrix=${matrix}"
  if [[ "${step_rc}" -ne 0 ]]; then
    rc=1
  fi
done < "${JOBS_FILE}"
exit "${rc}"
