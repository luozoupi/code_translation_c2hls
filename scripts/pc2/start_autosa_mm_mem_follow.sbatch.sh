#!/usr/bin/env bash
# Fresh 1-rep +mem DAG: flash/oneshot/DSE/stream/enf then mem_* siblings.
# Writes only the new stamp tree (default 20260919).
#SBATCH --job-name=vswmem
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH --output=artifacts/pc2/autosa_mm_variant_sweep_20260919/follow-%j.out
#SBATCH --error=artifacts/pc2/autosa_mm_variant_sweep_20260919/follow-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/autosa_mm_variant_sweep_20260919 artifacts/pc2/post_flash_mem_iter c2hls_tmp

export PYTHONUNBUFFERED=1
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_SWEEP_TMP_ROOT="${C2HLS_SWEEP_TMP_ROOT:-${C2HLS_TMP_ROOT}}"

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
if [[ -f "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

STAMP="${C2HLS_SWEEP_STAMP:-20260919}"
ENDPOINT="${C2HLS_SWEEP_ENDPOINT:-http://login5:18092/v1}"
MAX_INFLIGHT="${C2HLS_SWEEP_MAX_INFLIGHT:-32}"
REPS="${C2HLS_SWEEP_REPS:-1}"
WAVE="${C2HLS_SWEEP_WAVE:-flash,oneshot,dse,stream,enf,mem}"

pc2_log "variant-sweep +mem follow stamp=${STAMP} reps=${REPS} wave=${WAVE} endpoint=${ENDPOINT} max_inflight=${MAX_INFLIGHT}"

exec "${PY}" "${_SCRIPT_DIR}/autosa_mm_variant_sweep.py" \
  --follow \
  --submit \
  --wave "${WAVE}" \
  --stamp "${STAMP}" \
  --reps "${REPS}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --sleep 120 \
  --endpoint-url "${ENDPOINT}"
