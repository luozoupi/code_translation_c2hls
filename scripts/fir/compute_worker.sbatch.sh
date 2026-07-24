#!/usr/bin/env bash
#SBATCH --job-name=c2hls-fir-vitis
#SBATCH --output=artifacts/fir/slurm-compute-%j.out
#SBATCH --error=artifacts/fir/slurm-compute-%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/fir"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}" "${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export PYTHONPATH="${C2HLS_ROOT}:${C2HLS_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

fir_log "compute job ${SLURM_JOB_ID:-?} starting on $(hostname -s)"

_on_term() {
  fir_log "compute job received termination signal; marking interrupted"
  fir_session_py set compute_state interrupted || true
  exit 143
}
trap _on_term USR1 TERM

export C2HLS_SITE=fir

if command -v module >/dev/null 2>&1 && [[ -n "${FIR_COMPUTE_MODULES:-}" ]]; then
  module purge 2>/dev/null || true
  # shellcheck disable=SC2086
  module load ${FIR_COMPUTE_MODULES}
fi
unset LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true

if [[ -n "${FIR_COMPUTE_VENV:-}" && -f "${FIR_COMPUTE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${FIR_COMPUTE_VENV}/bin/activate"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
fir_setup_vitis_env

if ! command -v vitis-run >/dev/null 2>&1; then
  fir_log "ERROR: vitis-run not on PATH after fir_setup_vitis_env"
  fir_session_py set compute_state failed
  fir_session_py set last_error '"vitis-run missing on compute node"'
  exit 2
fi

fir_log "vitis-run=$(command -v vitis-run) sif=${XILINX_SIF:-<unset>}"

fir_session_py set compute_state waiting_for_llm

WORKER_CMD="${FIR_WORKER_CMD}"
if [[ -f "${FIR_SESSION_FILE}" ]]; then
  stored="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${FIR_SESSION_FILE}').read_text()).get('worker_cmd',''))
" 2>/dev/null || true)"
  if [[ -n "${stored}" ]]; then
    WORKER_CMD="${stored}"
  fi
fi

GPU_JOB_ID="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${FIR_SESSION_FILE}').read_text()).get('gpu_job_id') or '')
" 2>/dev/null || true)"

fir_log "compute allocated on $(hostname -s); waiting for gpu+llm before worker"

wait_llm=$((SECONDS + FIR_COMPUTE_LLM_WAIT_SEC))
while (( SECONDS < wait_llm )); do
  if fir_gpu_serving "${GPU_JOB_ID}"; then
    break
  fi
  if [[ -n "${GPU_JOB_ID}" ]] && ! fir_job_is_running "${GPU_JOB_ID}"; then
    fir_log "ERROR: gpu job ${GPU_JOB_ID} not running after compute allocation"
    fir_session_py set compute_state failed
    fir_session_py set last_error '"gpu job ended before worker start"'
    exit 2
  fi
  sleep 15
done

if ! fir_gpu_serving "${GPU_JOB_ID}"; then
  fir_log "ERROR: gpu+llm not ready within ${FIR_COMPUTE_LLM_WAIT_SEC}s"
  fir_session_py set compute_state failed
  fir_session_py set last_error '"llm endpoint not ready while gpu running"'
  exit 2
fi

fir_log "gpu serving and compute allocated — starting c2hls worker"

URL="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${FIR_ENDPOINT_FILE}').read_text())['url'])
")"
export OPENAI_BASE_URL="${URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${FIR_ENDPOINT_FILE}').read_text()).get('model',''))
" 2>/dev/null || true)"
if [[ -n "${MODEL}" ]]; then
  export C2HLS_MODEL="${MODEL}"
fi

fir_log "using LLM OPENAI_BASE_URL=${OPENAI_BASE_URL} model=${C2HLS_MODEL:-<unset>}"
fir_session_py set compute_state running

fir_log "running worker: ${WORKER_CMD}"
# shellcheck disable=SC2086
set +e
eval ${WORKER_CMD}
worker_rc=$?
set -e

if [[ "${worker_rc}" -eq 0 ]]; then
  fir_session_py set compute_state completed
  fir_log "compute worker finished successfully (rc=0)"
else
  fir_session_py set compute_state failed
  fir_session_py set last_error "\"worker exited rc=${worker_rc}\""
  fir_log "compute worker failed rc=${worker_rc}"
fi
exit "${worker_rc}"
