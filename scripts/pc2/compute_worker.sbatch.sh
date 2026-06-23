#!/usr/bin/env bash
#SBATCH --job-name=c2hls-vitis
#SBATCH --output=artifacts/pc2/slurm-compute-%j.out
#SBATCH --error=artifacts/pc2/slurm-compute-%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}" c2hls_tmp

pc2_log "compute job ${SLURM_JOB_ID:-?} starting on $(hostname -s)"

_on_term() {
  pc2_log "compute job received termination signal; marking interrupted"
  pc2_session_py set compute_state interrupted || true
  exit 143
}
trap _on_term USR1 TERM

export C2HLS_SITE=pc2
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

pc2_session_py set compute_state waiting_for_llm

WORKER_CMD="${PC2_WORKER_CMD}"
if [[ -f "${PC2_SESSION_FILE}" ]]; then
  stored="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${PC2_SESSION_FILE}').read_text()).get('worker_cmd',''))
" 2>/dev/null || true)"
  if [[ -n "${stored}" ]]; then
    WORKER_CMD="${stored}"
  fi
fi

GPU_JOB_ID="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${PC2_SESSION_FILE}').read_text()).get('gpu_job_id') or '')
" 2>/dev/null || true)"

pc2_log "compute allocated on $(hostname -s); waiting for gpu+llm before worker"

wait_llm=$((SECONDS + PC2_COMPUTE_LLM_WAIT_SEC))
while (( SECONDS < wait_llm )); do
  if pc2_gpu_serving "${GPU_JOB_ID}"; then
    break
  fi
  if [[ -n "${GPU_JOB_ID}" ]] && ! pc2_job_is_running "${GPU_JOB_ID}"; then
    pc2_log "ERROR: gpu job ${GPU_JOB_ID} not running after compute allocation"
    pc2_session_py set compute_state failed
    pc2_session_py set last_error '"gpu job ended before worker start"'
    exit 2
  fi
  sleep 15
done

if ! pc2_gpu_serving "${GPU_JOB_ID}"; then
  pc2_log "ERROR: gpu+llm not ready within ${PC2_COMPUTE_LLM_WAIT_SEC}s"
  pc2_session_py set compute_state failed
  pc2_session_py set last_error '"llm endpoint not ready while gpu running"'
  exit 2
fi

pc2_log "gpu serving and compute allocated — starting c2hls worker"

URL="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${PC2_ENDPOINT_FILE}').read_text())['url'])
")"
export OPENAI_BASE_URL="${URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${PC2_ENDPOINT_FILE}').read_text()).get('model',''))
" 2>/dev/null || true)"
if [[ -n "${MODEL}" ]]; then
  export C2HLS_MODEL="${MODEL}"
fi

pc2_log "using LLM OPENAI_BASE_URL=${OPENAI_BASE_URL} model=${C2HLS_MODEL:-<unset>}"
pc2_session_py set compute_state running

pc2_log "running worker: ${WORKER_CMD}"
# shellcheck disable=SC2086
set +e
eval ${WORKER_CMD}
worker_rc=$?
set -e

if [[ "${worker_rc}" -eq 0 ]]; then
  pc2_session_py set compute_state completed
  pc2_log "compute worker finished successfully (rc=0)"
else
  pc2_session_py set compute_state failed
  pc2_session_py set last_error "\"worker exited rc=${worker_rc}\""
  pc2_log "compute worker failed rc=${worker_rc}"
fi
exit "${worker_rc}"
