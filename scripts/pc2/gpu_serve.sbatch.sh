#!/usr/bin/env bash
#SBATCH --job-name=c2hls-llm
#SBATCH --output=artifacts/pc2/slurm-gpu-%j.out
#SBATCH --error=artifacts/pc2/slurm-gpu-%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

# Slurm runs a copy under /var/spool/slurmd/ — use submit directory, not BASH_SOURCE.
_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}"

pc2_log "gpu job ${SLURM_JOB_ID:-?} starting on $(hostname -s)"

_shutdown=0
_on_term() {
  _shutdown=1
  pc2_log "gpu job received termination signal; stopping server"
}
trap _on_term USR1 TERM

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vllm_env.sh"
pc2_setup_vllm_env

if [[ -z "${PC2_LLM_MODEL}" ]]; then
  pc2_log "ERROR: set PC2_LLM_MODEL or C2HLS_MODEL in local.env"
  pc2_session_py set gpu_state failed
  pc2_session_py set last_error '"PC2_LLM_MODEL unset"'
  exit 2
fi

PORT="${PC2_LLM_PORT}"
HOST="$(hostname -s)"
URL="http://${HOST}:${PORT}/v1"

if [[ -n "${PC2_LLM_SERVE_CMD}" ]]; then
  # shellcheck disable=SC2086
  eval "${PC2_LLM_SERVE_CMD}" &
else
  if ! command -v vllm >/dev/null 2>&1; then
    pc2_log "ERROR: vllm not in PATH; set PC2_VLLM_VENV or PC2_LLM_SERVE_CMD in local.env"
    pc2_session_py set gpu_state failed
    exit 2
  fi
  _serve_target="${PC2_LLM_WEIGHTS:-${PC2_LLM_MODEL}}"
  _vllm_args=(
    serve "${_serve_target}"
    --host 0.0.0.0
    --port "${PORT}"
    --served-model-name "${PC2_LLM_MODEL}"
    --tensor-parallel-size "${PC2_VLLM_TENSOR_PARALLEL_SIZE}"
  )
  if [[ -n "${PC2_VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    _extra=( ${PC2_VLLM_EXTRA_ARGS} )
    _vllm_args+=( "${_extra[@]}" )
  fi
  vllm "${_vllm_args[@]}" &
fi
SERVE_PID=$!

ready=0
for _ in $(seq 1 120); do
  if [[ "${_shutdown}" -eq 1 ]]; then
    break
  fi
  if curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
    || curl -sf --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    pc2_log "ERROR: LLM server process exited during startup"
    pc2_session_py set gpu_state failed
    exit 3
  fi
  sleep 5
done

if [[ "${ready}" -ne 1 ]]; then
  pc2_log "ERROR: LLM server did not become healthy in time"
  pc2_session_py set gpu_state failed
  kill "${SERVE_PID}" 2>/dev/null || true
  exit 4
fi

"${C2HLS_PYTHON:-python3}" -c "
import json, os
from pathlib import Path
payload = {
    'url': '${URL}',
    'model': os.environ.get('PC2_LLM_MODEL') or os.environ.get('C2HLS_MODEL', ''),
    'host': '${HOST}',
    'port': int('${PORT}'),
    'job_id': os.environ.get('SLURM_JOB_ID'),
    'partition': os.environ.get('PC2_GPU_PARTITION', 'gpu_h100'),
    'started_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
}
Path('${PC2_ENDPOINT_FILE}').write_text(json.dumps(payload, indent=2) + '\n')
"

pc2_session_py set gpu_state ready
pc2_log "LLM endpoint ready: ${URL}"

wait "${SERVE_PID}" || true
pc2_log "gpu serve process ended"
pc2_session_py set gpu_state ended
