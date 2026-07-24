#!/usr/bin/env bash
#SBATCH --job-name=c2hls-fir-llm
#SBATCH --output=artifacts/fir/slurm-gpu-%j.out
#SBATCH --error=artifacts/fir/slurm-gpu-%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/fir"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}"

fir_log "gpu job ${SLURM_JOB_ID:-?} starting on $(hostname -s)"

_shutdown=0
_on_term() {
  _shutdown=1
  fir_log "gpu job received termination signal; stopping server"
}
trap _on_term USR1 TERM

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vllm_env.sh"
fir_setup_vllm_env

if [[ -z "${FIR_LLM_MODEL}" ]]; then
  fir_log "ERROR: set FIR_LLM_MODEL or C2HLS_MODEL in fir.env"
  fir_session_py set gpu_state failed
  fir_session_py set last_error '"FIR_LLM_MODEL unset"'
  exit 2
fi

PORT="${FIR_LLM_PORT}"
HOST="$(hostname -s)"
URL="http://${HOST}:${PORT}/v1"
INFERENCE="${FIR_INFERENCE_ROOT}"

if [[ -n "${FIR_LLM_SERVE_CMD}" ]]; then
  # shellcheck disable=SC2086
  eval "${FIR_LLM_SERVE_CMD}" &
  elif [[ -x "${INFERENCE}/scripts/launch_vllm_devstral.sh" ]]; then
  if [[ ! -f "${VLLM_MODEL_PATH:-}/config.json" ]]; then
    fir_log "model missing at ${VLLM_MODEL_PATH:-<unset>}; attempting download"
    bash "${INFERENCE}/scripts/download_devstral_123b.sh" || true
  fi
  VLLM_LOG="${FIR_SESSION_DIR}/vllm-server-${SLURM_JOB_ID:-local}.log"
  bash "${INFERENCE}/scripts/launch_vllm_devstral.sh" >"${VLLM_LOG}" 2>&1 &
else
  if ! command -v vllm >/dev/null 2>&1; then
    fir_log "ERROR: vllm not in PATH; set FIR_INFERENCE_ROOT or FIR_VLLM_VENV"
    fir_session_py set gpu_state failed
    exit 2
  fi
  _serve_target="${VLLM_MODEL_PATH:-${FIR_LLM_MODEL}}"
  _vllm_args=(
    serve "${_serve_target}"
    --host 0.0.0.0
    --port "${PORT}"
    --served-model-name "${FIR_LLM_MODEL}"
    --tensor-parallel-size "${FIR_VLLM_TENSOR_PARALLEL_SIZE}"
  )
  if [[ -n "${FIR_VLLM_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    _extra=( ${FIR_VLLM_EXTRA_ARGS} )
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
    fir_log "ERROR: LLM server process exited during startup"
    if [[ -f "${VLLM_LOG:-}" ]]; then
      fir_log "vllm log tail:"
      tail -20 "${VLLM_LOG}" | while IFS= read -r line; do fir_log "${line}"; done
    fi
    fir_session_py set gpu_state failed
    exit 3
  fi
  sleep 5
done

if [[ "${ready}" -ne 1 ]]; then
  fir_log "ERROR: LLM server did not become healthy in time"
  fir_session_py set gpu_state failed
  kill "${SERVE_PID}" 2>/dev/null || true
  exit 4
fi

"${C2HLS_PYTHON:-python3}" -c "
import json, os
from pathlib import Path
payload = {
    'url': '${URL}',
    'model': os.environ.get('FIR_LLM_MODEL') or os.environ.get('C2HLS_MODEL', ''),
    'host': '${HOST}',
    'port': int('${PORT}'),
    'job_id': os.environ.get('SLURM_JOB_ID'),
    'partition': os.environ.get('FIR_GPU_PARTITION', 'gpubase_bynode_b1'),
    'site': 'fir',
    'started_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
}
Path('${FIR_ENDPOINT_FILE}').write_text(json.dumps(payload, indent=2) + '\n')
"

fir_session_py set gpu_state ready
fir_log "LLM endpoint ready: ${URL}"

wait "${SERVE_PID}" || true
fir_log "gpu serve process ended"
fir_session_py set gpu_state ended
