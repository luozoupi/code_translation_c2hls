#!/usr/bin/env bash
#SBATCH --job-name=c2hls-nibi-flash-smoke
# One-node Nibi flash smoke: Devstral vLLM + Vitis flash batch on the same GPU node.

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/nibi"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/nibi/slurm "${C2HLS_TMP_ROOT:-${_REPO_ROOT}/c2hls_tmp}"

STAMP="${C2HLS_NIBI_FLASH_SMOKE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHES="${C2HLS_NIBI_FLASH_SMOKE_BENCHES:-hlsfactory_gemm}"
LOG_DIR="${C2HLS_ROOT}/artifacts/nibi/slurm"
VLLM_LOG="${LOG_DIR}/vllm-server-${SLURM_JOB_ID:-local}.log"

echo "=== Nibi flash smoke job ${SLURM_JOB_ID:-local} on $(hostname -s) ==="
echo "stamp=${STAMP} benches=${BENCHES} model=${NIBI_LLM_MODEL}"

nibi_setup_vllm_env
nibi_activate_compute_venv
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/setup_vitis_env.sh"
nibi_setup_vitis_env

if ! command -v vitis-run >/dev/null 2>&1; then
  echo "ERROR: vitis-run not on PATH after nibi_setup_vitis_env" >&2
  echo "  Check C2HLS_XILINX_SIF / module load apptainer (see nibi.env)" >&2
  exit 2
fi
echo "vitis-run=$(command -v vitis-run) sif=${XILINX_SIF:-<unset>}"

if [[ ! -d "${VLLM_MODEL_PATH}" ]]; then
  echo "ERROR: Devstral weights missing at ${VLLM_MODEL_PATH}" >&2
  exit 2
fi

ENDPOINT_FILE="${LOG_DIR}/vllm-nibi-${SLURM_JOB_ID:-local}.endpoint"
export OPENAI_BASE_URL="http://127.0.0.1:${NIBI_LLM_PORT}/v1"
{
  echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
  echo "VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-${NIBI_LLM_MODEL}}"
  echo "VLLM_MODEL_PATH=${VLLM_MODEL_PATH}"
} > "${ENDPOINT_FILE}"

echo "Starting vLLM (Devstral-2) in background..."
bash "${NIBI_INFERENCE_ROOT}/scripts/launch_vllm_devstral.sh" >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
deadline=$((SECONDS + READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM exited early. Log tail:" >&2
    tail -40 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${NIBI_LLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready at ${OPENAI_BASE_URL}"
    break
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:${NIBI_LLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM not ready within ${READY_TIMEOUT}s" >&2
  tail -40 "${VLLM_LOG}" >&2 || true
  exit 1
fi

echo "Running flash smoke batch..."
export OPENAI_BASE_URL
export C2HLS_MODEL="${NIBI_LLM_MODEL}"
RC=0
"${C2HLS_PYTHON}" scripts/nibi/run_flash_smoke_batch.py \
  --nibi \
  --benches "${BENCHES}" \
  --stamp "${STAMP}" \
  --skip-preflight || RC=$?

kill "${VLLM_PID}" 2>/dev/null || true
trap - EXIT

echo "Flash smoke finished exit_code=${RC}"
exit "${RC}"
