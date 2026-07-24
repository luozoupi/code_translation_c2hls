#!/bin/bash
#SBATCH -J c2hls-devstral-lora-serve
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH -A hpc-prf-llmfpga
#SBATCH -t 24:00:00
#SBATCH -o /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-lora-serve-%j.out
#SBATCH -e /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl/logs/devstral-lora-serve-%j.err
#
# Serve Devstral FP8 base + optional PEFT LoRA modules for c2hls external_llm.
#
# Env:
#   ENDPOINT_DIR   required — write llm_endpoint.json here when /v1/models is healthy
#   SERVE_MODE     base | dpo | both   (default: both)
#   API_PORT       default 8000
#   DEFAULT_MODEL  name written into llm_endpoint.json (default: dpo if available else base)
#
# Example:
#   ENDPOINT_DIR=/scratch/.../c2hls/rl/eval/heldout_ab \
#     sbatch rl/slurm/serve_devstral_lora_for_c2hls.slurm

set -euo pipefail

RL_ROOT="${RL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/rl}"
DEVSTRAL_ROOT="${DEVSTRAL_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/devstral2}"
MODEL_PATH="${MODEL_PATH:-${DEVSTRAL_ROOT}/models/Devstral-2-123B-Instruct-2512}"
BASE_NAME="${BASE_SERVED_NAME:-mistralai/Devstral-2-123B-Instruct-2512}"
DPO_ADAPTER="${DPO_ADAPTER:-${RL_ROOT}/outputs/devstral_dpo_2024769/adapter}"
SFT_ADAPTER="${SFT_ADAPTER:-${RL_ROOT}/outputs/devstral_sft_2024406/adapter}"
SERVE_MODE="${SERVE_MODE:-both}"
API_PORT="${API_PORT:-8000}"
HOST_BIND="${HOST:-0.0.0.0}"
TP="${TENSOR_PARALLEL_SIZE:-4}"

mkdir -p "${RL_ROOT}/logs"
if [[ -z "${ENDPOINT_DIR:-}" ]]; then
  echo "ERROR: set ENDPOINT_DIR to a directory for llm_endpoint.json"
  exit 2
fi
mkdir -p "${ENDPOINT_DIR}"

# shellcheck disable=SC1091
source "${DEVSTRAL_ROOT}/load_gpu_modules.sh"
# shellcheck disable=SC1091
source "${DEVSTRAL_ROOT}/ensure_venv.sh"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model ${MODEL_PATH}"
  exit 1
fi

EXTRA_ARGS=( )
case "${SERVE_MODE}" in
  base)
    DEFAULT_MODEL="${DEFAULT_MODEL:-${BASE_NAME}}"
    ;;
  dpo|both)
    if [[ ! -d "${DPO_ADAPTER}" ]]; then
      echo "Missing DPO adapter ${DPO_ADAPTER}"
      exit 1
    fi
    EXTRA_ARGS+=(--enable-lora --max-loras 2 --max-lora-rank 16)
    # One --lora-modules flag: duplicate flags overwrite each other in vLLM argparse.
    if [[ "${SERVE_MODE}" == "both" && -d "${SFT_ADAPTER}" ]]; then
      EXTRA_ARGS+=(--lora-modules "dpo=${DPO_ADAPTER}" "sft=${SFT_ADAPTER}")
    else
      EXTRA_ARGS+=(--lora-modules "dpo=${DPO_ADAPTER}")
    fi
    DEFAULT_MODEL="${DEFAULT_MODEL:-dpo}"
    ;;
  *)
    echo "Unknown SERVE_MODE=${SERVE_MODE}"
    exit 2
    ;;
esac

# Cluster IP for cross-node c2hls workers (eno4np0 on Otus).
NODE_IP="$(ip -o -4 addr show dev eno4np0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
if [[ -z "${NODE_IP}" ]]; then
  NODE_IP="$(hostname -I | awk '{print $1}')"
fi
URL="http://${NODE_IP}:${API_PORT}/v1"

echo "Serving ${MODEL_PATH} mode=${SERVE_MODE} url=${URL} default_model=${DEFAULT_MODEL}"
echo "extra: ${EXTRA_ARGS[*]:-}"

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TP}" \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --served-model-name "${BASE_NAME}" \
  --host "${HOST_BIND}" \
  --port "${API_PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  "${EXTRA_ARGS[@]}" &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until healthy, then plant endpoint for c2hls.
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null 2>&1; then
    python3 - <<PY
import json
from pathlib import Path
ep = {
  "url": "${URL}",
  "model": "${DEFAULT_MODEL}",
  "base_model": "${BASE_NAME}",
  "job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "node_ip": "${NODE_IP}",
  "port": ${API_PORT},
  "serve_mode": "${SERVE_MODE}",
  "borrowed": True,
  "external_llm": True,
  "dpo_adapter": "${DPO_ADAPTER}",
  "sft_adapter": "${SFT_ADAPTER}",
}
Path("${ENDPOINT_DIR}").mkdir(parents=True, exist_ok=True)
Path("${ENDPOINT_DIR}/llm_endpoint.json").write_text(json.dumps(ep, indent=2) + "\\n")
Path("${ENDPOINT_DIR}/endpoint.url").write_text("${URL}\\n")
print("wrote ${ENDPOINT_DIR}/llm_endpoint.json")
print(json.dumps(ep, indent=2))
PY
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM died before becoming healthy"
    exit 1
  fi
  sleep 10
done

if [[ ! -f "${ENDPOINT_DIR}/llm_endpoint.json" ]]; then
  echo "Timed out waiting for vLLM /v1/models"
  exit 1
fi

echo "Endpoint ready: ${URL} (model=${DEFAULT_MODEL})"
wait "${VLLM_PID}"
