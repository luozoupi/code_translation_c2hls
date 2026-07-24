#!/usr/bin/env bash
# Shared setup for Nibi open-weight runs (source, do not execute).
set -euo pipefail

_NIBI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$(cd "${_NIBI_DIR}/../.." && pwd)}"
export C2HLS_SITE=nibi

# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/source_local_env.sh"

NIBI_SCRATCH_ROOT="${NIBI_SCRATCH_ROOT:-/home/asa582/scratch/asa582}"
NIBI_GPU_PARTITION="${NIBI_GPU_PARTITION:-gpubase_bynode_b1}"
NIBI_SLURM_ACCOUNT="${NIBI_SLURM_ACCOUNT:-def-zhenman_gpu}"
NIBI_WALLTIME="${NIBI_WALLTIME:-3:00:00}"
NIBI_LLM_PORT="${NIBI_LLM_PORT:-8000}"
NIBI_LLM_MODEL="${NIBI_LLM_MODEL:-${C2HLS_MODEL:-mistralai/Devstral-2-123B-Instruct-2512}}"
NIBI_GPU_GPUS="${NIBI_GPU_GPUS:-4}"
NIBI_GPU_CPUS_PER_TASK="${NIBI_GPU_CPUS_PER_TASK:-16}"
NIBI_GPU_MEM="${NIBI_GPU_MEM:-131072M}"
NIBI_VLLM_TENSOR_PARALLEL_SIZE="${NIBI_VLLM_TENSOR_PARALLEL_SIZE:-4}"
NIBI_GPU_MODULES="${NIBI_GPU_MODULES:-python/3.11.5 cuda/12.6}"
NIBI_INFERENCE_ROOT="${NIBI_INFERENCE_ROOT:-${NIBI_SCRATCH_ROOT}/workspaces/inference}"
NIBI_COMPUTE_VENV="${NIBI_COMPUTE_VENV:-${NIBI_SCRATCH_ROOT}/packages/c2hls-venv}"
NIBI_VLLM_VENV="${NIBI_VLLM_VENV:-${NIBI_SCRATCH_ROOT}/packages/c2hls-inference}"

nibi_apply_path_defaults() {
  [[ "${C2HLS_SITE}" == "nibi" ]] || return 0
  # shellcheck disable=SC1091
  source "${_NIBI_DIR}/vitis_paths.env"
}
nibi_apply_path_defaults

nibi_load_inference_env() {
  local loader="${NIBI_INFERENCE_ROOT}/scripts/load_inference_env.sh"
  if [[ -f "${loader}" ]]; then
    # shellcheck disable=SC1090
    source "${loader}"
  fi
  export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-${NIBI_SCRATCH_ROOT}/workspaces/Devstral-2-123B-Instruct-2512}"
  export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${NIBI_LLM_MODEL}}"
}

nibi_setup_vllm_env() {
  if command -v module >/dev/null 2>&1 && [[ -n "${NIBI_GPU_MODULES}" ]]; then
    module purge 2>/dev/null || true
    # shellcheck disable=SC2086
    module load ${NIBI_GPU_MODULES}
  fi
  nibi_load_inference_env
  if [[ -n "${NIBI_VLLM_VENV}" && -d "${NIBI_VLLM_VENV}" ]]; then
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${NIBI_VLLM_VENV}${PYTHONPATH:+:${PYTHONPATH}}"
    export PATH="${NIBI_VLLM_VENV}/bin:${PATH}"
    export VLLM_PYTHON=(python3 -S)
  fi
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${NIBI_LLM_PORT}/v1}"
}

nibi_activate_compute_venv() {
  if [[ -f "${NIBI_COMPUTE_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${NIBI_COMPUTE_VENV}/bin/activate"
    export C2HLS_PYTHON="${NIBI_COMPUTE_VENV}/bin/python"
  else
    export C2HLS_PYTHON="${C2HLS_PYTHON:-python3}"
  fi
  export PYTHONPATH="${C2HLS_ROOT}:${C2HLS_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
}
