#!/usr/bin/env bash
# Fir: GPU modules + vLLM env from ../inference/ (mirrors pc2/setup_vllm_env.sh).

fir_setup_vllm_env() {
  local inference="${FIR_INFERENCE_ROOT:-/scratch/asa582/workspaces/inference}"
  local modules="${FIR_GPU_MODULES:-python/3.11.5 cuda/12.6}"

  if command -v module >/dev/null 2>&1 && [[ -n "${modules}" ]]; then
    module purge 2>/dev/null || true
    # shellcheck disable=SC2086
    module load ${modules}
  fi

  if [[ -f "${inference}/scripts/setup_vllm_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${inference}/scripts/setup_vllm_env.sh"
    setup_vllm_env
  elif [[ -n "${FIR_VLLM_VENV:-}" && -f "${FIR_VLLM_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${FIR_VLLM_VENV}/bin/activate"
  fi

  if [[ -f "${inference}/scripts/load_inference_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${inference}/scripts/load_inference_env.sh"
  fi

  export FIR_LLM_MODEL="${FIR_LLM_MODEL:-${VLLM_SERVED_MODEL_NAME:-${C2HLS_MODEL:-}}}"
  export FIR_VLLM_TENSOR_PARALLEL_SIZE="${FIR_VLLM_TENSOR_PARALLEL_SIZE:-${VLLM_TENSOR_PARALLEL_SIZE:-4}}"
  export FIR_LLM_PORT="${FIR_LLM_PORT:-${VLLM_PORT:-8000}}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
}
