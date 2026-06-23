#!/usr/bin/env bash
# PC2-only: GPU modules + optional vLLM venv for local OpenAI-compatible serving.
# Source from gpu_serve.sbatch.sh — do not execute directly.

pc2_setup_vllm_env() {
  local modules="${PC2_GPU_MODULES:-lang system CUDA/12.6.0 Python/3.11.5-GCCcore-13.2.0}"

  if command -v module >/dev/null 2>&1 && [[ -n "${modules}" ]]; then
    module purge 2>/dev/null || true
    # shellcheck disable=SC2086
    module load ${modules}
  fi

  if [[ -n "${CUDA_HOME:-}" || -n "${EBROOTCUDA:-}" ]]; then
    export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA:-}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi

  if [[ -n "${PC2_VLLM_VENV:-}" && -f "${PC2_VLLM_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${PC2_VLLM_VENV}/bin/activate"
  fi

  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
}
