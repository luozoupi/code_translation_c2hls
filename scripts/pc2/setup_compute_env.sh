#!/usr/bin/env bash
# PC2-only: optional c2hls Python venv for compute workers (openai, dotenv, etc.).
# Source from setup_vitis_env.sh — do not execute directly.

pc2_setup_compute_python_env() {
  if [[ -n "${PC2_COMPUTE_VENV:-}" && -f "${PC2_COMPUTE_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${PC2_COMPUTE_VENV}/bin/activate"
    export C2HLS_PYTHON="${PC2_COMPUTE_VENV}/bin/python"
  fi
}
