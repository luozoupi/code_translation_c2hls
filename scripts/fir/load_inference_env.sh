#!/usr/bin/env bash
# Source LLM env from the inference repo (Fir open-weight, not team API).
set -euo pipefail

_FIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$(cd "${_FIR_DIR}/../.." && pwd)}"
export C2HLS_SITE=fir

# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/source_local_env.sh"

INFERENCE_ROOT="${FIR_INFERENCE_ROOT:-/scratch/asa582/workspaces/inference}"
LOADER="${INFERENCE_ROOT}/scripts/load_inference_env.sh"
if [[ ! -f "${LOADER}" ]]; then
  echo "ERROR: missing inference loader: ${LOADER}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${LOADER}"

# Propagate served model name to C2HLS when unset.
export C2HLS_MODEL="${C2HLS_MODEL:-${VLLM_SERVED_MODEL_NAME:-mistralai/Devstral-2-123B-Instruct-2512}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
