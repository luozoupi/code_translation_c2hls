#!/usr/bin/env bash
# One-bench MachSuite post-flash dataflow worker (for parallel Slurm array/jobs).
# Waits for campaign llm_endpoint.json, then runs run_post_flash_dataflow for --bench.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
BENCH="${1:?usage: $0 <bench>}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

export C2HLS_RUN_COSIM="${C2HLS_RUN_COSIM:-1}"
export C2HLS_REFERENCE_COSIM="${C2HLS_REFERENCE_COSIM:-1}"
export C2HLS_COSIM_REQUIRED="${C2HLS_COSIM_REQUIRED:-0}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-machsuite_stream_cosim_repairs}"
export C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

EP="${CAMPAIGN_ROOT}/llm_endpoint.json"
deadline=$((SECONDS + 14400))
pc2_log "dataflow bench=${BENCH} waiting for ${EP}"
while (( SECONDS < deadline )); do
  if [[ -f "${EP}" ]]; then
    OPENAI_BASE_URL="$("${PY}" -c "import json;print(json.load(open('${EP}'))['url'].rstrip('/'))")"
    export OPENAI_BASE_URL
    MODEL="$("${PY}" -c "import json;print(json.load(open('${EP}')).get('model') or '')" 2>/dev/null || true)"
    if [[ -n "${MODEL}" ]]; then
      export C2HLS_MODEL="${MODEL}"
    fi
    if curl -sf --max-time 10 "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
      pc2_log "endpoint ready ${OPENAI_BASE_URL} model=${C2HLS_MODEL:-unset}"
      break
    fi
  fi
  sleep 20
done
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  pc2_log "ERROR: timed out waiting for LLM endpoint"
  exit 2
fi

pc2_log "START dataflow ${BENCH}"
set +e
"${PY}" "${SCRIPT_DIR}/run_post_flash_dataflow.py" --pc2 \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --benches "${BENCH}" \
  --force \
  --results-suffix "${C2HLS_POST_FLASH_RESULTS_SUFFIX}" \
  --prompt-policy system_skills \
  --contract-turns "${C2HLS_DATAFLOW_CONTRACT_ROUNDS}"
rc=$?
set -e
pc2_log "DONE dataflow ${BENCH} rc=${rc}"
exit "${rc}"
