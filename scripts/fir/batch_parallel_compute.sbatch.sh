#!/usr/bin/env bash
#SBATCH --job-name=firbp-compute
#SBATCH --output=slurm-bp-compute-%j.out
#SBATCH --error=slurm-bp-compute-%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
SCRIPT_DIR="${_REPO_ROOT}/scripts/fir"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

export BATCH_PARALLEL_CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?}"
export FIR_BATCH_CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT}"
_fir_configure_session_paths
mkdir -p "${FIR_SESSION_DIR}/flow" "${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export PYTHONPATH="${C2HLS_ROOT}:${C2HLS_ROOT}/scripts:${C2HLS_ROOT}/scripts/fir${PYTHONPATH:+:${PYTHONPATH}}"

fir_log "batch_parallel compute job ${SLURM_JOB_ID:-?} node=${BATCH_PARALLEL_NODE_INDEX:-?} on $(hostname -s)"

_on_term() {
  fir_log "compute job received termination signal"
  exit 143
}
trap _on_term USR1 TERM

export C2HLS_SITE=fir

if command -v module >/dev/null 2>&1 && [[ -n "${FIR_COMPUTE_MODULES:-}" ]]; then
  module purge 2>/dev/null || true
  # shellcheck disable=SC2086
  module load ${FIR_COMPUTE_MODULES}
fi
unset LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true

if [[ -n "${FIR_COMPUTE_VENV:-}" && -f "${FIR_COMPUTE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${FIR_COMPUTE_VENV}/bin/activate"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
fir_setup_vitis_env

if ! command -v vitis-run >/dev/null 2>&1; then
  fir_log "ERROR: vitis-run not on PATH after fir_setup_vitis_env"
  exit 2
fi

GPU_JOB_ID="$("${C2HLS_PYTHON:-python3}" - <<'PY'
import json
from pathlib import Path
import os
p = Path(os.environ["BATCH_PARALLEL_CAMPAIGN_ROOT"]) / "campaign.json"
print(json.loads(p.read_text()).get("gpu_job_id") or "")
PY
)"

fir_log "waiting for shared gpu+llm endpoint"
wait_llm=$((SECONDS + FIR_COMPUTE_LLM_WAIT_SEC))
while (( SECONDS < wait_llm )); do
  if fir_gpu_serving "${GPU_JOB_ID}"; then
    break
  fi
  if [[ -n "${GPU_JOB_ID}" ]] && ! fir_job_active "${GPU_JOB_ID}" && ! fir_session_is_borrowed_gpu; then
    fir_log "ERROR: gpu job ${GPU_JOB_ID} not active"
    exit 2
  fi
  sleep 15
done

if ! fir_gpu_serving "${GPU_JOB_ID}"; then
  fir_log "ERROR: llm endpoint not ready within ${FIR_COMPUTE_LLM_WAIT_SEC}s"
  exit 2
fi

URL="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${FIR_ENDPOINT_FILE}').read_text())['url'])
")"
export OPENAI_BASE_URL="${URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
print(json.loads(Path('${FIR_ENDPOINT_FILE}').read_text()).get('model',''))
" 2>/dev/null || true)"
if [[ -n "${MODEL}" ]]; then
  export C2HLS_MODEL="${MODEL}"
fi

fir_log "shared LLM ready: ${OPENAI_BASE_URL}; starting node runner"

exec "${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/batch_parallel/node_runner.py" \
  --campaign-root "${BATCH_PARALLEL_CAMPAIGN_ROOT}" \
  --node-index "${BATCH_PARALLEL_NODE_INDEX}"
