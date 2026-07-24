#!/usr/bin/env bash
# Launch Devstral LoRA serve (if needed) + two external_llm held-out campaigns:
#   arm A: base served name
#   arm B: dpo LoRA module
#
# Usage:
#   bash rl/scripts/run_heldout_ab_eval.sh
#   SKIP_SERVE=1 ENDPOINT_URL=http://...:8000/v1 bash rl/scripts/run_heldout_ab_eval.sh

set -euo pipefail

C2HLS_ROOT="${C2HLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/c2hls}"
RL_ROOT="${RL_ROOT:-${C2HLS_ROOT}/rl}"
EVAL_ROOT="${EVAL_ROOT:-${RL_ROOT}/eval/heldout_ab_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="${BATCH_PARALLEL_CONFIG:-${RL_ROOT}/eval/batch_parallel_heldout_u280.json}"
BASE_NAME="${BASE_SERVED_NAME:-mistralai/Devstral-2-123B-Instruct-2512}"
DPO_NAME="${DPO_SERVED_NAME:-dpo}"

mkdir -p "${EVAL_ROOT}"
cd "${C2HLS_ROOT}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_RUN_COSIM="${C2HLS_RUN_COSIM:-1}"
export C2HLS_REFERENCE_COSIM="${C2HLS_REFERENCE_COSIM:-1}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-7200}"

if [[ "${SKIP_SERVE:-0}" != "1" ]]; then
  echo "Submitting LoRA serve → ${EVAL_ROOT}"
  SERVE_JOB=$(ENDPOINT_DIR="${EVAL_ROOT}" SERVE_MODE=both DEFAULT_MODEL="${DPO_NAME}" \
    sbatch --parsable "${RL_ROOT}/slurm/serve_devstral_lora_for_c2hls.slurm")
  echo "serve job ${SERVE_JOB}"
  echo "${SERVE_JOB}" > "${EVAL_ROOT}/serve.jobid"
  echo "Waiting for ${EVAL_ROOT}/llm_endpoint.json ..."
  for i in $(seq 1 240); do
    if [[ -f "${EVAL_ROOT}/llm_endpoint.json" ]]; then
      break
    fi
    # bail if serve failed
    st=$(sacct -j "${SERVE_JOB}" -n -o State -X 2>/dev/null | awk '{print $1; exit}')
    if [[ "${st}" == FAILED || "${st}" == CANCELLED || "${st}" == TIMEOUT ]]; then
      echo "Serve job ${SERVE_JOB} ended as ${st}"
      exit 1
    fi
    sleep 15
  done
  if [[ ! -f "${EVAL_ROOT}/llm_endpoint.json" ]]; then
    echo "Timed out waiting for endpoint"
    exit 1
  fi
fi

ENDPOINT_URL="${ENDPOINT_URL:-}"
if [[ -z "${ENDPOINT_URL}" ]]; then
  ENDPOINT_URL=$(python3 -c "import json; print(json.load(open('${EVAL_ROOT}/llm_endpoint.json'))['url'])")
fi
echo "Using endpoint ${ENDPOINT_URL}"

launch_arm() {
  local arm="$1"
  local model="$2"
  local stamp="heldout_${arm}_$(date -u +%Y%m%d_%H%M%S)"
  echo "=== launching arm=${arm} model=${model} stamp=${stamp} ==="
  BATCH_PARALLEL_CONFIG="${CONFIG}" \
  BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}" \
  BATCH_PARALLEL_EXTERNAL_MODEL="${model}" \
  C2HLS_MODEL="${model}" \
  OPENAI_API_KEY="${OPENAI_API_KEY}" \
  ./scripts/pc2/start_batch_parallel_campaign.sh \
    --external-llm \
    --stamp "${stamp}" \
    | tee "${EVAL_ROOT}/launch_${arm}.log"
  # Capture campaign root if printed
  rg -n "CAMPAIGN|campaign|artifacts/pc2" "${EVAL_ROOT}/launch_${arm}.log" | tail -20 || true
}

launch_arm base "${BASE_NAME}"
launch_arm dpo "${DPO_NAME}"

python3 - <<PY
import json
from pathlib import Path
root = Path("${EVAL_ROOT}")
meta = {
  "endpoint_url": "${ENDPOINT_URL}",
  "base_model": "${BASE_NAME}",
  "dpo_model": "${DPO_NAME}",
  "config": "${CONFIG}",
  "benches": json.loads(Path("${CONFIG}").read_text())["pilot"]["benches"],
  "metric_policy": {
    "tier": "4=csim&cosim > 3=cosim > 2=csim > 1=synth",
    "latency": "cosim_cycles if cosim_passed else csynth latency_cycles",
  },
}
(root / "ab_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", root / "ab_meta.json")
PY

echo "A/B campaigns submitted. Eval root: ${EVAL_ROOT}"
echo "After both finish, compare with:"
echo "  python ${RL_ROOT}/scripts/compare_heldout_ab.py --eval-root ${EVAL_ROOT}"
