#!/usr/bin/env bash
# Submit compute nodes for a Fir batch_parallel campaign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?}"
CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_pilot.json}"
export BATCH_PARALLEL_CONFIG="${CONFIG}"
PY="${C2HLS_PYTHON:-python3}"

read -r COMPUTE_NODES WORKERS_PER_NODE WORKER_CPUS WORKER_MEM <<<"$(
  "${PY}" - <<'PY'
import os, sys
sys.path.insert(0, f"{os.environ['C2HLS_ROOT']}/scripts/fir")
from batch_parallel.config import load_config
cfg = load_config()
print(cfg.compute_nodes, cfg.workers_per_node, cfg.worker_cpus, cfg.worker_mem_gb)
PY
)"

_register_compute_job() {
  local node_index="$1"
  local job_id="$2"
  "${PY}" - "${CAMPAIGN_ROOT}" "${node_index}" "${job_id}" <<'PY'
import json, sys
from pathlib import Path
root, node_index, job_id = sys.argv[1:4]
p = Path(root) / "campaign.json"
doc = json.loads(p.read_text())
doc.setdefault("compute_jobs", []).append({
    "node_index": int(node_index),
    "slurm_job_id": job_id,
})
if doc.get("compute_state", "waiting_for_gpu") == "waiting_for_gpu":
    doc["compute_state"] = "submitted"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

submit_node() {
  local node_index="$1"
  local cpus=$((WORKERS_PER_NODE * WORKER_CPUS))
  local mem=$((WORKERS_PER_NODE * WORKER_MEM))
  local job_tag="${FIR_JOB_TAG:-$(basename "${CAMPAIGN_ROOT}")}"
  local job_prefix
  job_prefix="$(fir_batch_job_prefix "${CAMPAIGN_ROOT}")"

  account_args=()
  local compute_account="${FIR_COMPUTE_SLURM_ACCOUNT:-${FIR_SLURM_ACCOUNT:-}}"
  if [[ -n "${compute_account}" ]]; then
    account_args=(--account="${compute_account}")
  fi

  echo "submit compute node ${node_index}: ${cpus} cpu ${mem}G"
  local job_id
  job_id="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --export=ALL,BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",FIR_BATCH_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",BATCH_PARALLEL_NODE_INDEX="${node_index}",BATCH_PARALLEL_CONFIG="${CONFIG}" \
      --job-name="${job_prefix}-compute-n${node_index}-${job_tag}" \
      --output="${CAMPAIGN_ROOT}/slurm-compute-n${node_index}-%j.out" \
      --error="${CAMPAIGN_ROOT}/slurm-compute-n${node_index}-%j.err" \
      "${account_args[@]}" \
      ${FIR_COMPUTE_PARTITION:+--partition="${FIR_COMPUTE_PARTITION}"} \
      --cpus-per-task="${cpus}" \
      --mem="${mem}G" \
      --time="${FIR_COMPUTE_WALLTIME:-${FIR_WALLTIME:-12:00:00}}" \
      "${SCRIPT_DIR}/batch_parallel_compute.sbatch.sh"
  )"
  job_id="${job_id%%;*}"
  _register_compute_job "${node_index}" "${job_id}"
  echo "${job_id}"
}

for i in $(seq 0 $((COMPUTE_NODES - 1))); do
  submit_node "${i}"
done
