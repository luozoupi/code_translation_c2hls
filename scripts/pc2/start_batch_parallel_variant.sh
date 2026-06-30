#!/usr/bin/env bash
# Submit synth/cosim compute nodes for one variant.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?}"
VARIANT="${BATCH_PARALLEL_VARIANT:?}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-python3}"
CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_pilot.json}"
export BATCH_PARALLEL_CONFIG="${CONFIG}"
read -r SYNTH_NODES SYNTH_WPN COSIM_NODES COSIM_WPN WORKER_CPUS WORKER_MEM <<<"$(
  "${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(d["synth_nodes_per_variant"], d["synth_workers_per_node"], d["cosim_nodes_per_variant"], d["cosim_workers_per_node"], d["worker_cpus"], d["worker_mem_gb"])
PY
)"

_register_compute_job() {
  local role="$1"
  local node_index="$2"
  local job_id="$3"
  "${PY}" - "${CAMPAIGN_ROOT}" "${VARIANT}" "${role}" "${node_index}" "${job_id}" <<'PY'
import json, sys
from pathlib import Path
root, variant, role, node_index, job_id = sys.argv[1:6]
p = Path(root) / "campaign.json"
doc = json.loads(p.read_text())
doc.setdefault("compute_jobs", []).append({
    "variant": variant,
    "role": role,
    "node_index": int(node_index),
    "slurm_job_id": job_id,
})
if doc.get("compute_state", "waiting_for_gpu") == "waiting_for_gpu":
    doc["compute_state"] = "submitted"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

submit_node() {
  local role="$1"
  local node_index="$2"
  local workers="$3"
  local cpus=$((workers * WORKER_CPUS))
  local mem=$((workers * WORKER_MEM))
  local template="${SCRIPT_DIR}/batch_parallel_${role}.sbatch.sh"
  echo "submit ${role} node ${node_index}: ${cpus} cpu ${mem}G"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  local job_id
  job_id="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --export=ALL,BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",BATCH_PARALLEL_VARIANT="${VARIANT}",BATCH_PARALLEL_NODE_INDEX="${node_index}",BATCH_PARALLEL_CONFIG="${CONFIG}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task="${cpus}" \
      --mem="${mem}G" \
      --time="${PC2_WALLTIME:-12:00:00}" \
      "${template}"
  )"
  job_id="${job_id%%;*}"
  _register_compute_job "${role}" "${node_index}" "${job_id}"
  echo "${job_id}"
}

for i in $(seq 0 $((SYNTH_NODES - 1))); do
  submit_node synth "${i}" "${SYNTH_WPN}"
done
for i in $(seq 0 $((COSIM_NODES - 1))); do
  submit_node cosim "${i}" "${COSIM_WPN}"
done
