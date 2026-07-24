#!/usr/bin/env bash
# Start tier_B_ready MachSuite gold-gate batch_parallel (csynth + csim, no GPU/LLM).
#
# Usage:
#   ./scripts/pc2/start_tier_b_machsuite_gold_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_tier_b_machsuite_gold_batch_parallel.sh --stamp 20260709_tier_b_gold
#
# Artifacts: artifacts/pc2/batch_parallel_tier_b_gold_<stamp>/
# After completion:
#   python3 scripts/pc2/aggregate_tier_b_gold_batch_parallel.py artifacts/pc2/batch_parallel_tier_b_gold_<stamp>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_machsuite_gold.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-tier_b_machsuite}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_tier_b_gold}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bptbgold}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-4:00:00}"
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
FOREGROUND_COORD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift; BATCH_PARALLEL_CONFIG="$1"; export BATCH_PARALLEL_CONFIG; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground-coordinator) FOREGROUND_COORD=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export PC2_JOB_TAG="${PC2_JOB_TAG:-${STAMP}}"

rm -rf "${CAMPAIGN_ROOT}"
mkdir -p "${CAMPAIGN_ROOT}/flow/snapshots" "${CAMPAIGN_ROOT}/variants" "${CAMPAIGN_ROOT}/reports/gold_gate"

"${PY}" - <<PY
import sys
sys.path.insert(0, "${C2HLS_ROOT}/scripts/pc2")
from batch_parallel_config import init_campaign_json, load_config, campaign_paths, benches_for_config, seed_kwargs_for_workflow
from batch_parallel_queue import BatchParallelQueue
cfg = load_config()
paths = campaign_paths(__import__("pathlib").Path("${CAMPAIGN_ROOT}"))
doc = init_campaign_json(paths["root"], cfg, stamp="${STAMP}", active_variants=["${BATCH_PARALLEL_VARIANT}"])
doc["no_gpu"] = True
doc["gpu_mode"] = "parked"
doc["gpu_job_id"] = None
doc["gpu_borrowed"] = False
doc["compute_state"] = "waiting_for_gpu"
paths["campaign_json"].write_text(__import__("json").dumps(doc, indent=2) + "\\n", encoding="utf-8")
queue = BatchParallelQueue(paths["queue_db"])
benches = cfg.sort_benches(benches_for_config(cfg))
queue.register_benches("${BATCH_PARALLEL_VARIANT}", benches)
seed_kw = seed_kwargs_for_workflow(cfg.pilot_workflow)
seeded = queue.seed_initial_wave("${BATCH_PARALLEL_VARIANT}", benches, max_inflight=cfg.max_inflight_benches, seed_kwargs=seed_kw)
print("campaign_root=${CAMPAIGN_ROOT}")
print("config=${BATCH_PARALLEL_CONFIG}")
print("variant=${BATCH_PARALLEL_VARIANT}")
print("benches:", len(benches))
print("seeded:", seeded)
print("deferred:", [b for b in benches if b not in seeded])
PY

read -r BENCH_COUNT SYNTH_NODES SYNTH_WPN COSIM_NODES COSIM_WPN <<<"$(
  "${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(len(d["pilot"]["benches"]), d["synth_nodes_per_variant"], d["synth_workers_per_node"], d["cosim_nodes_per_variant"], d["cosim_workers_per_node"])
PY
)"

echo "synth: ${SYNTH_NODES} nodes x ${SYNTH_WPN} workers (no GPU)"
echo "cosim: ${COSIM_NODES} nodes x ${COSIM_WPN} workers"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT}" \
    "${SCRIPT_DIR}/start_batch_parallel_variant.sh" --dry-run
  echo "dry-run ok"
  exit 0
fi

cat > "${CAMPAIGN_ROOT}/submit.json" <<EOF
{
  "stamp": "${STAMP}",
  "campaign_root": "${CAMPAIGN_ROOT}",
  "config": "${BATCH_PARALLEL_CONFIG}",
  "variant": "${BATCH_PARALLEL_VARIANT}",
  "bench_count": ${BENCH_COUNT},
  "no_gpu": true,
  "workflow": "tier_b_gold",
  "corpus": "tier_B_ready"
}
EOF

nohup env BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" C2HLS_PYTHON="${PY}" \
  BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG}" \
  "${SCRIPT_DIR}/batch_parallel_watch_session.sh" \
  >> "${CAMPAIGN_ROOT}/flow/watch.log" 2>&1 &

if [[ "${FOREGROUND_COORD}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel_coordinator.py" --campaign-root "${CAMPAIGN_ROOT}"
else
  nohup env BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel_coordinator.py" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    >> "${CAMPAIGN_ROOT}/flow/coordinator.log" 2>&1 &
fi

echo "campaign=${CAMPAIGN_ROOT}"
echo "monitor: tail -f ${CAMPAIGN_ROOT}/flow/events.jsonl"
echo "aggregate when done:"
echo "  python3 scripts/pc2/aggregate_tier_b_gold_batch_parallel.py ${CAMPAIGN_ROOT}"
echo "stop: BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT} ${SCRIPT_DIR}/stop_batch_parallel_campaign.sh"
