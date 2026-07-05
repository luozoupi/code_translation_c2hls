#!/usr/bin/env bash
# Start a batch_parallel campaign (pilot or full) from a JSON config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_pilot.json}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
VARIANT="${BATCH_PARALLEL_VARIANT:-}"
DRY_RUN=0
FOREGROUND_COORD=0
BORROW_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift; CONFIG="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --variant) shift; VARIANT="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground-coordinator) FOREGROUND_COORD=1; shift ;;
    --borrow-gpu) BORROW_GPU=1; shift ;;
    --no-borrow-gpu) BORROW_GPU=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export BATCH_PARALLEL_CONFIG="${CONFIG}"
export PC2_JOB_TAG="${PC2_JOB_TAG:-${STAMP}}"
if [[ -z "${PC2_BATCH_JOB_PREFIX:-}" ]]; then
  PC2_BATCH_JOB_PREFIX="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(d.get("job_prefix", "bpcplx"))
PY
)"
fi
export PC2_BATCH_JOB_PREFIX
# Slurm walltime must exceed cosim_timeout_s (default 12h); common.sh defaults to 3h.
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-13:00:00}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

if [[ -z "${VARIANT}" ]]; then
  VARIANT="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
print(json.loads(p.read_text())["pilot"]["variant"])
PY
)"
fi

ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel}"
CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${ARTIFACT_PREFIX}_${STAMP}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
rm -rf "${CAMPAIGN_ROOT}"
mkdir -p "${CAMPAIGN_ROOT}/flow/snapshots" "${CAMPAIGN_ROOT}/variants"

read -r BENCH_COUNT SYNTH_NODES SYNTH_WPN COSIM_NODES COSIM_WPN <<<"$(
  "${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(len(d["pilot"]["benches"]), d["synth_nodes_per_variant"], d["synth_workers_per_node"], d["cosim_nodes_per_variant"], d["cosim_workers_per_node"])
PY
)"

"${PY}" - <<PY
import sys
sys.path.insert(0, "${C2HLS_ROOT}/scripts/pc2")
from batch_parallel_config import init_campaign_json, load_config, campaign_paths, benches_for_config, seed_kwargs_for_workflow
from batch_parallel_queue import BatchParallelQueue
cfg = load_config()
paths = campaign_paths(__import__("pathlib").Path("${CAMPAIGN_ROOT}"))
init_campaign_json(paths["root"], cfg, stamp="${STAMP}", active_variants=["${VARIANT}"])
queue = BatchParallelQueue(paths["queue_db"])
benches = cfg.sort_benches(benches_for_config(cfg))
queue.register_benches("${VARIANT}", benches)
seed_kw = seed_kwargs_for_workflow(cfg.pilot_workflow)
seeded = queue.seed_initial_wave("${VARIANT}", benches, max_inflight=cfg.max_inflight_benches, seed_kwargs=seed_kw)
print("campaign_root=${CAMPAIGN_ROOT}")
print("config=${CONFIG}")
print("variant=${VARIANT}")
print("benches:", len(benches))
print("seeded:", seeded)
print("deferred:", [b for b in benches if b not in seeded])
PY

echo "synth: ${SYNTH_NODES} nodes x ${SYNTH_WPN} workers"
echo "cosim: ${COSIM_NODES} nodes x ${COSIM_WPN} workers"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${VARIANT}" \
    "${SCRIPT_DIR}/start_batch_parallel_variant.sh" --dry-run
  echo "dry-run ok"
  exit 0
fi

GPU_JOB=""
GPU_BORROWED=0
if [[ "${BORROW_GPU}" -eq 1 ]]; then
  export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  export PC2_SESSION_DIR="${CAMPAIGN_ROOT}"
  export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
  export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"
  mkdir -p "${CAMPAIGN_ROOT}/flow"
  if ! "${PY}" "${SCRIPT_DIR}/pc2_llm_discovery.py" adopt "${PC2_ENDPOINT_FILE}" --require-job-running; then
    echo "ERROR: --borrow-gpu set but no healthy borrowable endpoint found" >&2
    exit 1
  fi
  GPU_JOB="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["PC2_ENDPOINT_FILE"])
doc = json.loads(p.read_text())
print(doc.get("job_id") or "")
PY
)"
  GPU_BORROWED=1
  echo "borrowed GPU endpoint (job=${GPU_JOB:-unknown}); no new gpu_h100 job submitted"
else
  GPU_JOB="$(
    BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
      "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
  )"
fi
"${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = "${GPU_JOB}" or None
doc["gpu_mode"] = "up"
doc["gpu_borrowed"] = bool(int("${GPU_BORROWED}"))
p.write_text(json.dumps(doc, indent=2) + "\\n")
PY

nohup env BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" C2HLS_PYTHON="${PY}" \
  BATCH_PARALLEL_CONFIG="${CONFIG}" \
  "${SCRIPT_DIR}/batch_parallel_watch_session.sh" \
  >> "${CAMPAIGN_ROOT}/flow/watch.log" 2>&1 &

nohup env BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_CONFIG="${CONFIG}" OPENAI_BASE_URL="" \
  "${PY}" "${SCRIPT_DIR}/batch_parallel_gpu_drain.py" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  >> "${CAMPAIGN_ROOT}/flow/gpu_drain.log" 2>&1 &

if [[ "${FOREGROUND_COORD}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_CONFIG="${CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel_coordinator.py" --campaign-root "${CAMPAIGN_ROOT}"
else
  nohup env BATCH_PARALLEL_CONFIG="${CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel_coordinator.py" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    >> "${CAMPAIGN_ROOT}/flow/coordinator.log" 2>&1 &
fi

echo "compute: deferred until GPU RUNNING (batch_parallel_watch_session.sh)"
echo "campaign=${CAMPAIGN_ROOT}"
echo "tail -f ${CAMPAIGN_ROOT}/flow/events.jsonl"
echo "stop: BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT} ${SCRIPT_DIR}/stop_batch_parallel_campaign.sh"
