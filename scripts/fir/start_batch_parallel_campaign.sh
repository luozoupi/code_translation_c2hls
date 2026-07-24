#!/usr/bin/env bash
# Start a Fir batch_parallel campaign (shared GPU, many compute nodes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_pilot.json}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
FOREGROUND_COORD=0
BORROW_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift; CONFIG="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground-coordinator) FOREGROUND_COORD=1; shift ;;
    --borrow-gpu) BORROW_GPU=1; shift ;;
    --no-borrow-gpu) BORROW_GPU=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export BATCH_PARALLEL_CONFIG="${CONFIG}"
export FIR_JOB_TAG="${FIR_JOB_TAG:-${STAMP}}"
if [[ -z "${FIR_FORCE_WALLTIME:-}" ]]; then
  export FIR_FORCE_WALLTIME="${FIR_BATCH_PARALLEL_WALLTIME:-13:00:00}"
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

if [[ -z "${FIR_BATCH_JOB_PREFIX:-}" ]]; then
  FIR_BATCH_JOB_PREFIX="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(d.get("job_prefix", "firbp"))
PY
)"
fi
export FIR_BATCH_JOB_PREFIX

ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel}"
CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/fir/${ARTIFACT_PREFIX}_${STAMP}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export FIR_BATCH_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
rm -rf "${CAMPAIGN_ROOT}"
mkdir -p "${CAMPAIGN_ROOT}/flow"

"${PY}" - <<PY
import sys
sys.path.insert(0, "${C2HLS_ROOT}/scripts/fir")
from batch_parallel.config import init_campaign_json, load_config, campaign_paths, benches_for_campaign
from batch_parallel.queue import FirBatchParallelQueue
cfg = load_config()
paths = campaign_paths(__import__("pathlib").Path("${CAMPAIGN_ROOT}"))
init_campaign_json(paths["root"], cfg, stamp="${STAMP}")
queue = FirBatchParallelQueue(paths["queue_db"])
benches = benches_for_campaign({"config": cfg.to_dict()}, cfg)
added = queue.register_benches(benches)
print("campaign_root=${CAMPAIGN_ROOT}")
print("config=${CONFIG}")
print("benches:", len(benches), benches)
print("queued:", added)
PY

read -r COMPUTE_NODES WORKERS_PER_NODE <<<"$(
  "${PY}" - <<'PY'
import os, sys
sys.path.insert(0, f"{os.environ['C2HLS_ROOT']}/scripts/fir")
from batch_parallel.config import load_config
cfg = load_config()
print(cfg.compute_nodes, cfg.workers_per_node)
PY
)"
echo "compute: ${COMPUTE_NODES} nodes x ${WORKERS_PER_NODE} workers (shared GPU)"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok (campaign init at ${CAMPAIGN_ROOT})"
  exit 0
fi

GPU_JOB=""
GPU_BORROWED=0
if [[ "${BORROW_GPU}" -eq 1 ]]; then
  export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  export FIR_BATCH_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  _fir_configure_session_paths
  export FIR_ENDPOINT_FILE FIR_SESSION_DIR FIR_SESSION_FILE FIR_WATCH_LOG
  mkdir -p "${CAMPAIGN_ROOT}/flow"
  if ! "${PY}" "${SCRIPT_DIR}/fir_llm_discovery.py" adopt "${FIR_ENDPOINT_FILE}" --require-job-running; then
    echo "ERROR: --borrow-gpu set but no healthy borrowable endpoint found" >&2
    exit 1
  fi
  GPU_JOB="$("${PY}" - "${FIR_ENDPOINT_FILE}" <<'PY'
import json, sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text())
print(doc.get("job_id") or "")
PY
)"
  GPU_BORROWED=1
  echo "borrowed GPU endpoint (job=${GPU_JOB:-unknown}); no new gpu job submitted"
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
  "${SCRIPT_DIR}/batch_parallel_watch.sh" \
  >> "${CAMPAIGN_ROOT}/flow/watch.log" 2>&1 &

if [[ "${FOREGROUND_COORD}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_CONFIG="${CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel/coordinator.py" --campaign-root "${CAMPAIGN_ROOT}"
else
  nohup env BATCH_PARALLEL_CONFIG="${CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel/coordinator.py" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    >> "${CAMPAIGN_ROOT}/flow/coordinator.log" 2>&1 &
fi

echo "compute: deferred until GPU RUNNING (batch_parallel_watch.sh)"
echo "campaign=${CAMPAIGN_ROOT}"
echo "tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "tail -f ${CAMPAIGN_ROOT}/flow/events.jsonl"
echo "stop: BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT} ${SCRIPT_DIR}/stop_batch_parallel_campaign.sh"
