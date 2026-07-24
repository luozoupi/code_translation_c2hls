#!/usr/bin/env bash
# Recover batch_parallel compute after Slurm TIMEOUT: requeue orphaned jobs, resubmit nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Same precedence as start_batch_parallel_campaign.sh: explicit BATCH_PARALLEL,
# else preserve caller FORCE, else 13h default.
if [[ -n "${PC2_BATCH_PARALLEL_WALLTIME:-}" ]]; then
  export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
elif [[ -z "${PC2_FORCE_WALLTIME:-}" ]]; then
  export PC2_FORCE_WALLTIME="13:00:00"
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
VARIANT="${BATCH_PARALLEL_VARIANT:-}"
PY="${C2HLS_PYTHON:-python3}"
CONFIG="${BATCH_PARALLEL_CONFIG:-}"

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${SCRIPT_DIR}/batch_parallel_full_aav_n_always_on.json"
fi
export BATCH_PARALLEL_CONFIG="${CONFIG}"

if [[ -z "${VARIANT}" ]]; then
  VARIANT="$("${PY}" - <<PY
import json
from pathlib import Path
doc = json.loads(Path("${CAMPAIGN_ROOT}/campaign.json").read_text())
print(doc["active_variants"][0])
PY
)"
fi

pc2_log "recover: campaign=${CAMPAIGN_ROOT} variant=${VARIANT} walltime=${PC2_WALLTIME}"

"${PY}" - <<PY
import sys
sys.path.insert(0, "${C2HLS_ROOT}/scripts/pc2")
from pathlib import Path
from batch_parallel_queue import BatchParallelQueue

queue = BatchParallelQueue(Path("${CAMPAIGN_ROOT}") / "queue.db")
from batch_parallel_config import load_config
cfg = load_config()
stale_s = float(getattr(cfg, "stale_claim_s", 1800) or 1800)
stale = queue.requeue_stale_claimed(max_age_s=stale_s)
requeued = queue.requeue_orphaned_claimed()
cleared = queue.clear_node_slot_assignments()
pending_cosim = queue.pending_count(kind="cosim")
print(f"stale_requeued={len(stale)}")
print(f"requeued_jobs={len(requeued)}")
print(f"cleared_node_slots={cleared}")
print(f"pending_cosim={pending_cosim}")
for jid in sorted(set(stale) | set(requeued)):
    print(f"  requeued job_id={jid}")
PY

pc2_log "recover: cancelling stale compute for prefix $(pc2_batch_job_prefix "${CAMPAIGN_ROOT}")"
pc2_cancel_batch_parallel_named_jobs "$(pc2_batch_job_prefix "${CAMPAIGN_ROOT}")"
sleep 3

pc2_log "recover: resubmitting synth + cosim nodes"
export PC2_BATCH_JOB_PREFIX="$(pc2_batch_job_prefix "${CAMPAIGN_ROOT}")"
BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  BATCH_PARALLEL_VARIANT="${VARIANT}" \
  BATCH_PARALLEL_CONFIG="${CONFIG}" \
  "${SCRIPT_DIR}/start_batch_parallel_variant.sh"

pc2_log "recover: done"
