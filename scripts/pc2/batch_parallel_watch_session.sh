#!/usr/bin/env bash
# Campaign-scoped watch: GPU lifecycle + gated compute submit.
#
# Ordering:
#   1. GPU submitted → may queue on gpu_h100
#   2. GPU RUNNING (gpu_mode=up) → synth/cosim nodes submitted once
#   3. Coordinator parks GPU → compute keeps running (never cancelled here)
#
# Compute is NOT cancelled when gpu_mode is parked, pending_unpark, or completing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
export PC2_SESSION_DIR="${CAMPAIGN_ROOT}"
export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"
export PC2_BATCH_JOB_PREFIX="$(pc2_batch_job_prefix "${CAMPAIGN_ROOT}")"
mkdir -p "${CAMPAIGN_ROOT}/flow"

cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"

ONCE=0
if [[ "${1:-}" == "--once" ]]; then
  ONCE=1
fi

_campaign_py() {
  "${PY}" - "${CAMPAIGN_ROOT}" "$@"
}

_read_gpu_mode() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("gpu_mode", "up") if p.is_file() else "up")
PY
}

_read_gpu_job_id() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("gpu_job_id") or "" if p.is_file() else "")
PY
}

_read_compute_state() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("compute_state", "waiting_for_gpu") if p.is_file() else "waiting_for_gpu")
PY
}

_read_active_variants() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
for v in doc.get("active_variants") or []:
    print(v)
PY
}

_campaign_no_gpu() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
print("1" if doc.get("no_gpu") else "0")
PY
}

_campaign_external_llm() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p = root / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
if doc.get("external_llm"):
    print("1")
    raise SystemExit(0)
ep = root / "llm_endpoint.json"
if ep.is_file():
    try:
        epdoc = json.loads(ep.read_text())
    except Exception:
        epdoc = {}
    if epdoc.get("external_llm"):
        print("1")
        raise SystemExit(0)
print("0")
PY
}

_compute_job_ids() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
for row in doc.get("compute_jobs") or []:
    jid = row.get("slurm_job_id")
    if jid:
        print(jid)
PY
}

_set_compute_state() {
  local state="$1"
  _campaign_py "${state}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
state = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["compute_state"] = state
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_reset_compute_wait() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["compute_jobs"] = []
doc["compute_state"] = "waiting_for_gpu"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_any_compute_running() {
  local job_id
  for job_id in $(_compute_job_ids); do
    if pc2_job_is_running "${job_id}"; then
      return 0
    fi
  done
  return 1
}

_any_compute_active() {
  local job_id
  for job_id in $(_compute_job_ids); do
    if pc2_job_active "${job_id}"; then
      return 0
    fi
  done
  return 1
}

_all_compute_pending_only() {
  local job_id
  local saw=0
  for job_id in $(_compute_job_ids); do
    if ! pc2_job_active "${job_id}"; then
      continue
    fi
    saw=1
    if pc2_job_is_running "${job_id}"; then
      return 1
    fi
    if ! pc2_job_is_pending "${job_id}"; then
      return 1
    fi
  done
  [[ "${saw}" -eq 1 ]]
}

_compute_work_pending() {
  _campaign_py <<'PY'
import sqlite3, sys
from pathlib import Path
db = Path(sys.argv[1]) / "queue.db"
if not db.is_file():
    print(0)
    raise SystemExit(0)
conn = sqlite3.connect(db)
n = conn.execute(
    "SELECT COUNT(*) FROM jobs WHERE kind IN ('synth','cosim') AND status IN ('pending','claimed')"
).fetchone()[0]
print(n)
conn.close()
PY
}

_cancel_compute_jobs() {
  local job_id
  for job_id in $(_compute_job_ids); do
    pc2_cancel_job "${job_id}"
  done
}

_job_belongs_to_campaign() {
  local job_id="$1"
  scontrol show job "${job_id}" 2>/dev/null | grep -Fq "BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT}"
}

_discover_unregistered_compute_ids() {
  local name job_id
  if [[ -n "$(_compute_job_ids)" ]]; then
    return 0
  fi
  for name in bp-synth bp-cosim; do
    while IFS= read -r job_id; do
      [[ -n "${job_id}" ]] || continue
      echo "${job_id}"
    done < <(squeue -u "$(whoami)" -h -n "${name}" -o "%i" 2>/dev/null || true)
  done
}

_adopt_unregistered_compute() {
  local job_id
  local adopted=0
  for job_id in $(_discover_unregistered_compute_ids); do
    if ! pc2_job_active "${job_id}"; then
      continue
    fi
    if ! _job_belongs_to_campaign "${job_id}"; then
      continue
    fi
    _campaign_py "${job_id}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
job_id = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
for row in doc.get("compute_jobs") or []:
    if str(row.get("slurm_job_id")) == job_id:
        raise SystemExit(0)
doc.setdefault("compute_jobs", []).append({
    "variant": "unknown",
    "role": "unknown",
    "node_index": -1,
    "slurm_job_id": job_id,
})
doc["compute_state"] = "running"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
    adopted=1
  done
  [[ "${adopted}" -eq 1 ]]
}

_read_gpu_renew_before_s() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
cfg = doc.get("config") or {}
print(int(cfg.get("gpu_renew_before_s") or 600))
PY
}

_read_gpu_renew_pending() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
print(1 if doc.get("gpu_renew_pending") else 0)
PY
}

_set_gpu_renew_pending() {
  local pending="$1"
  _campaign_py "${pending}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
pending = sys.argv[2] == "1"
p = root / "campaign.json"
doc = json.loads(p.read_text())
if pending:
    doc["gpu_renew_pending"] = True
else:
    doc.pop("gpu_renew_pending", None)
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_wait_gpu_endpoint() {
  local tries=0
  while [[ "${tries}" -lt 120 ]]; do
    if pc2_endpoint_healthy; then
      return 0
    fi
    sleep 5
    tries=$((tries + 1))
  done
  return 1
}

_maybe_renew_gpu() {
  local gpu_mode job_id renew_before left old_id new_id
  gpu_mode="$(_read_gpu_mode)"
  [[ "${gpu_mode}" == "up" ]] || return 0
  if pc2_session_is_borrowed_gpu; then
    return 0
  fi
  if [[ "$(_read_gpu_renew_pending)" -eq 1 ]]; then
    return 0
  fi
  job_id="$(_read_gpu_job_id)"
  if ! pc2_job_is_running "${job_id}"; then
    return 0
  fi
  renew_before="$(_read_gpu_renew_before_s)"
  left="$(pc2_job_time_left_sec "${job_id}" 2>/dev/null || true)"
  if [[ -z "${left}" || "${left}" -gt "${renew_before}" ]]; then
    return 0
  fi
  _set_gpu_renew_pending 1
  old_id="${job_id}"
  pc2_log "watch: gpu ${old_id} TIME_LEFT=${left}s (<= ${renew_before}s); submitting replacement"
  new_id="$(
    BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
      "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
  )"
  new_id="${new_id%%;*}"
  _campaign_py "${new_id}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
new_id = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = new_id
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
  if _wait_gpu_endpoint; then
    pc2_log "watch: gpu rolling renew ok old=${old_id} new=${new_id}"
    pc2_cancel_job "${old_id}"
  else
    pc2_log "watch: gpu rolling renew failed to get healthy endpoint for ${new_id}; keeping ${old_id}"
    _campaign_py "${old_id}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
old_id = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = old_id
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
    pc2_cancel_job "${new_id}"
  fi
  _set_gpu_renew_pending 0
}

_check_gpu() {
  if [[ "$(_campaign_no_gpu)" == "1" ]]; then
    return 0
  fi
  if [[ "$(_campaign_external_llm)" == "1" ]]; then
    # External LLM endpoint is always "up" — no gpu_h100 job to watch/resubmit.
    return 0
  fi
  local gpu_mode job_id
  gpu_mode="$(_read_gpu_mode)"
  job_id="$(_read_gpu_job_id)"

  if [[ "${gpu_mode}" == "parked" || "${gpu_mode}" == "pending_unpark" || "${gpu_mode}" == "completing" ]]; then
    return 0
  fi

  if pc2_session_is_borrowed_gpu && pc2_llm_ready; then
    return 0
  fi

  if pc2_job_is_running "${job_id}"; then
    _maybe_renew_gpu
    return 0
  fi

  if ! pc2_job_active "${job_id}" && [[ "${gpu_mode}" == "up" ]]; then
    if pc2_session_is_borrowed_gpu; then
      pc2_log "watch: borrowed endpoint unhealthy; trying another borrow"
      if "${SCRIPT_DIR}/borrow_gpu.sh"; then
        _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ep = root / "llm_endpoint.json"
doc = json.loads((root / "campaign.json").read_text())
if ep.is_file():
    payload = json.loads(ep.read_text())
    doc["gpu_job_id"] = payload.get("job_id") or doc.get("gpu_job_id")
doc["gpu_borrowed"] = True
(root / "campaign.json").write_text(json.dumps(doc, indent=2) + "\n")
PY
        return 0
      fi
      pc2_session_py set gpu_borrowed false >/dev/null || true
    fi
    pc2_log "watch: gpu job missing while gpu_mode=up; resubmitting"
    local new_id
    new_id="$(
      BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
        "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
    )"
    _campaign_py "${new_id}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
new_id = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = new_id
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
  fi
}

_requeue_stale_before_resubmit() {
  local py="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python3}"
  [[ -x "${py}" ]] || py=python3
  "${py}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${C2HLS_ROOT}/scripts/pc2")
from batch_parallel_config import load_config
from batch_parallel_queue import BatchParallelQueue
cfg = load_config()
queue = BatchParallelQueue(Path("${CAMPAIGN_ROOT}") / "queue.db")
stale_s = float(getattr(cfg, "stale_claim_s", 1800) or 1800)
ids = queue.requeue_stale_claimed(max_age_s=stale_s)
orphans = queue.requeue_orphaned_claimed()
cleared = queue.clear_node_slot_assignments()
print(f"stale_requeued={len(ids)} orphan_requeued={len(orphans)} cleared_slots={cleared}")
PY
}

_check_compute() {
  local gpu_mode gpu_id comp_state variant
  gpu_mode="$(_read_gpu_mode)"
  gpu_id="$(_read_gpu_job_id)"
  comp_state="$(_read_compute_state)"

  if [[ "$(_campaign_no_gpu)" == "1" ]]; then
    if [[ "${comp_state}" == "waiting_for_gpu" ]]; then
      pc2_log "watch: no_gpu campaign; submitting compute nodes"
      while IFS= read -r variant; do
        [[ -n "${variant}" ]] || continue
        BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${variant}" \
          "${SCRIPT_DIR}/start_batch_parallel_variant.sh"
      done < <(_read_active_variants)
      _set_compute_state submitted
    elif _any_compute_running; then
      _set_compute_state running
    fi
    return 0
  fi

  # Intentional park/unpark/complete: never cancel compute here.
  if [[ "${gpu_mode}" == "parked" || "${gpu_mode}" == "pending_unpark" || "${gpu_mode}" == "completing" ]]; then
    if _any_compute_running; then
      _set_compute_state running
    fi
    return 0
  fi

  if [[ "${comp_state}" == "submitted" || "${comp_state}" == "running" ]]; then
    if _any_compute_running; then
      _set_compute_state running
      return 0
    fi
    # external_llm / no local GPU job: pending compute is expected; do not cancel.
    if [[ "$(_campaign_external_llm)" == "1" ]]; then
      if _any_compute_active; then
        return 0
      fi
      if [[ "$(_compute_work_pending)" -gt 0 ]]; then
        pc2_log "watch: external_llm compute gone with pending work; requeue+resubmit"
        _requeue_stale_before_resubmit || true
        _reset_compute_wait
      fi
      return 0
    fi
    # Recover from eager submit: queued compute before GPU ever ran.
    if _all_compute_pending_only && ! pc2_job_is_running "${gpu_id}"; then
      pc2_log "watch: cancelling orphan queued compute (gpu not running yet)"
      _cancel_compute_jobs
      _reset_compute_wait
    elif [[ "$(_compute_work_pending)" -gt 0 ]] && pc2_job_is_running "${gpu_id}"; then
      pc2_log "watch: compute finished with pending synth/cosim; requeue+resubmit"
      _requeue_stale_before_resubmit || true
      _reset_compute_wait
    fi
    return 0
  fi

  # waiting_for_gpu — submit once GPU is RUNNING (or borrowed/external endpoint is healthy).
  if [[ "${comp_state}" == "waiting_for_gpu" && "${gpu_mode}" == "up" ]]; then
    if _adopt_unregistered_compute; then
      pc2_log "watch: adopted already-running compute jobs (eager submit recovery)"
      return 0
    fi
    if [[ "$(_campaign_external_llm)" == "1" ]]; then
      if pc2_llm_ready; then
        pc2_log "watch: external_llm endpoint healthy; submitting compute nodes"
        while IFS= read -r variant; do
          [[ -n "${variant}" ]] || continue
          BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${variant}" \
            "${SCRIPT_DIR}/start_batch_parallel_variant.sh"
        done < <(_read_active_variants)
        _set_compute_state submitted
      fi
      return 0
    fi
    if pc2_session_is_borrowed_gpu && pc2_llm_ready; then
      pc2_log "watch: borrowed LLM ready; submitting compute nodes"
      while IFS= read -r variant; do
        [[ -n "${variant}" ]] || continue
        BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${variant}" \
          "${SCRIPT_DIR}/start_batch_parallel_variant.sh"
      done < <(_read_active_variants)
      _set_compute_state submitted
      return 0
    fi
    if [[ -z "${gpu_id}" ]]; then
      return 0
    fi
    if pc2_job_is_running "${gpu_id}"; then
      pc2_log "watch: gpu ${gpu_id} running; submitting compute nodes"
      while IFS= read -r variant; do
        [[ -n "${variant}" ]] || continue
        BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${variant}" \
          "${SCRIPT_DIR}/start_batch_parallel_variant.sh"
      done < <(_read_active_variants)
      _set_compute_state submitted
    fi
  fi
}

pc2_log "batch_parallel watch started (interval=${PC2_WATCH_INTERVAL_SEC}s)"
pc2_log "flow: gpu_queue → gpu_run → compute_submit; park keeps compute"

while true; do
  _check_gpu
  _check_compute

  gpu_mode="$(_read_gpu_mode)"
  comp_state="$(_read_compute_state)"
  pc2_log "watch: gpu_mode=${gpu_mode} compute_state=${comp_state}"

  [[ "${ONCE}" -eq 1 ]] && break
  sleep "${PC2_WATCH_INTERVAL_SEC}"
done
