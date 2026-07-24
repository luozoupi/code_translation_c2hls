#!/usr/bin/env bash
# Campaign-scoped watch: GPU lifecycle + gated compute submit (shared GPU pattern).
#
# Ordering:
#   1. GPU submitted → may queue on gpubase
#   2. GPU RUNNING (gpu_mode=up) or borrowed endpoint → compute nodes submitted once
#   3. Coordinator parks GPU → compute keeps running until each node's bench finishes
#   4. When serving GPU TimeLeft <= FIR_GPU_PRESUBMIT_SEC (default 10m), pre-submit replacement
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export FIR_BATCH_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
_fir_configure_session_paths
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

_read_dedicated_gpu_job_id() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("dedicated_gpu_job_id") or "" if p.is_file() else "")
PY
}

_read_borrowed_gpu_job_id() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
print(doc.get("borrowed_gpu_job_id") or (doc.get("gpu_job_id") if doc.get("gpu_borrowed") else "") or "")
PY
}

_set_dedicated_gpu_job() {
  local new_id="$1"
  _campaign_py "${new_id}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
new_id = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["dedicated_gpu_job_id"] = new_id
doc["dedicated_gpu_submitted_at"] = datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_read_serving_gpu_job_id() {
  local job_id borrowed_id
  job_id="$(_read_gpu_job_id)"
  borrowed_id="$(_read_borrowed_gpu_job_id)"
  if fir_llm_ready && [[ -f "${FIR_ENDPOINT_FILE}" ]]; then
    "${PY}" - "${FIR_ENDPOINT_FILE}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
jid = json.loads(p.read_text()).get("job_id")
if jid:
    print(jid)
PY
    return 0
  fi
  if [[ -n "${borrowed_id}" ]] && fir_job_is_running "${borrowed_id}"; then
    echo "${borrowed_id}"
    return 0
  fi
  if [[ -n "${job_id}" ]] && fir_job_is_running "${job_id}"; then
    echo "${job_id}"
  fi
}

_gpu_watch_status() {
  local serving_id tleft
  serving_id="$(_read_serving_gpu_job_id 2>/dev/null || true)"
  tleft=""
  if [[ -n "${serving_id}" ]]; then
    tleft="$(fir_job_time_left_sec "${serving_id}" 2>/dev/null || true)"
  fi
  _campaign_py "${serving_id:-}" "${tleft:-}" <<'PY'
import json, sys
from pathlib import Path
doc = json.loads((Path(sys.argv[1]) / "campaign.json").read_text())
serving = sys.argv[2]
tleft = sys.argv[3]
bits = []
if doc.get("gpu_borrowed"):
    bits.append(f"borrowed={doc.get('borrowed_gpu_job_id') or '?'}")
d = doc.get("dedicated_gpu_job_id")
if d:
    bits.append(f"dedicated={d}")
bits.append(f"owned={doc.get('gpu_job_id') or '?'}")
if serving:
    bits.append(f"serving={serving}")
if tleft:
    bits.append(f"tleft={tleft}s")
print(" ".join(bits))
PY
}

_presubmit_replacement_gpu_if_needed() {
  local serving_job_id dedicated_id left
  serving_job_id="${1:-}"
  [[ -n "${serving_job_id}" ]] || return 0

  dedicated_id="$(_read_dedicated_gpu_job_id)"
  if [[ -n "${dedicated_id}" ]] && fir_job_active "${dedicated_id}"; then
    return 0
  fi
  if ! fir_job_is_running "${serving_job_id}"; then
    return 0
  fi

  left="$(fir_job_time_left_sec "${serving_job_id}" 2>/dev/null || true)"
  [[ -n "${left}" ]] || return 0
  if [[ "${left}" -gt "${FIR_GPU_PRESUBMIT_SEC}" ]]; then
    return 0
  fi

  fir_log "watch: gpu ${serving_job_id} has ${left}s left (pre-submit threshold=${FIR_GPU_PRESUBMIT_SEC}s); queuing replacement"
  local new_id
  new_id="$(
    PRESERVE_ENDPOINT=1 BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
      "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
  )"
  _set_dedicated_gpu_job "${new_id}"
  fir_log "watch: pre-submitted dedicated gpu ${new_id} (current ${serving_job_id} still serving)"
}

_endpoint_still_borrowed() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
doc = json.loads((root / "campaign.json").read_text())
if doc.get("gpu_borrowed"):
    print("yes")
    raise SystemExit(0)
ep = root / "llm_endpoint.json"
if ep.is_file() and json.loads(ep.read_text()).get("borrowed"):
    print("yes")
PY
}

_promote_dedicated_gpu_if_ready() {
  _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
camp = root / "campaign.json"
ep = root / "llm_endpoint.json"
doc = json.loads(camp.read_text())
dedicated = doc.get("dedicated_gpu_job_id")
if not dedicated:
    raise SystemExit(0)
if not ep.is_file():
    raise SystemExit(0)
payload = json.loads(ep.read_text())
if payload.get("borrowed"):
    raise SystemExit(0)
if str(payload.get("job_id") or "") != str(dedicated):
    raise SystemExit(0)
doc["gpu_job_id"] = str(dedicated)
doc["gpu_borrowed"] = False
doc.pop("borrowed_gpu_job_id", None)
doc.pop("dedicated_gpu_job_id", None)
camp.write_text(json.dumps(doc, indent=2) + "\n")
print("promoted")
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

_any_compute_running() {
  local job_id
  for job_id in $(_compute_job_ids); do
    if fir_job_is_running "${job_id}"; then
      return 0
    fi
  done
  return 1
}

_any_compute_active() {
  local job_id
  for job_id in $(_compute_job_ids); do
    if fir_job_active "${job_id}"; then
      return 0
    fi
  done
  return 1
}

_all_compute_pending_only() {
  local job_id
  local saw=0
  for job_id in $(_compute_job_ids); do
    if ! fir_job_active "${job_id}"; then
      continue
    fi
    saw=1
    if fir_job_is_running "${job_id}"; then
      return 1
    fi
    if ! fir_job_is_pending "${job_id}"; then
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
    "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','claimed')"
).fetchone()[0]
print(n)
conn.close()
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

_check_gpu() {
  local gpu_mode job_id dedicated_id borrowed_id serving_job_id
  gpu_mode="$(_read_gpu_mode)"
  job_id="$(_read_gpu_job_id)"
  dedicated_id="$(_read_dedicated_gpu_job_id)"
  borrowed_id="$(_read_borrowed_gpu_job_id)"

  if [[ "${gpu_mode}" == "parked" || "${gpu_mode}" == "pending_unpark" || "${gpu_mode}" == "completing" || "${gpu_mode}" == "stopped" ]]; then
    return 0
  fi

  serving_job_id="$(_read_serving_gpu_job_id 2>/dev/null || true)"
  _presubmit_replacement_gpu_if_needed "${serving_job_id}"
  dedicated_id="$(_read_dedicated_gpu_job_id)"

  # Borrowed endpoint still healthy — keep serving it; dedicated may queue in parallel.
  if fir_session_is_borrowed_gpu && fir_llm_ready; then
    return 0
  fi

  # Dedicated replacement queued or running — never double-submit.
  if [[ -n "${dedicated_id}" ]] && fir_job_active "${dedicated_id}"; then
    if fir_job_is_running "${dedicated_id}" && fir_llm_ready; then
      if [[ "$(_promote_dedicated_gpu_if_ready)" == "promoted" ]]; then
        fir_log "watch: promoted dedicated gpu ${dedicated_id} to owned endpoint"
      fi
    fi
    return 0
  fi

  if fir_job_is_running "${job_id}"; then
    return 0
  fi

  if ! fir_job_active "${job_id}" && [[ "${gpu_mode}" == "up" ]]; then
    if fir_session_is_borrowed_gpu; then
      fir_log "watch: borrowed endpoint unhealthy; trying another borrow"
      if "${SCRIPT_DIR}/borrow_gpu.sh"; then
        _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ep = root / "llm_endpoint.json"
doc = json.loads((root / "campaign.json").read_text())
if ep.is_file():
    payload = json.loads(ep.read_text())
    jid = payload.get("job_id")
    if jid:
        doc["borrowed_gpu_job_id"] = str(jid)
        doc["gpu_job_id"] = str(jid)
doc["gpu_borrowed"] = True
(root / "campaign.json").write_text(json.dumps(doc, indent=2) + "\n")
PY
        return 0
      fi
      _campaign_py <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_borrowed"] = False
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
    fi
    fir_log "watch: gpu job missing while gpu_mode=up; resubmitting"
    local new_id
    new_id="$(
      PRESERVE_ENDPOINT=1 BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
        "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
    )"
    if [[ -n "$(_endpoint_still_borrowed 2>/dev/null || true)" ]]; then
      _set_dedicated_gpu_job "${new_id}"
    else
      _campaign_py "${new_id}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
new_id = sys.argv[2]
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = new_id
doc["gpu_borrowed"] = False
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
    fi
  fi
}

_check_compute() {
  local gpu_mode gpu_id comp_state
  gpu_mode="$(_read_gpu_mode)"
  gpu_id="$(_read_gpu_job_id)"
  comp_state="$(_read_compute_state)"

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
    if _any_compute_active; then
      return 0
    fi
    if [[ "$(_compute_work_pending)" -gt 0 ]] && ( fir_gpu_serving "${gpu_id}" || fir_llm_ready ); then
      fir_log "watch: compute finished with pending flash work; resubmitting nodes"
      _reset_compute_wait
    fi
    return 0
  fi

  if [[ "${comp_state}" == "waiting_for_gpu" && "${gpu_mode}" == "up" ]]; then
    if fir_session_is_borrowed_gpu && fir_llm_ready; then
      fir_log "watch: borrowed LLM ready; submitting compute nodes"
      BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
        "${SCRIPT_DIR}/start_batch_parallel_compute.sh"
      _set_compute_state submitted
      return 0
    fi
    if [[ -z "${gpu_id}" ]]; then
      return 0
    fi
    if fir_gpu_serving "${gpu_id}"; then
      fir_log "watch: gpu ${gpu_id} serving; submitting compute nodes"
      BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
        "${SCRIPT_DIR}/start_batch_parallel_compute.sh"
      _set_compute_state submitted
    fi
  fi
}

fir_log "fir batch_parallel watch started (interval=${FIR_WATCH_INTERVAL_SEC}s presubmit=${FIR_GPU_PRESUBMIT_SEC}s)"
fir_log "flow: gpu_queue → gpu_run → compute_submit; park keeps compute"

while true; do
  _check_gpu
  _check_compute
  fir_log "watch: gpu_mode=$(_read_gpu_mode) compute_state=$(_read_compute_state) $(_gpu_watch_status)"
  [[ "${ONCE}" -eq 1 ]] && break
  sleep "${FIR_WATCH_INTERVAL_SEC}"
done
