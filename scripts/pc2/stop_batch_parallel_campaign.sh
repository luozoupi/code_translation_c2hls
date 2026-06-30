#!/usr/bin/env bash
# Manual abort for batch_parallel campaign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
PY="${C2HLS_PYTHON:-python3}"
cd "${C2HLS_ROOT}"

"${PY}" - <<PY
import json, os, signal
from pathlib import Path
root = Path("${CAMPAIGN_ROOT}")
doc_path = root / "campaign.json"
doc = json.loads(doc_path.read_text()) if doc_path.is_file() else {}
doc["campaign_status"] = "aborted"
doc_path.write_text(json.dumps(doc, indent=2) + "\\n")
pid_file = root / "coordinator.pid"
if pid_file.is_file():
    try:
        os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
    except OSError:
        pass
PY

for job_id in $("${PY}" - <<'PY' "${CAMPAIGN_ROOT}"
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
if not p.is_file():
    raise SystemExit
doc = json.loads(p.read_text())
ids = []
if doc.get("gpu_job_id"):
    ids.append(str(doc["gpu_job_id"]))
for row in doc.get("compute_jobs") or []:
    if row.get("slurm_job_id"):
        ids.append(str(row["slurm_job_id"]))
print(" ".join(ids))
PY
); do
  pc2_cancel_job "${job_id}"
done

for name in bp-synth bp-cosim; do
  while IFS= read -r job_id; do
    [[ -n "${job_id}" ]] || continue
    pc2_cancel_job "${job_id}"
  done < <(squeue -u "$(whoami)" -h -n "${name}" -o "%i" 2>/dev/null || true)
done

BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" "${SCRIPT_DIR}/batch_parallel_stop_session.sh"
pkill -u "$(whoami)" -f "batch_parallel_gpu_drain.py --campaign-root ${CAMPAIGN_ROOT}" 2>/dev/null || true
pkill -u "$(whoami)" -f "batch_parallel_coordinator.py --campaign-root ${CAMPAIGN_ROOT}" 2>/dev/null || true
pc2_log "campaign aborted: ${CAMPAIGN_ROOT}"
