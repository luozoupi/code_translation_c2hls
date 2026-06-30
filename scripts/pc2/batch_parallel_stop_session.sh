#!/usr/bin/env bash
# Stop batch_parallel GPU session + watch (campaign-scoped).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
export PC2_SESSION_DIR="${CAMPAIGN_ROOT}"
export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"

cd "${C2HLS_ROOT}"

job_id="$("${C2HLS_PYTHON:-python3}" - <<'PY' "${CAMPAIGN_ROOT}"
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("gpu_job_id", "") if p.is_file() else "")
PY
)"
pc2_cancel_job "${job_id}"

if pkill -u "$(whoami)" -f "batch_parallel_watch_session.sh ${CAMPAIGN_ROOT}" 2>/dev/null; then
  pc2_log "stopped batch_parallel watch"
fi

rm -f "${PC2_ENDPOINT_FILE}"
"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
root = Path("${CAMPAIGN_ROOT}")
p = root / "campaign.json"
if p.is_file():
    doc = json.loads(p.read_text())
    doc["gpu_job_id"] = None
    doc["gpu_mode"] = "parked"
    p.write_text(json.dumps(doc, indent=2) + "\\n")
PY
pc2_log "batch_parallel GPU session stopped"
