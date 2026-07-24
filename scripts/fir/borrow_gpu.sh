#!/usr/bin/env bash
# Try to borrow a healthy vLLM endpoint from another Fir session/campaign.
# Returns 0 when an endpoint was adopted into ${FIR_ENDPOINT_FILE}.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
_fir_configure_session_paths
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}"

EXCLUDE_ARGS=()
if [[ -f "${FIR_ENDPOINT_FILE}" ]]; then
  EXCLUDE_ARGS+=(--exclude "${FIR_ENDPOINT_FILE}")
fi

_adopt_borrowed() {
  local adopted="$1"
  local url job_id borrowed_from
  url="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('url',''))")"
  job_id="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")"
  borrowed_from="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('borrowed_from',''))")"
  fir_session_py set gpu_borrowed true >/dev/null 2>&1 || true
  fir_session_py set gpu_job_id "${job_id:-null}" >/dev/null 2>&1 || true
  fir_session_py set gpu_state borrowed >/dev/null 2>&1 || true
  fir_session_py set borrowed_from "${borrowed_from}" >/dev/null 2>&1 || true
  if [[ -f "${FIR_SESSION_FILE}" && "$(basename "${FIR_SESSION_FILE}")" == "campaign.json" ]]; then
    "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${FIR_SESSION_FILE}")
doc = json.loads(p.read_text())
doc["gpu_borrowed"] = True
doc["gpu_job_id"] = "${job_id}" or doc.get("gpu_job_id")
doc["borrowed_from"] = "${borrowed_from}"
p.write_text(json.dumps(doc, indent=2) + "\\n")
PY
  fi
  fir_log "borrowed LLM endpoint ${url} (job=${job_id}) from ${borrowed_from}"
}

if adopted="$("${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/fir_llm_discovery.py" adopt \
  "${FIR_ENDPOINT_FILE}" \
  "${EXCLUDE_ARGS[@]}" \
  --require-job-running 2>/dev/null)"; then
  _adopt_borrowed "${adopted}"
  exit 0
fi

if adopted="$("${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/fir_llm_discovery.py" adopt \
  "${FIR_ENDPOINT_FILE}" \
  "${EXCLUDE_ARGS[@]}" 2>/dev/null)"; then
  _adopt_borrowed "${adopted}"
  exit 0
fi

exit 1
