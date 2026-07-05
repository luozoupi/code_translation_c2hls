#!/usr/bin/env bash
# Try to borrow a healthy vLLM endpoint from another PC2 session/campaign.
# Returns 0 when an endpoint was adopted into ${PC2_ENDPOINT_FILE}.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
_pc2_configure_session_paths
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}"

EXCLUDE_ARGS=()
if [[ -f "${PC2_ENDPOINT_FILE}" ]]; then
  EXCLUDE_ARGS+=(--exclude "${PC2_ENDPOINT_FILE}")
fi
if [[ -n "${PC2_BORROW_EXCLUDE_URL:-}" ]]; then
  # Reserved for future URL-based exclusion during fallback retries.
  :
fi

if adopted="$("${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/pc2_llm_discovery.py" adopt \
  "${PC2_ENDPOINT_FILE}" \
  "${EXCLUDE_ARGS[@]}" \
  --require-job-running 2>/dev/null)"; then
  url="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('url',''))")"
  job_id="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")"
  borrowed_from="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('borrowed_from',''))")"
  pc2_session_py set gpu_borrowed true >/dev/null
  pc2_session_py set gpu_job_id "${job_id:-null}" >/dev/null
  pc2_session_py set gpu_state borrowed >/dev/null
  pc2_session_py set borrowed_from "${borrowed_from}" >/dev/null
  pc2_log "borrowed LLM endpoint ${url} (job=${job_id}) from ${borrowed_from}"
  exit 0
fi

# Retry without requiring Slurm RUNNING — endpoint health is the real gate.
if adopted="$("${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/pc2_llm_discovery.py" adopt \
  "${PC2_ENDPOINT_FILE}" \
  "${EXCLUDE_ARGS[@]}" 2>/dev/null)"; then
  url="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('url',''))")"
  job_id="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")"
  borrowed_from="$(printf '%s' "${adopted}" | "${C2HLS_PYTHON:-python3}" -c "import json,sys; print(json.load(sys.stdin).get('borrowed_from',''))")"
  pc2_session_py set gpu_borrowed true >/dev/null
  pc2_session_py set gpu_job_id "${job_id:-null}" >/dev/null
  pc2_session_py set gpu_state borrowed >/dev/null
  pc2_session_py set borrowed_from "${borrowed_from}" >/dev/null
  pc2_log "borrowed LLM endpoint ${url} (job=${job_id}) from ${borrowed_from}"
  exit 0
fi

exit 1
