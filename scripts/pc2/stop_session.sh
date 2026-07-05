#!/usr/bin/env bash
# Cancel supervised PC2 session jobs and stop watch_session.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      shift
      export PC2_SESSION_ID="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
_pc2_configure_session_paths
cd "${C2HLS_ROOT}"

gpu_id="$(pc2_session_py get gpu_job_id 2>/dev/null || true)"
comp_id="$(pc2_session_py get compute_job_id 2>/dev/null || true)"
borrowed="$(pc2_session_py get gpu_borrowed 2>/dev/null || echo false)"

if [[ "${borrowed}" != "True" && "${borrowed}" != "true" && "${borrowed}" != "1" ]]; then
  if [[ -n "${gpu_id}" && "${gpu_id}" != "None" && "${gpu_id}" != "null" ]]; then
    pc2_log "cancelling job ${gpu_id}"
    pc2_cancel_job "${gpu_id}"
  fi
fi

if [[ -n "${comp_id}" && "${comp_id}" != "None" && "${comp_id}" != "null" ]]; then
  pc2_log "cancelling job ${comp_id}"
  pc2_cancel_job "${comp_id}"
fi

if [[ -n "${PC2_SESSION_ID:-}" ]]; then
  if pkill -u "$(whoami)" -f "watch_session.sh ${PC2_SESSION_ID}" 2>/dev/null; then
    pc2_log "stopped watch_session (${PC2_SESSION_ID})"
  fi
elif pkill -u "$(whoami)" -f "${SCRIPT_DIR}/watch_session.sh" 2>/dev/null; then
  pc2_log "stopped watch_session"
fi

rm -f "${PC2_ENDPOINT_FILE}"
pc2_session_py set gpu_job_id null 2>/dev/null || true
pc2_session_py set compute_job_id null 2>/dev/null || true
pc2_session_py set gpu_state ended 2>/dev/null || true
pc2_session_py set compute_state ended 2>/dev/null || true
pc2_log "session stopped (re-run start_session.sh for a clean start)"
