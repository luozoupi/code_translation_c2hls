#!/usr/bin/env bash
# Cancel supervised Fir session jobs and stop watch_session.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      shift
      export FIR_SESSION_ID="$1"
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
_fir_configure_session_paths
cd "${C2HLS_ROOT}"

gpu_id="$(fir_session_py get gpu_job_id 2>/dev/null || true)"
comp_id="$(fir_session_py get compute_job_id 2>/dev/null || true)"

if [[ -n "${gpu_id}" && "${gpu_id}" != "None" && "${gpu_id}" != "null" ]]; then
  fir_log "cancelling job ${gpu_id}"
  fir_cancel_job "${gpu_id}"
fi

if [[ -n "${comp_id}" && "${comp_id}" != "None" && "${comp_id}" != "null" ]]; then
  fir_log "cancelling job ${comp_id}"
  fir_cancel_job "${comp_id}"
fi

if [[ -n "${FIR_SESSION_ID:-}" ]]; then
  if pkill -u "$(whoami)" -f "watch_session.sh ${FIR_SESSION_ID}" 2>/dev/null; then
    fir_log "stopped watch_session (${FIR_SESSION_ID})"
  fi
elif pkill -u "$(whoami)" -f "${SCRIPT_DIR}/watch_session.sh" 2>/dev/null; then
  fir_log "stopped watch_session"
fi

rm -f "${FIR_ENDPOINT_FILE}"
fir_session_py set gpu_job_id null 2>/dev/null || true
fir_session_py set compute_job_id null 2>/dev/null || true
fir_session_py set gpu_state ended 2>/dev/null || true
fir_session_py set compute_state ended 2>/dev/null || true
fir_log "session stopped (re-run start_session.sh for a clean start)"
