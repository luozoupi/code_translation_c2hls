#!/usr/bin/env bash
# Wait for next DONE line in compute log, cancel cleanly, resubmit with 12h walltime.
# If the job vanishes (TIMEOUT/scancel) before DONE, resubmit anyway.
set -euo pipefail

SESSION_ID="${1:?session_id}"
JOB_ID="${2:?compute_job_id}"
LOG_FILE="${3:?compute_log_path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
export PC2_SESSION_ID="${SESSION_ID}"
export PC2_FORCE_WALLTIME="${PC2_FORCE_WALLTIME:-12:00:00}"
_pc2_configure_session_paths
cd "${C2HLS_ROOT}"

pc2_log "wait_done_resubmit: watching ${LOG_FILE} for DONE (job ${JOB_ID})"

_cancelled_by_us=0
_do_resubmit() {
  pc2_log "wait_done_resubmit: submitting compute walltime=${PC2_WALLTIME}"
  local new_id
  new_id="$("${SCRIPT_DIR}/submit_compute.sh")"
  pc2_log "wait_done_resubmit: new compute job ${new_id}"
  "${SCRIPT_DIR}/watch_session.sh" "${SESSION_ID}" --once
  pc2_log "wait_done_resubmit: finished session=${SESSION_ID}"
}

(
  tail -n0 -F "${LOG_FILE}" | while read -r line; do
    if [[ "${line}" == DONE* ]]; then
      pc2_log "wait_done_resubmit: DONE seen; cancelling compute ${JOB_ID}"
      _cancelled_by_us=1
      scancel "${JOB_ID}" 2>/dev/null || true
      exit 0
    fi
  done
) &
tail_pid=$!

while kill -0 "${tail_pid}" 2>/dev/null; do
  if ! squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q .; then
    pc2_log "wait_done_resubmit: job ${JOB_ID} left queue before DONE (timeout/cancel)"
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
    _do_resubmit
    exit 0
  fi
  sleep 5
done

wait "${tail_pid}" 2>/dev/null || true
for _ in $(seq 1 90); do
  if ! squeue -h -j "${JOB_ID}" 2>/dev/null | grep -q .; then
    break
  fi
  sleep 2
done
sleep 1
_do_resubmit
