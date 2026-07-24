#!/usr/bin/env bash
# Supervise GPU + compute jobs for one Fir session.
#
# Usage:
#   ./scripts/fir/start_session.sh
#   ./scripts/fir/watch_session.sh [SESSION_ID]
#   ./scripts/fir/watch_session.sh --once
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${1:-}" && "${1}" != "--once" ]]; then
  export FIR_SESSION_ID="$1"
  shift
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
_fir_configure_session_paths
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}"

ONCE=0
if [[ "${1:-}" == "--once" ]]; then
  ONCE=1
fi

fir_session_py init >/dev/null

_should_restart() {
  local which="$1"
  local count
  count="$(fir_session_py get "restarts.${which}" 2>/dev/null || echo 0)"
  [[ "${count}" -lt "${FIR_MAX_RESTARTS}" ]]
}

_reset_compute_wait() {
  fir_session_py set compute_job_id null
  fir_session_py set compute_state waiting_for_gpu
}

_check_gpu() {
  local job_id state sess_state
  job_id="$(fir_session_py get gpu_job_id 2>/dev/null || true)"
  state="$(fir_job_state "${job_id}")"
  sess_state="$(fir_session_py get gpu_state 2>/dev/null || echo queued)"

  if [[ "${state}" == "pending" ]]; then
    fir_session_py set gpu_state queued 2>/dev/null || true
    return 0
  fi

  if [[ "${state}" == "running" ]]; then
    fir_session_py set gpu_state running 2>/dev/null || true
    if fir_endpoint_healthy; then
      fir_session_py set gpu_state ready 2>/dev/null || true
    fi
    return 0
  fi

  if [[ "${state}" == "none" && ( -z "${job_id}" || "${job_id}" == "None" || "${job_id}" == "null" ) ]]; then
    if _should_restart gpu; then
      fir_log "no gpu job; submitting"
      fir_session_py bump-restart gpu >/dev/null
      "${SCRIPT_DIR}/submit_gpu.sh" >/dev/null
    fi
    return 0
  fi

  case "${state}" in
    COMPLETED|COMPLETING)
      if fir_endpoint_healthy; then
        fir_session_py set gpu_state ready 2>/dev/null || true
      else
        fir_session_py set gpu_state ended
        local comp_id
        comp_id="$(fir_session_py get compute_job_id 2>/dev/null || true)"
        if fir_job_is_pending "${comp_id}"; then
          fir_log "gpu ended; cancelling queued compute ${comp_id}"
          fir_cancel_job "${comp_id}"
          _reset_compute_wait
        fi
      fi
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)
      if [[ "${sess_state}" == "ready" ]] && fir_endpoint_healthy; then
        return 0
      fi
      if _should_restart gpu; then
        fir_log "gpu job ${job_id} state=${state}; resubmitting"
        fir_session_py bump-restart gpu >/dev/null
        rm -f "${FIR_ENDPOINT_FILE}"
        local comp_id
        comp_id="$(fir_session_py get compute_job_id 2>/dev/null || true)"
        if fir_job_active "${comp_id}"; then
          fir_log "cancelling compute ${comp_id} tied to failed gpu"
          fir_cancel_job "${comp_id}"
        fi
        _reset_compute_wait
        "${SCRIPT_DIR}/submit_gpu.sh" >/dev/null
      else
        fir_log "gpu restart limit reached; not resubmitting"
      fi
      ;;
  esac
}

_check_compute() {
  local gpu_id comp_id gpu_state comp_state sess_comp
  gpu_id="$(fir_session_py get gpu_job_id 2>/dev/null || true)"
  comp_id="$(fir_session_py get compute_job_id 2>/dev/null || true)"
  gpu_state="$(fir_job_state "${gpu_id}")"
  comp_state="$(fir_job_state "${comp_id}")"
  sess_comp="$(fir_session_py get compute_state 2>/dev/null || echo waiting_for_gpu)"

  if [[ "${sess_comp}" == "completed" ]]; then
    return 0
  fi

  if [[ "${comp_state}" == "pending" ]]; then
    fir_session_py set compute_state queued 2>/dev/null || true
    if ! fir_job_is_running "${gpu_id}"; then
      fir_log "compute ${comp_id} queued but gpu not running; cancelling compute"
      fir_cancel_job "${comp_id}"
      _reset_compute_wait
    fi
    return 0
  fi

  if [[ "${comp_state}" == "running" ]]; then
    if ! fir_job_is_running "${gpu_id}"; then
      fir_log "compute ${comp_id} running but gpu not running; cancelling compute"
      fir_cancel_job "${comp_id}"
      _reset_compute_wait
    fi
    return 0
  fi

  if [[ "${gpu_state}" != "running" ]]; then
    fir_session_py set compute_state waiting_for_gpu 2>/dev/null || true
    return 0
  fi

  if [[ "${comp_state}" == "none" || -z "${comp_id}" || "${comp_id}" == "None" || "${comp_id}" == "null" ]]; then
    fir_log "gpu ${gpu_id} running; submitting compute (may queue on ${FIR_COMPUTE_PARTITION})"
    "${SCRIPT_DIR}/submit_compute.sh" >/dev/null
    return 0
  fi

  case "${comp_state}" in
    COMPLETED|COMPLETING)
      fir_session_py set compute_state completed 2>/dev/null || true
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|INTERRUPTED|BOOT_FAIL|DEADLINE)
      if _should_restart compute; then
        if ! fir_job_is_running "${gpu_id}"; then
          fir_log "compute failed but gpu not running; wait for gpu before resubmit"
          _reset_compute_wait
          return 0
        fi
        fir_log "compute job ${comp_id} state=${comp_state}; resubmitting"
        fir_session_py bump-restart compute >/dev/null
        fir_session_py set compute_job_id null
        "${SCRIPT_DIR}/submit_compute.sh" >/dev/null
      else
        fir_log "compute restart limit reached"
      fi
      ;;
  esac
}

fir_log "watch started session=${FIR_SESSION_ID:-default} (interval=${FIR_WATCH_INTERVAL_SEC}s max_restarts=${FIR_MAX_RESTARTS})"
fir_log "flow: gpu_queue → gpu_run → compute_submit → compute_queue → both_run → worker"

while true; do
  _check_gpu
  _check_compute

  gpu_state="$(fir_session_py get gpu_state 2>/dev/null || echo ?)"
  compute_state="$(fir_session_py get compute_state 2>/dev/null || echo ?)"
  fir_log "status gpu=${gpu_state} compute=${compute_state}"

  if [[ "${compute_state}" == "completed" ]]; then
    fir_log "session complete (compute finished)"
    skip_auto="$(fir_session_py get skip_auto_stop 2>/dev/null || echo false)"
    if [[ "${FIR_AUTO_STOP_ON_COMPLETE}" == "1" && "${FIR_SKIP_AUTO_STOP:-}" != "1" \
      && "${skip_auto}" != "True" && "${skip_auto}" != "true" && "${skip_auto}" != "1" ]]; then
      fir_log "auto-stop: waiting ${FIR_AUTO_STOP_DELAY_SEC}s before cancelling gpu and stopping session"
      sleep "${FIR_AUTO_STOP_DELAY_SEC}"
      if [[ -n "${FIR_SESSION_ID:-}" ]]; then
        "${SCRIPT_DIR}/stop_session.sh" --session-id "${FIR_SESSION_ID}"
      else
        "${SCRIPT_DIR}/stop_session.sh"
      fi
    else
      fir_log "auto-stop skipped; gpu job left running (stop_session.sh when done)"
    fi
    exit 0
  fi

  if [[ "${ONCE}" -eq 1 ]]; then
    exit 0
  fi
  sleep "${FIR_WATCH_INTERVAL_SEC}"
done
