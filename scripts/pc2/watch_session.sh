#!/usr/bin/env bash
# Supervise GPU + compute jobs for one PC2 test session (Mode B, --pc2 only).
#
# Ordering (queue waits are normal on PC2):
#   1. GPU submitted → may queue 1–2 days
#   2. GPU RUNNING → LLM server starts → compute submitted (after:gpu_job)
#   3. Compute may queue while GPU keeps serving
#   4. Compute RUNNING + GPU still serving + endpoint healthy → worker starts
#
# Usage (from login node, repo root):
#   ./scripts/pc2/start_session.sh
#   ./scripts/pc2/watch_session.sh          # foreground
#   ./scripts/pc2/watch_session.sh --once   # single check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${1:-}" ]]; then
  export PC2_SESSION_ID="$1"
  shift
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
_pc2_configure_session_paths
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}"

ONCE=0
if [[ "${1:-}" == "--once" ]]; then
  ONCE=1
fi

pc2_session_py init >/dev/null

_should_restart() {
  local which="$1"
  local count
  count="$(pc2_session_py get "restarts.${which}" 2>/dev/null || echo 0)"
  [[ "${count}" -lt "${PC2_MAX_RESTARTS}" ]]
}

_reset_compute_wait() {
  pc2_session_py set compute_job_id null
  pc2_session_py set compute_state waiting_for_gpu
}

_check_gpu() {
  local job_id state sess_state
  job_id="$(pc2_session_py get gpu_job_id 2>/dev/null || true)"
  state="$(pc2_job_state "${job_id}")"
  sess_state="$(pc2_session_py get gpu_state 2>/dev/null || echo queued)"

  if [[ "${state}" == "pending" ]]; then
    pc2_session_py set gpu_state queued 2>/dev/null || true
    return 0
  fi

  if [[ "${state}" == "running" ]]; then
    pc2_session_py set gpu_state running 2>/dev/null || true
    if pc2_endpoint_healthy; then
      pc2_session_py set gpu_state ready 2>/dev/null || true
    fi
    return 0
  fi

  if [[ "${state}" == "none" && ( -z "${job_id}" || "${job_id}" == "None" || "${job_id}" == "null" ) ]]; then
    if _should_restart gpu; then
      pc2_log "no gpu job; submitting (expect queue wait on gpu_h100)"
      pc2_session_py bump-restart gpu >/dev/null
      "${SCRIPT_DIR}/submit_gpu.sh" >/dev/null
    fi
    return 0
  fi

  case "${state}" in
    COMPLETED|COMPLETING)
      if pc2_endpoint_healthy; then
        pc2_session_py set gpu_state ready 2>/dev/null || true
      else
        pc2_session_py set gpu_state ended
        local comp_id
        comp_id="$(pc2_session_py get compute_job_id 2>/dev/null || true)"
        if pc2_job_is_pending "${comp_id}"; then
          pc2_log "gpu ended; cancelling queued compute ${comp_id}"
          pc2_cancel_job "${comp_id}"
          _reset_compute_wait
        fi
      fi
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)
      if [[ "${sess_state}" == "ready" ]] && pc2_endpoint_healthy; then
        return 0
      fi
      if _should_restart gpu; then
        pc2_log "gpu job ${job_id} state=${state}; resubmitting"
        pc2_session_py bump-restart gpu >/dev/null
        rm -f "${PC2_ENDPOINT_FILE}"
        local comp_id
        comp_id="$(pc2_session_py get compute_job_id 2>/dev/null || true)"
        if pc2_job_active "${comp_id}"; then
          pc2_log "cancelling compute ${comp_id} tied to failed gpu"
          pc2_cancel_job "${comp_id}"
        fi
        _reset_compute_wait
        "${SCRIPT_DIR}/submit_gpu.sh" >/dev/null
      else
        pc2_log "gpu restart limit reached; not resubmitting"
      fi
      ;;
  esac
}

_check_compute() {
  local gpu_id comp_id gpu_state comp_state sess_comp
  gpu_id="$(pc2_session_py get gpu_job_id 2>/dev/null || true)"
  comp_id="$(pc2_session_py get compute_job_id 2>/dev/null || true)"
  gpu_state="$(pc2_job_state "${gpu_id}")"
  comp_state="$(pc2_job_state "${comp_id}")"
  sess_comp="$(pc2_session_py get compute_state 2>/dev/null || echo waiting_for_gpu)"

  if [[ "${sess_comp}" == "completed" ]]; then
    return 0
  fi

  # Compute allocated or running: require GPU still serving.
  if [[ "${comp_state}" == "pending" ]]; then
    pc2_session_py set compute_state queued 2>/dev/null || true
    if ! pc2_job_is_running "${gpu_id}"; then
      pc2_log "compute ${comp_id} queued but gpu not running; cancelling compute"
      pc2_cancel_job "${comp_id}"
      _reset_compute_wait
    fi
    return 0
  fi

  if [[ "${comp_state}" == "running" ]]; then
    if ! pc2_job_is_running "${gpu_id}"; then
      pc2_log "compute ${comp_id} running but gpu not running; cancelling compute"
      pc2_cancel_job "${comp_id}"
      _reset_compute_wait
    fi
    return 0
  fi

  # No active compute job — submit once GPU is RUNNING (worker waits for LLM health).
  if [[ "${gpu_state}" != "running" ]]; then
    pc2_session_py set compute_state waiting_for_gpu 2>/dev/null || true
    return 0
  fi

  if [[ "${comp_state}" == "none" || -z "${comp_id}" || "${comp_id}" == "None" || "${comp_id}" == "null" ]]; then
    pc2_log "gpu ${gpu_id} running; submitting compute (may queue on ${PC2_COMPUTE_PARTITION})"
    "${SCRIPT_DIR}/submit_compute.sh" >/dev/null
    return 0
  fi

  case "${comp_state}" in
    COMPLETED|COMPLETING)
      pc2_session_py set compute_state completed 2>/dev/null || true
      ;;
    FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|INTERRUPTED|BOOT_FAIL|DEADLINE)
      if _should_restart compute; then
        if ! pc2_job_is_running "${gpu_id}"; then
          pc2_log "compute failed but gpu not running; wait for gpu before resubmit"
          _reset_compute_wait
          return 0
        fi
        pc2_log "compute job ${comp_id} state=${comp_state}; resubmitting"
        pc2_session_py bump-restart compute >/dev/null
        pc2_session_py set compute_job_id null
        "${SCRIPT_DIR}/submit_compute.sh" >/dev/null
      else
        pc2_log "compute restart limit reached"
      fi
      ;;
  esac
}

pc2_log "watch started session=${PC2_SESSION_ID:-default} (interval=${PC2_WATCH_INTERVAL_SEC}s max_restarts=${PC2_MAX_RESTARTS})"
pc2_log "flow: gpu_queue → gpu_run → compute_submit → compute_queue → both_run → worker"

while true; do
  _check_gpu
  _check_compute

  gpu_state="$(pc2_session_py get gpu_state 2>/dev/null || echo ?)"
  compute_state="$(pc2_session_py get compute_state 2>/dev/null || echo ?)"
  pc2_log "status gpu=${gpu_state} compute=${compute_state}"

  if [[ "${compute_state}" == "completed" ]]; then
    pc2_log "session complete (compute finished)"
    if [[ "${PC2_AUTO_STOP_ON_COMPLETE}" == "1" ]]; then
      pc2_log "auto-stop: waiting ${PC2_AUTO_STOP_DELAY_SEC}s before stopping session"
      sleep "${PC2_AUTO_STOP_DELAY_SEC}"
      pc2_log "auto-stop: stopping session (gpu + watch)"
      if [[ -n "${PC2_SESSION_ID:-}" ]]; then
        "${SCRIPT_DIR}/stop_session.sh" --session-id "${PC2_SESSION_ID}"
      else
        "${SCRIPT_DIR}/stop_session.sh"
      fi
    fi
    exit 0
  fi

  if [[ "${ONCE}" -eq 1 ]]; then
    exit 0
  fi
  sleep "${PC2_WATCH_INTERVAL_SEC}"
done
