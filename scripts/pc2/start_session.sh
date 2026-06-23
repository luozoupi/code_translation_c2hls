#!/usr/bin/env bash
# Start a supervised PC2 batch session (Mode B, --pc2 only).
#
# Queue ordering:
#   GPU job submitted first (gpu_h100, may wait in queue)
#   → when GPU is RUNNING, compute is submitted with Slurm after:gpu
#   → compute may also queue; worker starts only when BOTH are running
#
# From the login node:
#   cp local.env.example local.env   # once
#   # set PC2_LLM_MODEL=... and optional PC2_* vars in local.env
#   ./scripts/pc2/start_session.sh
#
# Options:
#   --worker-cmd 'python c2hls.py --pc2 --bench nw'
#   --foreground-watch   run watch in foreground (default: background + log)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}"

FOREGROUND_WATCH=0
AUTO_STOP_ON_COMPLETE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      shift
      export PC2_SESSION_ID="$1"
      shift
      ;;
    --worker-cmd)
      shift
      export PC2_WORKER_CMD="$1"
      shift
      ;;
    --foreground-watch)
      FOREGROUND_WATCH=1
      shift
      ;;
    --auto-stop-on-complete)
      AUTO_STOP_ON_COMPLETE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# Re-resolve per-session paths after --session-id.
_pc2_configure_session_paths
mkdir -p "${PC2_SESSION_DIR}"

if [[ "${AUTO_STOP_ON_COMPLETE}" -eq 1 ]]; then
  export PC2_AUTO_STOP_ON_COMPLETE=1
  export PC2_AUTO_STOP_DELAY_SEC="${PC2_AUTO_STOP_DELAY_SEC:-120}"
fi

pc2_session_py init --reset >/dev/null
pc2_session_py set worker_cmd "${PC2_WORKER_CMD}"

pc2_log "starting session id=${PC2_SESSION_ID:-default} gpu_partition=${PC2_GPU_PARTITION} compute_partition=${PC2_COMPUTE_PARTITION} walltime=${PC2_WALLTIME}"
pc2_log "worker: ${PC2_WORKER_CMD}"
if [[ "${PC2_AUTO_STOP_ON_COMPLETE}" == "1" ]]; then
  pc2_log "auto-stop on worker success: delay=${PC2_AUTO_STOP_DELAY_SEC}s"
fi

"${SCRIPT_DIR}/submit_gpu.sh" >/dev/null

if [[ "${FOREGROUND_WATCH}" -eq 1 ]]; then
  exec "${SCRIPT_DIR}/watch_session.sh" "${PC2_SESSION_ID:-}"
fi

nohup "${SCRIPT_DIR}/watch_session.sh" "${PC2_SESSION_ID:-}" >> "${PC2_WATCH_LOG}" 2>&1 &
watch_pid=$!
pc2_log "watch_session running in background pid=${watch_pid}"
pc2_log "tail -f ${PC2_WATCH_LOG}"
pc2_log "session file: ${PC2_SESSION_FILE}"
