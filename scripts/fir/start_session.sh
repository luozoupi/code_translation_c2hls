#!/usr/bin/env bash
# Start a supervised Fir batch session (GPU vLLM + Apptainer Vitis compute).
#
# From the login node:
#   cp fir.env.example fir.env   # once
#   ./scripts/fir/start_session.sh
#
# Options:
#   --session-id NAME
#   --worker-cmd 'python scripts/fir/run_flash_smoke_batch.py --fir --benches hlsfactory_gemm'
#   --foreground-watch
#   --no-auto-stop          keep GPU running after compute worker finishes
#   --auto-stop-on-complete explicit enable (default on; delay FIR_AUTO_STOP_DELAY_SEC)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}"

FOREGROUND_WATCH=0
AUTO_STOP_ON_COMPLETE="${FIR_AUTO_STOP_ON_COMPLETE:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      shift
      export FIR_SESSION_ID="$1"
      shift
      ;;
    --worker-cmd)
      shift
      export FIR_WORKER_CMD="$1"
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
    --no-auto-stop)
      AUTO_STOP_ON_COMPLETE=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

_fir_configure_session_paths
mkdir -p "${FIR_SESSION_DIR}"

if [[ "${AUTO_STOP_ON_COMPLETE}" -eq 1 ]]; then
  export FIR_AUTO_STOP_ON_COMPLETE=1
else
  export FIR_AUTO_STOP_ON_COMPLETE=0
fi
export FIR_AUTO_STOP_DELAY_SEC="${FIR_AUTO_STOP_DELAY_SEC:-120}"

fir_session_py init --reset >/dev/null
fir_session_py set worker_cmd "${FIR_WORKER_CMD}"

fir_log "starting Fir session id=${FIR_SESSION_ID:-default} gpu_partition=${FIR_GPU_PARTITION} compute_partition=${FIR_COMPUTE_PARTITION} walltime=${FIR_WALLTIME}"
fir_log "worker: ${FIR_WORKER_CMD}"
if [[ "${FIR_AUTO_STOP_ON_COMPLETE}" == "1" ]]; then
  fir_log "auto-stop on worker success: delay=${FIR_AUTO_STOP_DELAY_SEC}s"
fi

"${SCRIPT_DIR}/submit_gpu.sh" >/dev/null

if [[ "${FOREGROUND_WATCH}" -eq 1 ]]; then
  exec "${SCRIPT_DIR}/watch_session.sh" "${FIR_SESSION_ID:-}"
fi

nohup "${SCRIPT_DIR}/watch_session.sh" "${FIR_SESSION_ID:-}" >> "${FIR_WATCH_LOG}" 2>&1 &
watch_pid=$!
fir_log "watch_session running in background pid=${watch_pid}"
fir_log "tail -f ${FIR_WATCH_LOG}"
fir_log "session file: ${FIR_SESSION_FILE}"
