#!/usr/bin/env bash
# Live progress dashboard for a Fir batch_parallel campaign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
HOST="${FIR_DASHBOARD_HOST:-127.0.0.1}"
PORT="${FIR_DASHBOARD_PORT:-8765}"
PY="${C2HLS_PYTHON:-python3}"
RESTART="${FIR_DASHBOARD_RESTART:-1}"

_stop_dashboard_on_port() {
  local pattern="batch_parallel/dashboard.py.*--port ${PORT}"
  if pkill -u "$(whoami)" -f "${pattern}" 2>/dev/null; then
    fir_log "stopped existing dashboard on port ${PORT}"
    sleep 0.5
  fi
}

_port_in_use() {
  ss -tln 2>/dev/null | grep -q ":${PORT} "
}

if _port_in_use; then
  if [[ "${RESTART}" == "1" ]]; then
    _stop_dashboard_on_port
  fi
  if _port_in_use; then
    echo "ERROR: port ${PORT} already in use (set FIR_DASHBOARD_RESTART=1 or choose another FIR_DASHBOARD_PORT)" >&2
    exit 1
  fi
fi

exec "${PY}" "${SCRIPT_DIR}/batch_parallel/dashboard.py" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}"
