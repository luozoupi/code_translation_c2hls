#!/usr/bin/env bash
# Experiment explorer — disk-cached catalog index for fast /api/index loads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HOST="${EXPLORER_HOST:-127.0.0.1}"
PORT="${EXPLORER_PORT:-8766}"
PY="${C2HLS_PYTHON:-python3}"
RESTART="${EXPLORER_RESTART:-1}"
CACHE_SEC="${EXPLORER_CACHE_SEC:-300}"
PREWARM="${EXPLORER_PREWARM:-1}"

_stop_explorer_on_port() {
  local pattern="explorer/server.py.*--port ${PORT}"
  if pkill -u "$(whoami)" -f "${pattern}" 2>/dev/null; then
    echo "[explorer] stopped existing server on port ${PORT}"
    sleep 0.5
  fi
}

_port_in_use() {
  ss -tln 2>/dev/null | grep -q ":${PORT} "
}

if _port_in_use; then
  if [[ "${RESTART}" == "1" ]]; then
    _stop_explorer_on_port
  fi
  if _port_in_use; then
    echo "ERROR: port ${PORT} already in use (set EXPLORER_RESTART=1 or EXPLORER_PORT)" >&2
    exit 1
  fi
fi

exec "${PY}" "${SCRIPT_DIR}/server.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --repo-root "${REPO_ROOT}" \
  --registry "${REPO_ROOT}/experiments_registry.json" \
  --cache-sec "${CACHE_SEC}" \
  $([[ "${PREWARM}" == "1" ]] && echo --prewarm)
