#!/usr/bin/env bash
# One DeepSeek queue proxy for one campaign, on this login node.
# Prints the compute-node URL (http://<login>:<port>/v1) and nothing else.
#
# The shared :18092 proxy is left alone. workers=1 serializes DeepSeek
# upstream. max_queue must be >1 or extra compute-node clients get 429
# "llm proxy busy" instead of waiting.
set -euo pipefail

SESSION_DIR="${1:?usage: start_dedicated_deepseek_proxy.sh <session_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

if [[ "$(hostname -s)" != login* ]]; then
  echo "dedicated deepseek proxy must be started on a login node (internet); host=$(hostname -s)" >&2
  exit 1
fi

mkdir -p "${SESSION_DIR}" "${C2HLS_ROOT}/artifacts/pc2"
LOCK="${C2HLS_ROOT}/artifacts/pc2/deepseek_proxy_port.lock"
BASE_PORT="${C2HLS_DEDICATED_PROXY_BASE_PORT:-18140}"
LOGIN_HOST="${CHATHLS_LOGIN_HOST:-$(hostname -s)}"

exec 9>>"${LOCK}"
flock 9

port="${BASE_PORT}"
while ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$" \
  || [[ "${port}" == "18082" || "${port}" == "18092" ]]; do
  port=$((port + 1))
  if [[ "${port}" -gt $((BASE_PORT + 400)) ]]; then
    echo "no free port for a dedicated deepseek proxy" >&2
    exit 1
  fi
done

# Must be exported onto the child. An unexported assignment is dropped, and
# the child then inherits C2HLS_MODEL from local.env (the vLLM Devstral id).
_RAW_MODEL="${DEEPSEEK_PROXY_MODEL:-}"
if [[ "${_RAW_MODEL}" == deepseek-* || "${_RAW_MODEL}" == deepseek ]]; then
  export DEEPSEEK_PROXY_MODEL="${_RAW_MODEL}"
else
  if [[ -n "${_RAW_MODEL}" ]]; then
    echo "dedicated deepseek proxy: ignoring non-deepseek model ${_RAW_MODEL}" >&2
  fi
  export DEEPSEEK_PROXY_MODEL="deepseek-v4-flash"
fi
export DEEPSEEK_PROXY_UPSTREAM_TIMEOUT="${DEEPSEEK_PROXY_UPSTREAM_TIMEOUT:-3600}"
export CHATHLS_DEEPSEEK_MAX_QUEUE="${CHATHLS_DEEPSEEK_MAX_QUEUE:-32}"
DEEPSEEK_PROXY_MODEL="${DEEPSEEK_PROXY_MODEL}" \
  CHATHLS_DEEPSEEK_PROXY_PORT="${port}" \
  CHATHLS_DEEPSEEK_QUEUE_WORKERS=1 \
  CHATHLS_DEEPSEEK_MAX_QUEUE="${CHATHLS_DEEPSEEK_MAX_QUEUE}" \
  CHATHLS_DEEPSEEK_IDLE_EXIT_S="${CHATHLS_DEEPSEEK_IDLE_EXIT_S:-86400}" \
  CHATHLS_LOGIN_HOST="${LOGIN_HOST}" \
  "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${SESSION_DIR}" \
  >>"${SESSION_DIR}/launcher.log" 2>&1 9>&-

python3 - "${SESSION_DIR}" "${LOGIN_HOST}" "${port}" <<'PY'
import json
import sys
from pathlib import Path

session = Path(sys.argv[1])
host = sys.argv[2]
port = int(sys.argv[3])
url = f"http://{host}:{port}/v1"
path = session / "llm_endpoint.json"
doc = json.loads(path.read_text()) if path.is_file() else {}
doc.update({
    "url": url,
    "host": host,
    "port": port,
    "workers": 1,
    "max_queue": int(__import__("os").environ.get("CHATHLS_DEEPSEEK_MAX_QUEUE") or 32),
    "dedicated": True,
})
path.write_text(json.dumps(doc, indent=2) + "\n")
print(url)
PY
