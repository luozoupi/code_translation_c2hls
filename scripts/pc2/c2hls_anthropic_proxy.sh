#!/usr/bin/env bash
# Start a login-node Anthropic OpenAI-compat queue proxy and write llm_endpoint.json.
#
# Usage: c2hls_anthropic_proxy.sh <campaign_or_session_dir>
#
# Env:
#   C2HLS_ANTHROPIC_PROXY_PORT   default 18192
#   C2HLS_ANTHROPIC_QUEUE_WORKERS  default 32 (max parallelism)
#   ANTHROPIC_PROXY_MODEL / C2HLS_MODEL  default claude-sonnet-5
#   ANTHROPIC_API_KEY or Claude_API from ~/.bashrc
set -euo pipefail

CAMPAIGN_DIR="${1:?usage: c2hls_anthropic_proxy.sh <campaign_or_session_dir>}"
mkdir -p "${CAMPAIGN_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PORT="${C2HLS_ANTHROPIC_PROXY_PORT:-18192}"
WORKERS="${C2HLS_ANTHROPIC_QUEUE_WORKERS:-32}"
MODEL="${ANTHROPIC_PROXY_MODEL:-${C2HLS_MODEL:-claude-sonnet-5}}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

# Map Claude_API → ANTHROPIC_API_KEY without sourcing full interactive bashrc.
# Prefer sk-ant-* over any poisoned ANTHROPIC_API_KEY (e.g. inherited xai-*).
if [[ -z "${Claude_API:-}" ]]; then
  line="$(grep -E '^[[:space:]]*Claude_API=' "${HOME}/.bashrc" 2>/dev/null | tail -1 || true)"
  if [[ -n "${line}" ]]; then
    val="${line#*Claude_API=}"
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    export Claude_API="${val}"
  fi
fi
if [[ -n "${Claude_API:-}" && "${Claude_API}" == sk-ant-* ]]; then
  export ANTHROPIC_API_KEY="${Claude_API}"
elif [[ -z "${ANTHROPIC_API_KEY:-}" || "${ANTHROPIC_API_KEY}" == "EMPTY" || "${ANTHROPIC_API_KEY}" != sk-ant-* ]]; then
  if [[ -n "${Claude_API:-}" ]]; then
    export ANTHROPIC_API_KEY="${Claude_API}"
  fi
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" || "${ANTHROPIC_API_KEY}" == "EMPTY" || ${#ANTHROPIC_API_KEY} -lt 20 || "${ANTHROPIC_API_KEY}" != sk-ant-* ]]; then
  echo "c2hls_anthropic_proxy: ANTHROPIC_API_KEY / Claude_API missing or not sk-ant-*" >&2
  exit 1
fi
# Proxy speaks OpenAI-compat to compute; Bearer uses the Anthropic key.
export OPENAI_API_KEY="${ANTHROPIC_API_KEY}"

# Reuse healthy listener only when model matches.
if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
  || curl -sf --max-time 2 "http://$(hostname):${PORT}/v1/models" >/dev/null 2>&1; then
  _health_model="$("${PY}" - <<PY
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:${PORT}/v1/health", timeout=2) as r:
        print(json.loads(r.read().decode()).get("model",""))
except Exception:
    print("")
PY
)"
  if [[ "${_health_model}" == "${MODEL}" ]]; then
    _existing_url="http://$(hostname):${PORT}/v1"
    echo "c2hls_anthropic_proxy: reusing healthy proxy ${_existing_url} model=${MODEL}" >&2
    "${PY}" - <<PY
import json, time
from pathlib import Path
endpoint = {
    "url": "${_existing_url}",
    "host": "$(hostname)",
    "port": int("${PORT}"),
    "model": "${MODEL}",
    "provider": "anthropic",
    "queued": True,
    "workers": int("${WORKERS}"),
    "reused": True,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
p = Path("${CAMPAIGN_DIR}")
(p / "anthropic_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
(p / "llm_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
PY
    echo "${_existing_url}"
    exit 0
  fi
  echo "c2hls_anthropic_proxy: port ${PORT} busy with model='${_health_model}' (want ${MODEL}); not reusing" >&2
fi

PID_FILE="${CAMPAIGN_DIR}/anthropic_proxy.pid"
if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "c2hls_anthropic_proxy: stopping stale proxy pid=${old_pid}" >&2
    kill "${old_pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${old_pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi
rm -f "${CAMPAIGN_DIR}/anthropic_endpoint.json" "${CAMPAIGN_DIR}/llm_endpoint.json"

LOG="${CAMPAIGN_DIR}/anthropic_proxy.log"
nohup "${PY}" "${SCRIPT_DIR}/anthropic_queue_proxy.py" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers "${WORKERS}" \
  --model "${MODEL}" \
  --session-dir "${CAMPAIGN_DIR}" \
  >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

# Wait for endpoint file + /models
for _ in $(seq 1 60); do
  if [[ -f "${CAMPAIGN_DIR}/llm_endpoint.json" ]]; then
    url="$("${PY}" -c "import json;print(json.load(open('${CAMPAIGN_DIR}/llm_endpoint.json'))['url'])")"
    if curl -sf --max-time 3 "${url}/models" >/dev/null 2>&1; then
      echo "c2hls_anthropic_proxy: ready ${url} workers=${WORKERS} model=${MODEL} pid=$(cat "${PID_FILE}")" >&2
      echo "${url}"
      exit 0
    fi
  fi
  sleep 0.5
done

echo "c2hls_anthropic_proxy: failed to become ready; see ${LOG}" >&2
exit 1
