#!/usr/bin/env bash
# Start a login-node OpenAI queue proxy and write llm_endpoint.json.
#
# Usage: c2hls_openai_proxy.sh <campaign_or_session_dir>
#
# Env:
#   C2HLS_OPENAI_PROXY_PORT     default 18200
#   C2HLS_OPENAI_QUEUE_WORKERS  default 16
#   OPENAI_PROXY_MODEL / C2HLS_MODEL  default gpt-5.6-luna
#   C2HLS_REASONING_EFFORT      optional (e.g. xhigh)
#   C2HLS_OPENAI_UPSTREAM       openai.com (Luna) or api.x.ai (Grok)
#   OPENAI_API_KEY / OPEN_AI_API for OpenAI; Grok_API / XAI_API_KEY for xAI
#
# Anti-poison rules:
#   - Upstream alias "xai" is normalized to https://api.x.ai/v1
#   - Key source is chosen from upstream (never mix OpenAI keys with xAI)
#   - Port reuse only if /v1/health model+upstream match requested identity
set -euo pipefail

CAMPAIGN_DIR="${1:?usage: c2hls_openai_proxy.sh <campaign_or_session_dir>}"
mkdir -p "${CAMPAIGN_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PORT="${C2HLS_OPENAI_PROXY_PORT:-18200}"
WORKERS="${C2HLS_OPENAI_QUEUE_WORKERS:-16}"
MODEL="${OPENAI_PROXY_MODEL:-${C2HLS_MODEL:-gpt-5.6-luna}}"
EFFORT="${C2HLS_REASONING_EFFORT:-}"
UPSTREAM="${C2HLS_OPENAI_UPSTREAM:-https://api.openai.com/v1}"
case "${UPSTREAM}" in
  xai|XAI) UPSTREAM="https://api.x.ai/v1" ;;
esac
UPSTREAM="${UPSTREAM%/}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

_is_xai_upstream() {
  [[ "${UPSTREAM}" == *x.ai* ]]
}

_load_key_named() {
  local name="$1" line val
  line="$(grep -E "^[[:space:]]*${name}=" "${HOME}/.bashrc" 2>/dev/null | tail -1 || true)"
  [[ -n "${line}" ]] || return 1
  val="${line#*=}"
  val="${val%\"}"; val="${val#\"}"
  val="${val%\'}"; val="${val#\'}"
  [[ -n "${val}" ]] || return 1
  printf '%s' "${val}"
}

# Provider-scoped key selection (do NOT fall through OpenAI ↔ xAI).
if _is_xai_upstream; then
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || "${OPENAI_API_KEY}" != xai-* ]]; then
    if [[ -n "${Grok_API:-}" && "${Grok_API}" == xai-* ]]; then
      export OPENAI_API_KEY="${Grok_API}"
    elif [[ -n "${XAI_API_KEY:-}" && "${XAI_API_KEY}" == xai-* ]]; then
      export OPENAI_API_KEY="${XAI_API_KEY}"
    else
      for name in Grok_API XAI_API_KEY GROK_API; do
        val="$(_load_key_named "${name}" || true)"
        if [[ -n "${val}" && "${val}" == xai-* ]]; then
          export OPENAI_API_KEY="${val}"
          break
        fi
      done
    fi
  fi
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" != xai-* ]]; then
    echo "c2hls_openai_proxy: xAI upstream requires Grok_API / XAI_API_KEY (xai-*)" >&2
    exit 1
  fi
else
  # OpenAI hosted: refuse xAI keys even if inherited in OPENAI_API_KEY.
  if [[ -n "${OPENAI_API_KEY:-}" && "${OPENAI_API_KEY}" == xai-* ]]; then
    unset OPENAI_API_KEY
  fi
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
    if [[ -n "${OPEN_AI_API:-}" && "${OPEN_AI_API}" != xai-* ]]; then
      export OPENAI_API_KEY="${OPEN_AI_API}"
    else
      for name in OPEN_AI_API OPENAI_API_KEY; do
        val="$(_load_key_named "${name}" || true)"
        if [[ -n "${val}" && "${val}" != xai-* ]]; then
          export OPENAI_API_KEY="${val}"
          break
        fi
      done
    fi
  fi
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || ${#OPENAI_API_KEY} -lt 20 || "${OPENAI_API_KEY}" == xai-* ]]; then
    echo "c2hls_openai_proxy: OpenAI upstream requires OPENAI_API_KEY / OPEN_AI_API (not xai-*)" >&2
    exit 1
  fi
fi

_proxy_identity_ok() {
  MODEL="${MODEL}" UPSTREAM="${UPSTREAM}" PORT="${PORT}" \
  "${PY}" - <<'PY'
import json, os, urllib.request
base = f"http://127.0.0.1:{os.environ['PORT']}/v1"
want_model = os.environ["MODEL"]
want_up = os.environ["UPSTREAM"].rstrip("/")
try:
    with urllib.request.urlopen(base + "/health", timeout=2) as resp:
        doc = json.loads(resp.read().decode())
except Exception:
    raise SystemExit(1)
got_model = str(doc.get("model") or "")
got_up = str(doc.get("upstream") or "").rstrip("/")
if got_model != want_model:
    raise SystemExit(2)
if got_up and got_up != want_up:
    raise SystemExit(3)
raise SystemExit(0)
PY
}

# Reuse only when model + upstream match (prevents Luna↔Grok port collisions).
_existing_url=""
if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
  || curl -sf --max-time 2 "http://$(hostname):${PORT}/v1/models" >/dev/null 2>&1; then
  if _proxy_identity_ok; then
    _existing_url="http://$(hostname):${PORT}/v1"
    echo "c2hls_openai_proxy: reusing healthy proxy ${_existing_url} model=${MODEL} upstream=${UPSTREAM}" >&2
    mkdir -p "${CAMPAIGN_DIR}"
    "${PY}" - <<PY
import json, time
from pathlib import Path
endpoint = {
    "url": "${_existing_url}",
    "host": "$(hostname)",
    "port": int("${PORT}"),
    "model": "${MODEL}",
    "provider": "openai",
    "upstream": "${UPSTREAM}",
    "queued": True,
    "workers": int("${WORKERS}"),
    "reasoning_effort": "${EFFORT}" or None,
    "reused": True,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
p = Path("${CAMPAIGN_DIR}")
(p / "openai_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
(p / "llm_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
PY
    echo "${_existing_url}"
    exit 0
  fi
  echo "c2hls_openai_proxy: port ${PORT} busy with mismatched identity; not reusing (want model=${MODEL} upstream=${UPSTREAM})" >&2
  # Fall through: try to bind will fail if still occupied — kill only our prior pid file if any.
fi

PID_FILE="${CAMPAIGN_DIR}/openai_proxy.pid"
if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "c2hls_openai_proxy: stopping stale proxy pid=${old_pid}" >&2
    kill "${old_pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${old_pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi
rm -f "${CAMPAIGN_DIR}/openai_endpoint.json" "${CAMPAIGN_DIR}/llm_endpoint.json"

LOG="${CAMPAIGN_DIR}/openai_proxy.log"
ARGS=(
  --host 0.0.0.0
  --port "${PORT}"
  --workers "${WORKERS}"
  --model "${MODEL}"
  --upstream "${UPSTREAM}"
  --session-dir "${CAMPAIGN_DIR}"
)
if [[ -n "${EFFORT}" ]]; then
  ARGS+=(--reasoning-effort "${EFFORT}")
fi

nohup "${PY}" "${SCRIPT_DIR}/openai_queue_proxy.py" "${ARGS[@]}" >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

for _ in $(seq 1 60); do
  if [[ -f "${CAMPAIGN_DIR}/llm_endpoint.json" ]]; then
    url="$("${PY}" -c "import json;print(json.load(open('${CAMPAIGN_DIR}/llm_endpoint.json'))['url'])")"
    if curl -sf --max-time 3 "${url}/models" >/dev/null 2>&1; then
      echo "c2hls_openai_proxy: ready ${url} workers=${WORKERS} model=${MODEL} upstream=${UPSTREAM} effort=${EFFORT:--} pid=$(cat "${PID_FILE}")" >&2
      echo "${url}"
      exit 0
    fi
  fi
  sleep 0.5
done

echo "c2hls_openai_proxy: failed to become ready; see ${LOG}" >&2
exit 1
