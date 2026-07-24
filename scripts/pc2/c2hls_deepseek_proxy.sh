#!/usr/bin/env bash
# Thin wrapper: source ChatHLS's DeepSeek API env, start ChatHLS's login-node
# DeepSeek queue proxy into a campaign/session dir, then translate its
# deepseek_endpoint.json into the llm_endpoint.json shape c2hls expects for
# external_llm campaigns.
#
# Usage: c2hls_deepseek_proxy.sh <campaign_or_session_dir>
#
# Env:
#   CHATHLS_ROOT                     ChatHLS-ACL-26 checkout
#                                     (default: sibling test-chathls repo)
#   CHATHLS_DEEPSEEK_PROXY_PORT      login-node proxy port
#                                     (default 18092; ChatHLS's own proxy uses 18082)
#   CHATHLS_DEEPSEEK_QUEUE_WORKERS   proxy queue worker count (default 1)
set -euo pipefail

CAMPAIGN_DIR="${1:?usage: c2hls_deepseek_proxy.sh <campaign_or_session_dir>}"
mkdir -p "${CAMPAIGN_DIR}"

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
export CHATHLS_DEEPSEEK_PROXY_PORT="${CHATHLS_DEEPSEEK_PROXY_PORT:-18092}"  # avoid clash with ChatHLS's own proxy on 18082
export CHATHLS_DEEPSEEK_QUEUE_WORKERS="${CHATHLS_DEEPSEEK_QUEUE_WORKERS:-1}"

if [[ ! -d "${CHATHLS_ROOT}" ]]; then
  echo "c2hls_deepseek_proxy: CHATHLS_ROOT not found: ${CHATHLS_ROOT}" >&2
  exit 1
fi

# compute_worker / prior shells often export OPENAI_API_KEY=EMPTY as a
# placeholder. That is non-empty, so setup_deepseek_api.sh skips loading
# DeepSeek_API from ~/.bashrc and the proxy forwards Bearer EMPTY → 401.
if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi
if [[ "${CHATHLS_API_KEY:-}" == "EMPTY" || "${CHATHLS_API_KEY:-}" == "empty" ]]; then
  unset CHATHLS_API_KEY || true
fi

# shellcheck disable=SC1091
source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"

if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || "${OPENAI_API_KEY}" == "empty" ]]; then
  echo "c2hls_deepseek_proxy: OPENAI_API_KEY missing/EMPTY after setup_deepseek_api.sh" >&2
  exit 1
fi
if [[ "${#OPENAI_API_KEY}" -lt 20 ]]; then
  echo "c2hls_deepseek_proxy: OPENAI_API_KEY looks too short (len=${#OPENAI_API_KEY})" >&2
  exit 1
fi
echo "c2hls_deepseek_proxy: API key loaded (len=${#OPENAI_API_KEY} prefix=${OPENAI_API_KEY:0:7})"

# If a stale proxy is already bound to this port/session, stop it so we do not
# keep serving with a previous EMPTY key (start_deepseek_queue_proxy exits 0
# when the old pid is still alive).
PID_FILE="${CAMPAIGN_DIR}/deepseek_proxy.pid"
if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "c2hls_deepseek_proxy: stopping stale proxy pid=${old_pid}"
    kill "${old_pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${old_pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi
# Also clear endpoint so start script rewrites it for this port/host.
rm -f "${CAMPAIGN_DIR}/deepseek_endpoint.json" "${CAMPAIGN_DIR}/llm_endpoint.json"

bash "${CHATHLS_ROOT}/scripts/pc2/start_deepseek_queue_proxy.sh" "${CAMPAIGN_DIR}"

DS_ENDPOINT="${CAMPAIGN_DIR}/deepseek_endpoint.json"
if [[ ! -f "${DS_ENDPOINT}" ]]; then
  echo "c2hls_deepseek_proxy: ${DS_ENDPOINT} missing after proxy start" >&2
  exit 1
fi

# start_deepseek_queue_proxy.sh writes deepseek_endpoint.json (ChatHLS shape);
# c2hls's external_llm campaigns read llm_endpoint.json, so translate here.
python3 - "${CAMPAIGN_DIR}" <<'PY'
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
ds = json.loads((root / "deepseek_endpoint.json").read_text())
endpoint = {
    "url": ds["url"],
    "model": "deepseek-chat",
    "job_id": None,
    "borrowed": True,
    "external_llm": True,
    "queued": True,
    "workers": ds.get("workers", 1),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(root / "llm_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
print(endpoint["url"])
PY
