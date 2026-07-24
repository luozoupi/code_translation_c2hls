#!/usr/bin/env bash
# Sequential DeepSeek ChatHLS U280 RAG2 campaigns: rag2_skills -> rag2_ns.
#
# Starts ONE shared login-node DeepSeek queue proxy (workers=1, default port
# 18093 to avoid clashing with the scrape sequence on 18092) and runs the
# two RAG2 flavors one after another via start_chathls_deepseek_one.sh,
# gating each start on Beijing off-peak hours (see deepseek_peak.py).
#
# Usage:
#   ./scripts/pc2/start_chathls_deepseek_rag2_sequence.sh [--dry-run] [--skip-peak-wait]
#
# Env:
#   CHATHLS_DEEPSEEK_PROXY_PORT      default 18093 for this script
#   C2HLS_DEEPSEEK_PEAK_POLL_SEC     peak-wait poll interval (default 300)
#   C2HLS_DEEPSEEK_STATUS_POLL_SEC   campaign_status poll interval (default 120)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

# External-llm DeepSeek runs need a real upstream key on the login proxy.
if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi
CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
# shellcheck disable=SC1091
source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
  echo "ERROR: OPENAI_API_KEY missing after setup_deepseek_api.sh" >&2
  exit 2
fi

DRY_RUN=0
SKIP_PEAK_WAIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-peak-wait) SKIP_PEAK_WAIT=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Separate port from scrape DeepSeek sequence (18092) and ChatHLS (18082).
export CHATHLS_DEEPSEEK_PROXY_PORT="${CHATHLS_DEEPSEEK_PROXY_PORT:-18093}"
export CHATHLS_DEEPSEEK_QUEUE_WORKERS="${CHATHLS_DEEPSEEK_QUEUE_WORKERS:-1}"

PEAK_POLL_SEC="${C2HLS_DEEPSEEK_PEAK_POLL_SEC:-300}"
STATUS_POLL_SEC="${C2HLS_DEEPSEEK_STATUS_POLL_SEC:-120}"
PY="${C2HLS_PYTHON:-python3}"

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/deepseek_u280_rag2_seq_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${SEQ_ROOT}"
STATE_JSON="${SEQ_ROOT}/sequence_state.json"

echo "=== ChatHLS DeepSeek U280 RAG2 sequence: rag2_skills -> rag2_ns ==="
echo "seq_root=${SEQ_ROOT}"
echo "proxy_port=${CHATHLS_DEEPSEEK_PROXY_PORT}"
echo "dry_run=${DRY_RUN} skip_peak_wait=${SKIP_PEAK_WAIT}"

# --- sequence_state.json helpers -------------------------------------------
_state_init() {
  "${PY}" - "${SEQ_ROOT}" "${DRY_RUN}" "${SKIP_PEAK_WAIT}" <<'PY'
import json, sys, time
from pathlib import Path

seq_root, dry_run, skip_peak_wait = sys.argv[1], sys.argv[2], sys.argv[3]
p = Path(seq_root) / "sequence_state.json"
doc = {
    "seq_root": seq_root,
    "method": "rag2",
    "dry_run": bool(int(dry_run)),
    "skip_peak_wait": bool(int(skip_peak_wait)),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "flavors": ["rag2_skills", "rag2_ns"],
    "endpoint_url": None,
    "campaigns": {},
    "sequence_status": "running",
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_state_set_endpoint() {
  local url="$1"
  "${PY}" - "${STATE_JSON}" "${url}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc["endpoint_url"] = sys.argv[2]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_state_set_flavor() {
  local flavor="$1" key="$2" value="$3"
  "${PY}" - "${STATE_JSON}" "${flavor}" "${key}" "${value}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
flavor, key, value = sys.argv[2], sys.argv[3], sys.argv[4]
doc["campaigns"].setdefault(flavor, {})[key] = value
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_state_finish() {
  local status="$1"
  "${PY}" - "${STATE_JSON}" "${status}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc["sequence_status"] = sys.argv[2]
doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_state_init

# --- 1. Shared login-node DeepSeek proxy (or fake endpoint for --dry-run) --
ENDPOINT_JSON="${SEQ_ROOT}/llm_endpoint.json"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] writing fake llm_endpoint.json (no proxy started)"
  "${PY}" - "${ENDPOINT_JSON}" "${CHATHLS_DEEPSEEK_PROXY_PORT}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
port = sys.argv[2]
p.write_text(json.dumps({
    "url": f"http://127.0.0.1:{port}/v1",
    "model": "deepseek-chat",
    "job_id": None,
    "borrowed": True,
    "external_llm": True,
    "queued": True,
    "workers": 1,
    "dry_run": True,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2) + "\n")
PY
else
  echo "starting shared DeepSeek login-node proxy into ${SEQ_ROOT} (port ${CHATHLS_DEEPSEEK_PROXY_PORT}) ..."
  bash "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${SEQ_ROOT}"
fi

URL="$(
  "${PY}" -c "import json; print(json.load(open('${ENDPOINT_JSON}'))['url'])"
)"
echo "endpoint_url=${URL}"
_state_set_endpoint "${URL}"

# --- Beijing peak-hour gate -------------------------------------------------
wait_off_peak() {
  if [[ "${DRY_RUN}" -eq 1 || "${SKIP_PEAK_WAIT}" -eq 1 ]]; then
    return 0
  fi
  while "${PY}" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from deepseek_peak import is_beijing_peak
raise SystemExit(0 if is_beijing_peak() else 1)
"; do
    echo "[$(date -Is)] Beijing peak hours — sleeping ${PEAK_POLL_SEC}s before starting/continuing"
    sleep "${PEAK_POLL_SEC}"
  done
}

campaign_prefix_for_flavor() {
  case "$1" in
    rag2_skills) echo "batch_parallel_chathls_fd_ds_rag2" ;;
    rag2_ns) echo "batch_parallel_chathls_fd_ds_rag2_ns" ;;
    *) echo "ERROR: unknown flavor '$1'" >&2; exit 2 ;;
  esac
}

campaign_status() {
  local campaign_root="$1"
  "${PY}" - "${campaign_root}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
if not p.is_file():
    print("missing")
else:
    print(json.loads(p.read_text()).get("campaign_status", "unknown"))
PY
}

SEQUENCE_FAILED=0

for flavor in rag2_skills rag2_ns; do
  wait_off_peak

  stamp="$(date -u +%Y%m%d_%H%M%S)_${flavor}"
  prefix="$(campaign_prefix_for_flavor "${flavor}")"
  CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${prefix}_${stamp}"

  echo ""
  echo "=== starting flavor=${flavor} stamp=${stamp} ==="
  echo "campaign_root=${CAMPAIGN_ROOT}"

  _state_set_flavor "${flavor}" "stamp" "${stamp}"
  _state_set_flavor "${flavor}" "campaign_root" "${CAMPAIGN_ROOT}"
  _state_set_flavor "${flavor}" "status" "starting"

  ONE_ARGS=(--flavor "${flavor}" --stamp "${stamp}" --endpoint-url "${URL}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    ONE_ARGS+=(--dry-run)
  fi

  "${SCRIPT_DIR}/start_chathls_deepseek_one.sh" "${ONE_ARGS[@]}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would wait for ${CAMPAIGN_ROOT}/campaign.json campaign_status in (complete|completed|failed|aborted), polling every ${STATUS_POLL_SEC}s"
    _state_set_flavor "${flavor}" "status" "dry-run-ok"
    continue
  fi

  _state_set_flavor "${flavor}" "status" "running"
  echo "waiting for flavor=${flavor} campaign_status to settle (poll ${STATUS_POLL_SEC}s) ..."
  while true; do
    st="$(campaign_status "${CAMPAIGN_ROOT}")"
    echo "[$(date -Is)] flavor=${flavor} campaign_status=${st}"
    case "${st}" in
      complete|completed)
        _state_set_flavor "${flavor}" "status" "${st}"
        break
        ;;
      failed|aborted)
        _state_set_flavor "${flavor}" "status" "${st}"
        SEQUENCE_FAILED=1
        break
        ;;
      *)
        sleep "${STATUS_POLL_SEC}"
        ;;
    esac
  done
done

if [[ "${SEQUENCE_FAILED}" -eq 1 ]]; then
  _state_finish "failed"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  _state_finish "dry-run-ok"
else
  _state_finish "complete"
fi

echo ""
echo "=== ChatHLS DeepSeek U280 RAG2 sequence done ==="
echo "seq_root=${SEQ_ROOT}"
echo "sequence_state=${STATE_JSON}"

if [[ "${DRY_RUN}" -ne 1 ]]; then
  PID_FILE="${SEQ_ROOT}/deepseek_proxy.pid"
  if [[ -f "${PID_FILE}" ]]; then
    echo "shared DeepSeek proxy left running (pid=$(cat "${PID_FILE}"))."
    echo "  to stop it: kill \$(cat ${PID_FILE})"
  fi
fi

if [[ "${SEQUENCE_FAILED}" -eq 1 ]]; then
  exit 1
fi
