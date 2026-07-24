#!/usr/bin/env bash
# Sequential DeepSeek ChatHLS U280 campaigns: rag_skills -> skills -> rag_ns.
#
# Starts ONE shared login-node DeepSeek queue proxy (workers=1) and runs the
# three flavors one after another via start_chathls_deepseek_one.sh, gating
# each start (and, for real runs, the wait loop) on Beijing off-peak hours
# (see deepseek_peak.py). Each flavor's campaign is a combined-HLS (16 synth
# nodes doubling as cosim), external_llm (no GPU vLLM) batch_parallel run.
#
# Usage:
#   ./scripts/pc2/start_chathls_deepseek_u280_sequence.sh [--dry-run] [--skip-peak-wait]
#
# Options:
#   --dry-run          Do not start the real DeepSeek proxy and do not submit
#                       any Slurm jobs. Writes a fake llm_endpoint.json under
#                       SEQ_ROOT, skips all peak-hour waits, and calls each
#                       start_chathls_deepseek_one.sh with --dry-run.
#   --skip-peak-wait    Skip the Beijing peak-hour gate (still starts the real
#                       proxy and submits real campaigns unless --dry-run).
#
# Env:
#   C2HLS_DEEPSEEK_PEAK_POLL_SEC    peak-wait poll interval (default 300)
#   C2HLS_DEEPSEEK_STATUS_POLL_SEC  campaign_status poll interval (default 120)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

# External-llm DeepSeek runs need a real upstream key on the login proxy.
# Placeholders like EMPTY must not block loading DeepSeek_API from ~/.bashrc.
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

PEAK_POLL_SEC="${C2HLS_DEEPSEEK_PEAK_POLL_SEC:-300}"
STATUS_POLL_SEC="${C2HLS_DEEPSEEK_STATUS_POLL_SEC:-120}"
PY="${C2HLS_PYTHON:-python3}"

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/deepseek_u280_seq_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${SEQ_ROOT}"
STATE_JSON="${SEQ_ROOT}/sequence_state.json"

echo "=== ChatHLS DeepSeek U280 sequence: rag_skills -> skills -> rag_ns ==="
echo "seq_root=${SEQ_ROOT}"
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
    "dry_run": bool(int(dry_run)),
    "skip_peak_wait": bool(int(skip_peak_wait)),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "flavors": ["rag_skills", "skills", "rag_ns"],
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
  "${PY}" - "${ENDPOINT_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
p.write_text(json.dumps({
    "url": "http://127.0.0.1:18092/v1",
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
  echo "starting shared DeepSeek login-node proxy into ${SEQ_ROOT} ..."
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

# --- Resolve CAMPAIGN_ROOT prefix per flavor (must match
#     start_chathls_deepseek_one.sh's BATCH_PARALLEL_ARTIFACT_PREFIX) --------
campaign_prefix_for_flavor() {
  case "$1" in
    rag_skills) echo "batch_parallel_chathls_fd_ds_rag" ;;
    skills) echo "batch_parallel_chathls_fd_ds_skills" ;;
    rag_ns) echo "batch_parallel_chathls_fd_ds_rag_ns" ;;
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

for flavor in rag_skills skills rag_ns; do
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
echo "=== ChatHLS DeepSeek U280 sequence done ==="
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
