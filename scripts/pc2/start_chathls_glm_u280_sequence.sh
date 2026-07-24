#!/usr/bin/env bash
# Sequential GLM-4.7-FP8 ChatHLS U280 campaigns:
#   rag2_skills -> rag2_ns -> rag_skills -> skills
#
# Starts ONE shared 2-node GLM vLLM serve (TP=4 + PP=2, max_model_len=202752)
# via glm4.7/run_glm47_serve_for_c2hls.slurm, waits for llm_endpoint.json +
# /v1/models health, then runs each flavor via start_chathls_glm_one.sh.
# On sequence end, scancels the GLM serve job.
#
# Usage:
#   ./scripts/pc2/start_chathls_glm_u280_sequence.sh [--dry-run]
#
# Env:
#   GLM47_ROOT                 default /scratch/hpc-prf-llmfpga/asa582/projects/glm4.7
#   GLM_WALLTIME               sbatch --time for GLM serve (default 7-00:00:00)
#   MAX_MODEL_LEN              default 202752
#   C2HLS_GLM_STATUS_POLL_SEC  campaign_status poll interval (default 120)
#   C2HLS_GLM_READY_POLL_SEC   endpoint ready poll interval (default 30)
#   C2HLS_GLM_READY_TIMEOUT_SEC  max wait for GLM serve (default 10800)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

GLM47_ROOT="${GLM47_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/glm4.7}"
GLM_SERVE_SCRIPT="${GLM47_ROOT}/run_glm47_serve_for_c2hls.slurm"
GLM_WALLTIME="${GLM_WALLTIME:-7-00:00:00}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-202752}"
STATUS_POLL_SEC="${C2HLS_GLM_STATUS_POLL_SEC:-120}"
READY_POLL_SEC="${C2HLS_GLM_READY_POLL_SEC:-30}"
READY_TIMEOUT_SEC="${C2HLS_GLM_READY_TIMEOUT_SEC:-10800}"
PY="${C2HLS_PYTHON:-python3}"

if [[ ! -f "${GLM_SERVE_SCRIPT}" ]]; then
  echo "ERROR: GLM serve script not found: ${GLM_SERVE_SCRIPT}" >&2
  exit 2
fi

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/glm_u280_seq_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${SEQ_ROOT}"
STATE_JSON="${SEQ_ROOT}/sequence_state.json"
ENDPOINT_JSON="${SEQ_ROOT}/llm_endpoint.json"
SEQ_LOG="${SEQ_ROOT}/sequence.log"

echo "=== ChatHLS GLM-4.7 U280 sequence: rag2_skills -> rag2_ns -> rag_skills -> skills ==="
echo "seq_root=${SEQ_ROOT}"
echo "glm_root=${GLM47_ROOT}"
echo "max_model_len=${MAX_MODEL_LEN} walltime=${GLM_WALLTIME}"
echo "dry_run=${DRY_RUN}"

_state_init() {
  "${PY}" - "${SEQ_ROOT}" "${DRY_RUN}" "${MAX_MODEL_LEN}" <<'PY'
import json, sys, time
from pathlib import Path

seq_root, dry_run, max_model_len = sys.argv[1], sys.argv[2], sys.argv[3]
p = Path(seq_root) / "sequence_state.json"
doc = {
    "seq_root": seq_root,
    "model": "GLM-4.7-FP8",
    "dry_run": bool(int(dry_run)),
    "max_model_len": int(max_model_len),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "flavors": ["rag2_skills", "rag2_ns", "rag_skills", "skills"],
    "endpoint_url": None,
    "glm_job_id": None,
    "campaigns": {},
    "sequence_status": "running",
}
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_state_set() {
  local key="$1" value="$2"
  "${PY}" - "${STATE_JSON}" "${key}" "${value}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc[sys.argv[2]] = sys.argv[3]
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

campaign_prefix_for_flavor() {
  case "$1" in
    rag2_skills) echo "batch_parallel_chathls_fd_glm_rag2" ;;
    rag2_ns) echo "batch_parallel_chathls_fd_glm_rag2_ns" ;;
    rag_skills) echo "batch_parallel_chathls_fd_glm_rag" ;;
    skills) echo "batch_parallel_chathls_fd_glm_skills" ;;
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

cleanup_glm() {
  if [[ -n "${GLM_JOB_ID:-}" && "${DRY_RUN}" -ne 1 ]]; then
    echo "scancel GLM serve job ${GLM_JOB_ID}"
    scancel "${GLM_JOB_ID}" 2>/dev/null || true
    _state_set "glm_job_cancelled" "1"
  fi
}

trap cleanup_glm EXIT

_state_init

# --- 1. Shared GLM serve (or fake endpoint for --dry-run) -----------------
GLM_JOB_ID=""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] writing fake llm_endpoint.json (no GLM sbatch)"
  "${PY}" - "${ENDPOINT_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
p.write_text(json.dumps({
    "url": "http://127.0.0.1:8000/v1",
    "model": "GLM-4.7-FP8",
    "job_id": None,
    "borrowed": False,
    "external_llm": True,
    "max_model_len": 202752,
    "dry_run": True,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2) + "\n")
PY
else
  echo "submitting GLM serve job (${GLM_SERVE_SCRIPT}) ..."
  # Need PATH/modules via ALL for srun/bash. Clear inherited venv vars so the
  # serve script's absolute GLM47_ROOT/.venv path is used (not c2hls).
  unset VENV VIRTUAL_ENV || true
  GLM_JOB_ID="$(
    sbatch --parsable \
      --chdir="${GLM47_ROOT}" \
      --time="${GLM_WALLTIME}" \
      --export=ALL,C2HLS_GLM_ENDPOINT_DIR="${SEQ_ROOT}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",API_PORT="${API_PORT:-8000}",GLM47_ROOT="${GLM47_ROOT}",VENV="${GLM47_ROOT}/.venv-vllm-0.25.1-cu130" \
      "${GLM_SERVE_SCRIPT}"
  )"
  echo "glm_job_id=${GLM_JOB_ID}"
  _state_set "glm_job_id" "${GLM_JOB_ID}"
  echo "${GLM_JOB_ID}" > "${SEQ_ROOT}/glm_job_id.txt"

  echo "waiting for ${ENDPOINT_JSON} and /v1/models (timeout ${READY_TIMEOUT_SEC}s) ..."
  elapsed=0
  while [[ "${elapsed}" -lt "${READY_TIMEOUT_SEC}" ]]; do
    if [[ -f "${ENDPOINT_JSON}" ]]; then
      URL="$(
        "${PY}" -c "import json; print(json.load(open('${ENDPOINT_JSON}'))['url'])"
      )"
      if curl -sf --max-time 10 "${URL}/models" >/dev/null 2>&1; then
        echo "GLM endpoint healthy: ${URL}"
        break
      fi
      echo "[$(date -Is)] endpoint file present but /models not ready yet (${URL})"
    else
      # Fail fast if the serve job vanished from the queue.
      if ! squeue -j "${GLM_JOB_ID}" -h >/dev/null 2>&1; then
        # Job may have completed writing then exited, or failed — check file once more.
        if [[ ! -f "${ENDPOINT_JSON}" ]]; then
          echo "ERROR: GLM job ${GLM_JOB_ID} left the queue before writing llm_endpoint.json" >&2
          echo "  check: ${GLM47_ROOT}/glm47_c2hls-${GLM_JOB_ID}.log" >&2
          _state_finish "failed"
          exit 1
        fi
      fi
      echo "[$(date -Is)] waiting for llm_endpoint.json (elapsed ${elapsed}s)"
    fi
    sleep "${READY_POLL_SEC}"
    elapsed=$((elapsed + READY_POLL_SEC))
  done

  if [[ ! -f "${ENDPOINT_JSON}" ]]; then
    echo "ERROR: timed out waiting for GLM llm_endpoint.json" >&2
    _state_finish "failed"
    exit 1
  fi
fi

URL="$(
  "${PY}" -c "import json; print(json.load(open('${ENDPOINT_JSON}'))['url'])"
)"
echo "endpoint_url=${URL}"
_state_set "endpoint_url" "${URL}"

# --- 2. Flavors -----------------------------------------------------------
SEQUENCE_FAILED=0

for flavor in rag2_skills rag2_ns rag_skills skills; do
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

  "${SCRIPT_DIR}/start_chathls_glm_one.sh" "${ONE_ARGS[@]}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would wait for ${CAMPAIGN_ROOT}/campaign.json campaign_status"
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
echo "=== ChatHLS GLM-4.7 U280 sequence done ==="
echo "seq_root=${SEQ_ROOT}"
echo "sequence_state=${STATE_JSON}"

if [[ "${SEQUENCE_FAILED}" -eq 1 ]]; then
  exit 1
fi
