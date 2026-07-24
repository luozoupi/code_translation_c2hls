#!/usr/bin/env bash
# U280 latency-opt A/B orchestrator (rag2_skills ± latency_opt).
#
# Schedule:
#   1) Devstral-2 A/B  (ctrl then lat, sequential — 1 GPU)
#   2) DeepSeek A/B    (ctrl then lat, shared login proxy) — in parallel with (1)
#   3) After Devstral campaigns finish → GLM-4.7 A/B (ctrl then lat)
#
# Usage:
#   ./scripts/pc2/start_u280_latency_opt_ab.sh [--dry-run] [--skip-peak-wait]
#
# No git commit. Arm = rag2_skills only (control vs +latency_opt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
SKIP_PEAK_WAIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-peak-wait) SKIP_PEAK_WAIT=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-python3}"
STATUS_POLL_SEC="${C2HLS_AB_STATUS_POLL_SEC:-120}"
STAMP_BASE="$(date -u +%Y%m%d_%H%M%S)"
SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/latency_opt_ab_${STAMP_BASE}"
mkdir -p "${SEQ_ROOT}"
STATE_JSON="${SEQ_ROOT}/ab_state.json"
SEQ_LOG="${SEQ_ROOT}/orchestrator.log"

exec > >(tee -a "${SEQ_LOG}") 2>&1

echo "=== U280 latency-opt A/B (rag2_skills ± lat) ==="
echo "seq_root=${SEQ_ROOT}"
echo "dry_run=${DRY_RUN} skip_peak_wait=${SKIP_PEAK_WAIT}"

"${PY}" - "${SEQ_ROOT}" "${DRY_RUN}" "${STAMP_BASE}" <<'PY'
import json, sys, time
from pathlib import Path
root, dry, stamp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
doc = {
    "seq_root": str(root),
    "flavor": "rag2_skills",
    "arms": ["ctrl", "lat"],
    "dry_run": bool(int(dry)),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stamp_base": stamp,
    "models": {
        "devstral2": {"status": "pending", "campaigns": {}},
        "deepseek": {"status": "pending", "campaigns": {}},
        "glm47": {"status": "pending", "campaigns": {}},
    },
    "sequence_status": "running",
}
(root / "ab_state.json").write_text(json.dumps(doc, indent=2) + "\n")
PY

_state_set_model() {
  local model="$1" key="$2" value="$3"
  "${PY}" - "${STATE_JSON}" "${model}" "${key}" "${value}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc["models"][sys.argv[2]][sys.argv[3]] = sys.argv[4]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

_state_set_campaign() {
  local model="$1" arm="$2" key="$3" value="$4"
  "${PY}" - "${STATE_JSON}" "${model}" "${arm}" "${key}" "${value}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
m, arm, key, value = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
doc["models"].setdefault(m, {}).setdefault("campaigns", {}).setdefault(arm, {})[key] = value
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
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

wait_campaign() {
  local label="$1" campaign_root="$2"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] skip wait for ${label} (${campaign_root})"
    return 0
  fi
  echo "waiting for ${label} campaign_status ... (${campaign_root})"
  while true; do
    st="$(campaign_status "${campaign_root}")"
    echo "[$(date -Is)] ${label} status=${st}"
    case "${st}" in
      complete|completed|failed|aborted) return 0 ;;
    esac
    sleep "${STATUS_POLL_SEC}"
  done
}

EXTRA_ONE=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ONE+=(--dry-run)
fi
PEAK_ARGS=()
if [[ "${SKIP_PEAK_WAIT}" -eq 1 ]]; then
  PEAK_ARGS+=(--skip-peak-wait)
fi

# ---------------------------------------------------------------------------
# DeepSeek A/B (background): shared proxy, ctrl then lat
# ---------------------------------------------------------------------------
DS_LOG="${SEQ_ROOT}/deepseek_ab.log"
(
  set -euo pipefail
  echo "=== DeepSeek A/B start ==="
  DS_ROOT="${SEQ_ROOT}/deepseek"
  mkdir -p "${DS_ROOT}"
  ENDPOINT_JSON="${DS_ROOT}/llm_endpoint.json"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    "${PY}" - "${ENDPOINT_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "url": "http://127.0.0.1:18092/v1",
    "model": "deepseek-chat",
    "dry_run": True,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2) + "\n")
PY
  else
    # External-llm DeepSeek runs need a real upstream key on the login proxy.
    if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
      unset OPENAI_API_KEY || true
    fi
    CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
    # shellcheck disable=SC1091
    source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
    bash "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${DS_ROOT}"
  fi
  URL="$("${PY}" -c "import json; print(json.load(open('${ENDPOINT_JSON}'))['url'])")"
  echo "deepseek endpoint=${URL}"

  if [[ "${DRY_RUN}" -eq 0 && "${SKIP_PEAK_WAIT}" -eq 0 ]]; then
    while "${PY}" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from deepseek_peak import is_beijing_peak
raise SystemExit(0 if is_beijing_peak() else 1)
"; do
      echo "[$(date -Is)] Beijing peak — sleep before DeepSeek campaign"
      sleep "${C2HLS_DEEPSEEK_PEAK_POLL_SEC:-300}"
    done
  fi

  for arm in ctrl lat; do
    stamp="${STAMP_BASE}_ds_${arm}"
    LAT_FLAG=()
    PREFIX="batch_parallel_chathls_fd_ds_rag2"
    if [[ "${arm}" == "lat" ]]; then
      LAT_FLAG=(--latency-opt)
      PREFIX="${PREFIX}_lat"
    fi
    campaign_root="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${stamp}"
    echo "submitting DeepSeek arm=${arm} stamp=${stamp}"
    "${SCRIPT_DIR}/start_chathls_deepseek_one.sh" \
      --flavor rag2_skills \
      --stamp "${stamp}" \
      --endpoint-url "${URL}" \
      "${LAT_FLAG[@]}" \
      "${EXTRA_ONE[@]}"
    _state_set_campaign deepseek "${arm}" campaign_root "${campaign_root}"
    _state_set_campaign deepseek "${arm}" stamp "${stamp}"
    wait_campaign "deepseek/${arm}" "${campaign_root}"
    _state_set_campaign deepseek "${arm}" status "$(campaign_status "${campaign_root}")"
  done
  _state_set_model deepseek status complete
  echo "=== DeepSeek A/B done ==="
) >"${DS_LOG}" 2>&1 &
DS_PID=$!
echo "DeepSeek A/B background pid=${DS_PID} log=${DS_LOG}"
_state_set_model deepseek status running
_state_set_model deepseek pid "${DS_PID}"

# ---------------------------------------------------------------------------
# Devstral-2 A/B (foreground): ctrl then lat
# ---------------------------------------------------------------------------
_state_set_model devstral2 status running
for arm in ctrl lat; do
  stamp="${STAMP_BASE}_dv_${arm}"
  LAT_FLAG=()
  PREFIX="batch_parallel_chathls_fd_rag2"
  if [[ "${arm}" == "lat" ]]; then
    LAT_FLAG=(--latency-opt)
    PREFIX="${PREFIX}_lat"
  fi
  campaign_root="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${stamp}"
  echo "submitting Devstral-2 arm=${arm} stamp=${stamp}"
  "${SCRIPT_DIR}/start_chathls_devstral_rag2_one.sh" \
    --flavor rag2_skills \
    --stamp "${stamp}" \
    "${LAT_FLAG[@]}" \
    "${EXTRA_ONE[@]}"
  _state_set_campaign devstral2 "${arm}" campaign_root "${campaign_root}"
  _state_set_campaign devstral2 "${arm}" stamp "${stamp}"
  wait_campaign "devstral2/${arm}" "${campaign_root}"
  _state_set_campaign devstral2 "${arm}" status "$(campaign_status "${campaign_root}")"
done
_state_set_model devstral2 status complete
echo "=== Devstral-2 A/B done ==="

# ---------------------------------------------------------------------------
# GLM-4.7 A/B after Devstral finishes
# ---------------------------------------------------------------------------
_state_set_model glm47 status running
GLM_ROOT="${SEQ_ROOT}/glm"
mkdir -p "${GLM_ROOT}"
ENDPOINT_JSON="${GLM_ROOT}/llm_endpoint.json"
GLM47_ROOT="${GLM47_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/glm4.7}"
GLM_SERVE_SCRIPT="${GLM47_ROOT}/run_glm47_serve_for_c2hls.slurm"
GLM_WALLTIME="${GLM_WALLTIME:-7-00:00:00}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-202752}"
GLM_JOB_ID=""

cleanup_glm() {
  if [[ -n "${GLM_JOB_ID}" && "${DRY_RUN}" -eq 0 ]]; then
    echo "scancel GLM serve job ${GLM_JOB_ID}"
    scancel "${GLM_JOB_ID}" || true
  fi
}
trap cleanup_glm EXIT

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${PY}" - "${ENDPOINT_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "url": "http://127.0.0.1:8000/v1",
    "model": "GLM-4.7-FP8",
    "dry_run": True,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2) + "\n")
PY
  URL="http://127.0.0.1:8000/v1"
else
  echo "submitting GLM serve job ..."
  GLM_JOB_ID="$(
    sbatch --parsable \
      --chdir="${GLM47_ROOT}" \
      --time="${GLM_WALLTIME}" \
      --export=ALL,C2HLS_GLM_ENDPOINT_DIR="${GLM_ROOT}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",API_PORT="${API_PORT:-8000}",GLM47_ROOT="${GLM47_ROOT}",VENV="${GLM47_ROOT}/.venv-vllm-0.25.1-cu130" \
      "${GLM_SERVE_SCRIPT}"
  )"
  _state_set_model glm47 glm_job_id "${GLM_JOB_ID}"
  echo "GLM job_id=${GLM_JOB_ID}; waiting for endpoint at ${ENDPOINT_JSON} ..."
  READY_TIMEOUT_SEC="${C2HLS_GLM_READY_TIMEOUT_SEC:-10800}"
  READY_POLL_SEC="${C2HLS_GLM_READY_POLL_SEC:-30}"
  start_ts="$(date +%s)"
  while true; do
    if [[ -f "${ENDPOINT_JSON}" ]]; then
      URL="$("${PY}" -c "import json; print(json.load(open('${ENDPOINT_JSON}')).get('url',''))" 2>/dev/null || true)"
      if [[ -n "${URL}" ]]; then
        if curl -sf "${URL}/models" >/dev/null 2>&1; then
          echo "GLM endpoint ready: ${URL}"
          break
        fi
      fi
    fi
    now="$(date +%s)"
    if (( now - start_ts > READY_TIMEOUT_SEC )); then
      echo "ERROR: GLM endpoint not ready within ${READY_TIMEOUT_SEC}s" >&2
      exit 2
    fi
    sleep "${READY_POLL_SEC}"
  done
fi

for arm in ctrl lat; do
  stamp="${STAMP_BASE}_glm_${arm}"
  LAT_FLAG=()
  PREFIX="batch_parallel_chathls_fd_glm_rag2"
  if [[ "${arm}" == "lat" ]]; then
    LAT_FLAG=(--latency-opt)
    PREFIX="${PREFIX}_lat"
  fi
  campaign_root="${C2HLS_ROOT}/artifacts/pc2/${PREFIX}_${stamp}"
  echo "submitting GLM arm=${arm} stamp=${stamp}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-local-glm}"
  "${SCRIPT_DIR}/start_chathls_glm_one.sh" \
    --flavor rag2_skills \
    --stamp "${stamp}" \
    --endpoint-url "${URL}" \
    "${LAT_FLAG[@]}" \
    "${EXTRA_ONE[@]}"
  _state_set_campaign glm47 "${arm}" campaign_root "${campaign_root}"
  _state_set_campaign glm47 "${arm}" stamp "${stamp}"
  wait_campaign "glm47/${arm}" "${campaign_root}"
  _state_set_campaign glm47 "${arm}" status "$(campaign_status "${campaign_root}")"
done
_state_set_model glm47 status complete
cleanup_glm
trap - EXIT

# Wait for DeepSeek background track
echo "waiting for DeepSeek A/B pid=${DS_PID} ..."
if ! wait "${DS_PID}"; then
  echo "WARNING: DeepSeek A/B background failed — see ${DS_LOG}" >&2
  _state_set_model deepseek status failed
else
  echo "DeepSeek A/B finished"
fi

"${PY}" - "${STATE_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc["sequence_status"] = "complete"
doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "=== A/B orchestrator complete ==="
echo "state=${STATE_JSON}"
echo "log=${SEQ_LOG}"
echo "deepseek_log=${DS_LOG}"
