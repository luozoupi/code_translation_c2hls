#!/usr/bin/env bash
# Relaunch Devstral-2 then GLM-4.7 latency-opt A/B (skip DeepSeek — already done).
#
# Usage:
#   ./scripts/pc2/start_u280_latency_opt_dv_glm_relaunch.sh [--dry-run]
#
# Fixes vs prior A/B:
#   - PC2_BATCH_PARALLEL_WALLTIME=48h exported for Devstral (honored by campaign start)
#   - GLM ready timeout default 24h (2x4 H100 often queues long)
#   - New stamp; does not wipe DeepSeek results
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

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
STATUS_POLL_SEC="${C2HLS_AB_STATUS_POLL_SEC:-120}"
STAMP_BASE="$(date -u +%Y%m%d_%H%M%S)"
SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/latency_opt_dv_glm_relaunch_${STAMP_BASE}"
mkdir -p "${SEQ_ROOT}"
STATE_JSON="${SEQ_ROOT}/ab_state.json"
SEQ_LOG="${SEQ_ROOT}/orchestrator.log"

exec > >(tee -a "${SEQ_LOG}") 2>&1

echo "=== U280 latency-opt Devstral+GLM relaunch (rag2_skills ± lat) ==="
echo "seq_root=${SEQ_ROOT}"
echo "dry_run=${DRY_RUN}"

# Devstral walltime: must be set before start_chathls_devstral_rag2_one.sh
export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
# GLM often waits hours for 2x4 H100
export C2HLS_GLM_READY_TIMEOUT_SEC="${C2HLS_GLM_READY_TIMEOUT_SEC:-86400}"
export C2HLS_GLM_READY_POLL_SEC="${C2HLS_GLM_READY_POLL_SEC:-60}"

"${PY}" - "${SEQ_ROOT}" "${DRY_RUN}" "${STAMP_BASE}" <<'PY'
import json, sys, time
from pathlib import Path
root, dry, stamp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
doc = {
    "seq_root": str(root),
    "flavor": "rag2_skills",
    "arms": ["ctrl", "lat"],
    "models": ["devstral2", "glm47"],
    "dry_run": bool(int(dry)),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stamp_base": stamp,
    "notes": "relaunch after failed 20260719_095037 Devstral/GLM; DeepSeek skipped",
    "models_state": {
        "devstral2": {"status": "pending", "campaigns": {}},
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
doc.setdefault("models_state", {}).setdefault(sys.argv[2], {})[sys.argv[3]] = sys.argv[4]
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
doc.setdefault("models_state", {}).setdefault(m, {}).setdefault("campaigns", {}).setdefault(arm, {})[key] = value
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

# Fail the wait if campaign completed with zero useful flash selections.
campaign_usable() {
  local campaign_root="$1"
  "${PY}" - "${campaign_root}" <<'PY'
import json, sqlite3, sys
from pathlib import Path
root = Path(sys.argv[1])
locks_failed = 0
locks_done = 0
db = root / "queue.db"
if db.is_file():
    con = sqlite3.connect(db)
    locks_failed = con.execute("SELECT count(*) FROM bench_lock WHERE bench_status='failed'").fetchone()[0]
    locks_done = con.execute("SELECT count(*) FROM bench_lock WHERE bench_status='done'").fetchone()[0]
    con.close()
flash = 0
for p in (root / "variants").glob("*/*/*/*_selected.cpp"):
    flash += 1
for p in (root / "flash_selected").glob("*/selected"):
    flash += 1
# usable if any flash selected or any bench done
print("1" if (flash > 0 or locks_done > 0) else "0")
print(f"flash={flash} done={locks_done} failed={locks_failed}", file=sys.stderr)
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
      complete|completed)
        if [[ "$(campaign_usable "${campaign_root}" 2>/dev/null | head -1)" != "1" ]]; then
          echo "ERROR: ${label} completed but unusable (no flash/done benches)" >&2
          return 1
        fi
        return 0
        ;;
      failed|aborted) return 1 ;;
    esac
    sleep "${STATUS_POLL_SEC}"
  done
}

EXTRA_ONE=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ONE+=(--dry-run)
fi

# ---------------------------------------------------------------------------
# Devstral-2 A/B
# ---------------------------------------------------------------------------
_state_set_model devstral2 status running
echo "Devstral walltime FORCE=${PC2_FORCE_WALLTIME} BATCH_PARALLEL=${PC2_BATCH_PARALLEL_WALLTIME}"
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
  if ! wait_campaign "devstral2/${arm}" "${campaign_root}"; then
    _state_set_campaign devstral2 "${arm}" status failed
    _state_set_model devstral2 status failed
    echo "ERROR: Devstral ${arm} failed/unusable — aborting before GLM" >&2
    exit 2
  fi
  _state_set_campaign devstral2 "${arm}" status "$(campaign_status "${campaign_root}")"
done
_state_set_model devstral2 status complete
echo "=== Devstral-2 A/B done ==="

# ---------------------------------------------------------------------------
# GLM-4.7 A/B
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
  echo "submitting GLM serve job (ready timeout=${C2HLS_GLM_READY_TIMEOUT_SEC}s) ..."
  GLM_JOB_ID="$(
    sbatch --parsable \
      --chdir="${GLM47_ROOT}" \
      --time="${GLM_WALLTIME}" \
      --export=ALL,C2HLS_GLM_ENDPOINT_DIR="${GLM_ROOT}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",API_PORT="${API_PORT:-8000}",GLM47_ROOT="${GLM47_ROOT}",VENV="${GLM47_ROOT}/.venv-vllm-0.25.1-cu130" \
      "${GLM_SERVE_SCRIPT}"
  )"
  _state_set_model glm47 glm_job_id "${GLM_JOB_ID}"
  echo "GLM job_id=${GLM_JOB_ID}; waiting for endpoint at ${ENDPOINT_JSON} ..."
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
    # surface queue state periodically
    if (( ( $(date +%s) - start_ts ) % 600 < C2HLS_GLM_READY_POLL_SEC )); then
      squeue -j "${GLM_JOB_ID}" -o '%i %T %M %R' 2>/dev/null || true
    fi
    now="$(date +%s)"
    if (( now - start_ts > C2HLS_GLM_READY_TIMEOUT_SEC )); then
      echo "ERROR: GLM endpoint not ready within ${C2HLS_GLM_READY_TIMEOUT_SEC}s" >&2
      exit 2
    fi
    sleep "${C2HLS_GLM_READY_POLL_SEC}"
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
  if ! wait_campaign "glm47/${arm}" "${campaign_root}"; then
    _state_set_campaign glm47 "${arm}" status failed
    _state_set_model glm47 status failed
    echo "ERROR: GLM ${arm} failed/unusable" >&2
    exit 2
  fi
  _state_set_campaign glm47 "${arm}" status "$(campaign_status "${campaign_root}")"
done
_state_set_model glm47 status complete
cleanup_glm
trap - EXIT

"${PY}" - "${STATE_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc["sequence_status"] = "complete"
doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "=== Devstral+GLM relaunch complete ==="
echo "seq_root=${SEQ_ROOT}"
