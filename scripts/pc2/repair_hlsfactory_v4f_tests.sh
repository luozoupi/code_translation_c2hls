#!/usr/bin/env bash
# Attempt auto-repair for HLSFactory v4f sequence health failures.
# Usage: ./scripts/pc2/repair_hlsfactory_v4f_tests.sh SEQ_ROOT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

SEQ_ROOT="${1:?seq root}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3
LOG="${SEQ_ROOT}/watch/repair.log"
mkdir -p "${SEQ_ROOT}/watch"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] repair start" | tee -a "${LOG}"

# Ensure API key
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
  # shellcheck disable=SC1091
  source /scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/scripts/pc2/setup_deepseek_api.sh
fi

declare -A PORTS=([skills]=18092 [noskills]=18093 [bare]=18094)

restart_proxy() {
  local FLAVOR="$1"
  local PORT="${PORTS[${FLAVOR}]}"
  local PROXY_DIR="${SEQ_ROOT}/proxy_${FLAVOR}"
  mkdir -p "${PROXY_DIR}"
  echo "[repair] restarting proxy ${FLAVOR} :${PORT}" | tee -a "${LOG}"
  # kill listeners on port if any
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
  pkill -f "CHATHLS_DEEPSEEK_PROXY_PORT=${PORT}" 2>/dev/null || true
  sleep 2
  DEEPSEEK_PROXY_MODEL=deepseek-v4-flash \
    C2HLS_MODEL=deepseek-v4-flash \
    CHATHLS_DEEPSEEK_PROXY_PORT="${PORT}" \
    CHATHLS_DEEPSEEK_QUEUE_WORKERS=1 \
    "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${PROXY_DIR}" | tee -a "${LOG}"
  local URL
  URL="$("${PY}" -c "import json;print(json.load(open('${PROXY_DIR}/llm_endpoint.json'))['url'])")"
  curl -sf --max-time 10 "${URL}/models" >/dev/null
  echo "[repair] proxy ${FLAVOR} OK ${URL}" | tee -a "${LOG}"
}

for FLAVOR in skills noskills bare; do
  EP="${SEQ_ROOT}/proxy_${FLAVOR}/llm_endpoint.json"
  NEED=0
  if [[ ! -f "${EP}" ]]; then
    NEED=1
  else
    URL="$("${PY}" -c "import json;print(json.load(open('${EP}'))['url'])" 2>/dev/null || true)"
    if [[ -z "${URL}" ]] || ! curl -sf --max-time 8 "${URL}/models" >/dev/null 2>&1; then
      NEED=1
    fi
  fi
  if [[ "${NEED}" -eq 1 ]]; then
    restart_proxy "${FLAVOR}"
  fi
done

# If orchestrator died mid-sequence, resume remaining tests.
ORCH_PID_FILE="${SEQ_ROOT}/orchestrator.pid"
MANIFEST="${SEQ_ROOT}/sequence_manifest.json"
NEED_RESUME=0
if [[ -f "${ORCH_PID_FILE}" ]]; then
  OPID="$(cat "${ORCH_PID_FILE}")"
  if ! kill -0 "${OPID}" 2>/dev/null; then
    NEED_RESUME=1
  fi
else
  NEED_RESUME=1
fi

if [[ "${NEED_RESUME}" -eq 1 && -f "${MANIFEST}" ]]; then
  REMAINING="$("${PY}" - <<PY
import json
from pathlib import Path
seq = Path("${SEQ_ROOT}")
doc = json.loads((seq / "sequence_manifest.json").read_text())
order = ["lat_opt", "rag2_lat", "rag2"]
tests = doc.get("tests") or {}

def selection_done(tname):
    meta = tests.get(tname) or {}
    flavors = meta.get("flavors") or {}
    if len(flavors) < 3:
        return False
    for fl, m in flavors.items():
        root = Path(m.get("campaign_root") or "")
        cj = root / "campaign.json"
        if not cj.is_file():
            return False
        camp = json.loads(cj.read_text())
        if camp.get("selection_complete"):
            continue
        n = 0
        variants = root / "variants"
        if variants.is_dir():
            for cell in variants.glob("*/*/*"):
                if not cell.is_dir():
                    continue
                b = cell.parent.name
                if (cell / f"{b}_dataflow_candidate_ranking.json").is_file():
                    n += 1
        if n < 1:
            return False
    return True

todo = [t for t in order if not selection_done(t)]
print(",".join(todo))
PY
)"
  echo "[repair] remaining tests: '${REMAINING}'" | tee -a "${LOG}"
  if [[ -n "${REMAINING}" ]]; then
    # Only resume the first incomplete test via --only if orchestrator dead and campaigns not yet submitted for it.
    FIRST="${REMAINING%%,*}"
    STAMP_BASE="$("${PY}" -c "import json;print(json.load(open('${MANIFEST}')).get('stamp_base',''))")"
    # If first test already has campaign dirs, do NOT relaunch (would collide). Just note.
    HAS_CAMP=0
    if [[ -n "${STAMP_BASE}" ]]; then
      for FLAVOR in skills noskills bare; do
        case "${FLAVOR}" in
          skills) P="batch_parallel_hlsfactory_ds_v4f_skills_${FIRST}" ;;
          noskills) P="batch_parallel_hlsfactory_ds_v4f_noskills_${FIRST}" ;;
          bare) P="batch_parallel_hlsfactory_ds_v4f_bare_${FIRST}" ;;
        esac
        if [[ -d "${C2HLS_ROOT}/artifacts/pc2/${P}_${STAMP_BASE}_${FIRST}" ]]; then
          HAS_CAMP=1
        fi
      done
    fi
    if [[ "${HAS_CAMP}" -eq 1 ]]; then
      echo "[repair] campaigns already exist for ${FIRST}; not relaunching (post/proxy repair only)" | tee -a "${LOG}"
      # Resubmit missing/dead post watchers if campaign complete but selection not done
      "${PY}" - <<PY | tee -a "${LOG}"
import json, subprocess, os
from pathlib import Path
seq = Path("${SEQ_ROOT}")
doc = json.loads((seq / "sequence_manifest.json").read_text())
tests = doc.get("tests") or {}
script_dir = Path("${SCRIPT_DIR}")
for tname, tmeta in tests.items():
    for fl, m in (tmeta.get("flavors") or {}).items():
        root = Path(m["campaign_root"])
        cj = root / "campaign.json"
        if not cj.is_file():
            continue
        camp = json.loads(cj.read_text())
        if camp.get("selection_complete"):
            continue
        status = camp.get("campaign_status")
        post = str(camp.get("post_watcher_job_id") or "")
        alive = False
        if post:
            r = subprocess.run(["squeue", "-j", post, "-h", "-o", "%T"], capture_output=True, text=True)
            alive = bool(r.stdout.strip())
        if status in {"complete", "completed"} and not alive and not camp.get("selection_complete"):
            print(f"NEED_POST_RESUBMIT {tname} {fl} {root}")
PY
    else
      echo "[repair] relaunching missing test arm --only ${FIRST}" | tee -a "${LOG}"
      nohup bash "${SCRIPT_DIR}/start_hlsfactory_v4f_test_sequence.sh" \
        --stamp "${STAMP_BASE}" \
        --only "${FIRST}" \
        >> "${SEQ_ROOT}/launch_resume_${FIRST}.log" 2>&1 &
      echo $! > "${SEQ_ROOT}/orchestrator.pid"
      echo "[repair] resume orchestrator pid=$(cat "${SEQ_ROOT}/orchestrator.pid")" | tee -a "${LOG}"
    fi
  else
    echo "[repair] all tests selection-done; nothing to resume" | tee -a "${LOG}"
  fi
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] repair done" | tee -a "${LOG}"
