#!/usr/bin/env bash
# Resume DeepSeek corpus A/B after MT completed and the main orchestrator crashed.
# Adopts existing hlsfactory ctrl (if provided), then runs lat + remaining suites.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

SEQ_ROOT="${1:-${C2HLS_ROOT}/artifacts/pc2/latency_opt_corpus_ab_20260721_073836}"
HF_CTRL="${2:-${C2HLS_ROOT}/artifacts/pc2/batch_parallel_hlsfactory_ds_rag2_20260721_123506}"
STATE_JSON="${SEQ_ROOT}/ab_state.json"
LOG="${SEQ_ROOT}/resume_orchestrator.log"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
STATUS_POLL_SEC="${C2HLS_AB_STATUS_POLL_SEC:-120}"

export C2HLS_DEEPSEEK_SKIP_PEAK="${C2HLS_DEEPSEEK_SKIP_PEAK:-1}"
export C2HLS_DEEPSEEK_PEAK_PAUSE="${C2HLS_DEEPSEEK_PEAK_PAUSE:-1}"
export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"

exec > >(tee -a "${LOG}") 2>&1

ENDPOINT_URL="$("${PY}" -c "import json; print(json.load(open('${STATE_JSON}'))['suites_state']['_meta']['endpoint_url'])")"
echo "=== resume corpus A/B ==="
echo "seq=${SEQ_ROOT}"
echo "endpoint=${ENDPOINT_URL}"
echo "hf_ctrl=${HF_CTRL}"
curl -sf "${ENDPOINT_URL}/models" >/dev/null

_state_set() {
  local suite="$1" key="$2" value="$3"
  "${PY}" - "${STATE_JSON}" "${suite}" "${key}" "${value}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc.setdefault("suites_state", {}).setdefault(sys.argv[2], {})[sys.argv[3]] = sys.argv[4]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

campaign_status() {
  "${PY}" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("campaign_status", "missing") if p.is_file() else "missing")
PY
}

campaign_usable() {
  "${PY}" - "$1" <<'PY'
import sqlite3, sys
from pathlib import Path
root = Path(sys.argv[1])
flash = sum(1 for _ in (root / "variants").glob("*/*/*/*_selected.cpp"))
done = 0
db = root / "queue.db"
if db.is_file():
    con = sqlite3.connect(db)
    done = con.execute("SELECT count(*) FROM bench_lock WHERE bench_status='done'").fetchone()[0]
    con.close()
print("1" if flash > 0 or done > 0 else "0")
PY
}

wait_campaign() {
  local label="$1" root="$2"
  echo "waiting for ${label} ..."
  while true; do
    st="$(campaign_status "${root}")"
    echo "[$(date -Is)] ${label} status=${st}"
    case "${st}" in
      complete|completed)
        if [[ "$(campaign_usable "${root}")" != 1 ]]; then
          echo "ERROR: ${label} complete but unusable" >&2
          return 1
        fi
        return 0
        ;;
      failed|aborted)
        echo "ERROR: ${label} ${st}" >&2
        return 1
        ;;
    esac
    sleep "${STATUS_POLL_SEC}"
  done
}

plant_skip_peak() {
  local root="$1"
  [[ -f "${root}/campaign.json" ]] || return 0
  "${PY}" - "${root}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text())
doc["skip_peak_pause"] = True
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
}

resolve_root() {
  local suite="$1" arm="$2" stamp="$3"
  local base
  if [[ "${suite}" == "machsuite" ]]; then
    base="batch_parallel_machsuite_ds_rag2"
  elif [[ "${suite}" == "hlsfactory" ]]; then
    base="batch_parallel_hlsfactory_ds_rag2"
  else
    base="batch_parallel_${suite}_ds_rag2"
  fi
  if [[ "${arm}" == "lat" ]]; then
    base="${base}_lat"
  fi
  echo "${C2HLS_ROOT}/artifacts/pc2/${base}_${stamp}"
}

run_arm() {
  local suite="$1" arm="$2" starter="$3"
  shift 3
  local stamp lat_flag=()
  stamp="$(date -u +%Y%m%d_%H%M%S)_${suite}_${arm}"
  if [[ "${arm}" == "lat" ]]; then
    lat_flag=(--latency-opt)
  fi
  echo "submitting ${suite}/${arm} stamp=${stamp}"
  "${starter}" \
    --stamp "${stamp}" \
    --endpoint-url "${ENDPOINT_URL}" \
    "${lat_flag[@]}" \
    "$@"
  local cr
  cr="$(resolve_root "${suite}" "${arm}" "${stamp}")"
  _state_set "${suite}" "${arm}_root" "${cr}"
  plant_skip_peak "${cr}"
  wait_campaign "${suite}/${arm}" "${cr}"
}

# --- hlsfactory: adopt ctrl, then lat ---
_state_set hlsfactory status running
_state_set hlsfactory ctrl_root "${HF_CTRL}"
plant_skip_peak "${HF_CTRL}"
wait_campaign "hlsfactory/ctrl" "${HF_CTRL}"
run_arm hlsfactory lat "${SCRIPT_DIR}/start_hlsfactory_deepseek_rag2_skills_u280.sh"
_state_set hlsfactory status complete

# --- remaining corpora ---
run_suite() {
  local suite="$1" starter="$2"
  shift 2
  _state_set "${suite}" status running
  run_arm "${suite}" ctrl "${starter}" "$@"
  run_arm "${suite}" lat "${starter}" "$@"
  _state_set "${suite}" status complete
}

run_suite machsuite "${SCRIPT_DIR}/start_machsuite_deepseek_rag2_skills_u280.sh"
for suite in forgebench hp_fft spector; do
  run_suite "${suite}" "${SCRIPT_DIR}/start_tier_a_deepseek_rag2_skills_u280.sh" --suite "${suite}"
done

"${PY}" - "${STATE_JSON}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
doc = json.loads(p.read_text())
doc["sequence_status"] = "complete"
doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "=== resume complete ==="
echo "seq_root=${SEQ_ROOT}"
