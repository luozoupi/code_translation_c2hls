#!/usr/bin/env bash
# DeepSeek latency-opt A/B for:
#   1) ChatHLS mobilenet+transformer rerun (fixed headers/TB)
#   2) hlsfactory / machsuite / forgebench / hp_fft / spector corpora
#
# Usage:
#   ./scripts/pc2/start_u280_latency_opt_corpus_ab.sh [--dry-run]
#   ./scripts/pc2/start_u280_latency_opt_corpus_ab.sh --only mt|hlsfactory|machsuite|forgebench|hp_fft|spector
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --only) shift; ONLY="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
STATUS_POLL_SEC="${C2HLS_AB_STATUS_POLL_SEC:-120}"
STAMP_BASE="$(date -u +%Y%m%d_%H%M%S)"
SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/latency_opt_corpus_ab_${STAMP_BASE}"
mkdir -p "${SEQ_ROOT}/proxy"
STATE_JSON="${SEQ_ROOT}/ab_state.json"
SEQ_LOG="${SEQ_ROOT}/orchestrator.log"
PROXY_DIR="${SEQ_ROOT}/proxy"

export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
export C2HLS_DEEPSEEK_PEAK_PAUSE="${C2HLS_DEEPSEEK_PEAK_PAUSE:-1}"
# Operator A/B runs skip the Beijing peak gate so drain keeps codegen moving
# (same as prior DeepSeek ctrl A/B which planted skip_peak_pause=true).
export C2HLS_DEEPSEEK_SKIP_PEAK="${C2HLS_DEEPSEEK_SKIP_PEAK:-1}"

exec > >(tee -a "${SEQ_LOG}") 2>&1

echo "=== U280 DeepSeek latency-opt corpus A/B ==="
echo "seq_root=${SEQ_ROOT} dry_run=${DRY_RUN} only=${ONLY:-all} skip_peak=${C2HLS_DEEPSEEK_SKIP_PEAK}"

"${PY}" - "${SEQ_ROOT}" "${DRY_RUN}" "${STAMP_BASE}" <<'PY'
import json, sys, time
from pathlib import Path
root, dry, stamp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
doc = {
    "seq_root": str(root),
    "flavor": "rag2_skills",
    "arms": ["ctrl", "lat"],
    "suites": ["mt_chathls", "hlsfactory", "machsuite", "forgebench", "hp_fft", "spector"],
    "model": "deepseek-chat",
    "dry_run": bool(int(dry)),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stamp_base": stamp,
    "suites_state": {},
    "sequence_status": "running",
}
(root / "ab_state.json").write_text(json.dumps(doc, indent=2) + "\n")
PY

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
  local campaign_root="$1"
  "${PY}" - "${campaign_root}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
print(json.loads(p.read_text()).get("campaign_status", "missing") if p.is_file() else "missing")
PY
}

campaign_usable() {
  local campaign_root="$1"
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

plant_skip_peak() {
  local campaign_root="$1"
  [[ "${C2HLS_DEEPSEEK_SKIP_PEAK}" == "1" ]] || return 0
  [[ -f "${campaign_root}/campaign.json" ]] || return 0
  "${PY}" - "${campaign_root}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "campaign.json"
doc = json.loads(p.read_text())
doc["skip_peak_pause"] = True
p.write_text(json.dumps(doc, indent=2) + "\n")
print(f"planted skip_peak_pause on {p}")
PY
}

wait_campaign() {
  local label="$1" campaign_root="$2"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] skip wait ${label}"
    return 0
  fi
  echo "waiting for ${label} ..."
  while true; do
    st="$(campaign_status "${campaign_root}")"
    echo "[$(date -Is)] ${label} status=${st}"
    case "${st}" in
      complete|completed)
        if [[ "$(campaign_usable "${campaign_root}" | head -1)" != 1 ]]; then
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

want_suite() {
  local s="$1"
  [[ -z "${ONLY}" || "${ONLY}" == "${s}" || "${ONLY}" == "all" ]]
}

# Shared DeepSeek proxy
ENDPOINT_URL=""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18092/v1"
else
  echo "starting DeepSeek proxy in ${PROXY_DIR}"
  bash "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${PROXY_DIR}"
  ENDPOINT_URL="$("${PY}" -c "import json; print(json.load(open('${PROXY_DIR}/llm_endpoint.json'))['url'])")"
  echo "proxy ready: ${ENDPOINT_URL}"
  # smoke
  curl -sf "${ENDPOINT_URL}/models" >/dev/null
fi
_state_set _meta endpoint_url "${ENDPOINT_URL}"

EXTRA=()
if [[ "${DRY_RUN}" -eq 1 ]]; then EXTRA+=(--dry-run); fi

# --- 1) ChatHLS mobilenet+transformer ---
if want_suite mt || want_suite mt_chathls; then
  _state_set mt_chathls status running
  for arm in ctrl lat; do
    stamp="${STAMP_BASE}_mt_${arm}"
    LAT_FLAG=()
    PREFIX="batch_parallel_chathls_fd_ds_rag2"
    if [[ "${arm}" == lat ]]; then
      LAT_FLAG=(--latency-opt)
      PREFIX="${PREFIX}_lat"
    fi
    # Override config to 2-bench MT set; keep streaming dataflow from deepseek_one.
    export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_chathls_deepseek_mt_u280.json"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="${PREFIX}"
    echo "submitting ChatHLS MT arm=${arm} stamp=${stamp}"
    "${SCRIPT_DIR}/start_chathls_deepseek_one.sh" \
      --flavor rag2_skills \
      --stamp "${stamp}" \
      --endpoint-url "${ENDPOINT_URL}" \
      "${LAT_FLAG[@]}" \
      "${EXTRA[@]}"
    # MT config forces prefix via env only if start script honors ARTIFACT_PREFIX -
    # start_chathls_deepseek_one sets prefix from flavor; for lat it appends _lat.
    # Campaign root follows flavor prefix + stamp.
    if [[ "${arm}" == lat ]]; then
      CR="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_lat_${stamp}"
    else
      CR="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_${stamp}"
    fi
    # If MT config used different synth node count, campaign still under flavor prefix.
    _state_set mt_chathls "${arm}_root" "${CR}"
    plant_skip_peak "${CR}"
    if ! wait_campaign "mt/${arm}" "${CR}"; then
      _state_set mt_chathls status failed
      exit 2
    fi
  done
  _state_set mt_chathls status complete
fi

# --- 2) Corpora ---
run_corpus() {
  local suite="$1"
  local starter="$2"
  shift 2
  local extra_args=("$@")
  _state_set "${suite}" status running
  for arm in ctrl lat; do
    stamp="${STAMP_BASE}_${suite}_${arm}"
    LAT_FLAG=()
    if [[ "${arm}" == lat ]]; then LAT_FLAG=(--latency-opt); fi
    echo "submitting ${suite} arm=${arm} stamp=${stamp}"
    "${starter}" \
      --stamp "${stamp}" \
      --endpoint-url "${ENDPOINT_URL}" \
      "${LAT_FLAG[@]}" \
      "${extra_args[@]}" \
      "${EXTRA[@]}"
    # Resolve campaign root from artifact prefix conventions
    local cr
    cr="$("${PY}" - <<PY
from pathlib import Path
import json
stamp = "${stamp}"
suite = "${suite}"
lat = "${arm}" == "lat"
cands = []
if suite == "hlsfactory":
    base = "batch_parallel_hlsfactory_ds_rag2"
elif suite == "machsuite":
    base = "batch_parallel_machsuite_ds_rag2"
else:
    base = f"batch_parallel_{suite}_ds_rag2"
pref = base + ("_lat" if lat else "")
print(Path("artifacts/pc2") / f"{pref}_{stamp}")
PY
)"
    # make absolute
    cr="${C2HLS_ROOT}/${cr#./}"
    if [[ ! -d "${cr}" ]]; then
      # dry-run may still create skeleton
      cr="${C2HLS_ROOT}/artifacts/pc2/$(basename "${cr}")"
    fi
    _state_set "${suite}" "${arm}_root" "${cr}"
    plant_skip_peak "${cr}"
    if ! wait_campaign "${suite}/${arm}" "${cr}"; then
      _state_set "${suite}" status failed
      return 1
    fi
  done
  _state_set "${suite}" status complete
}

if want_suite hlsfactory; then
  run_corpus hlsfactory "${SCRIPT_DIR}/start_hlsfactory_deepseek_rag2_skills_u280.sh" || exit 2
fi
if want_suite machsuite; then
  run_corpus machsuite "${SCRIPT_DIR}/start_machsuite_deepseek_rag2_skills_u280.sh" || exit 2
fi
for suite in forgebench hp_fft spector; do
  if want_suite "${suite}"; then
    run_corpus "${suite}" "${SCRIPT_DIR}/start_tier_a_deepseek_rag2_skills_u280.sh" --suite "${suite}" || exit 2
  fi
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

echo "=== corpus A/B complete ==="
echo "seq_root=${SEQ_ROOT}"
