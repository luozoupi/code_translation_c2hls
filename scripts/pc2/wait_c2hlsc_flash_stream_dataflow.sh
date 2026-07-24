#!/usr/bin/env bash
# Stream c2hlsc flash→dataflow: when a bench's flash is selected, start dataflow
# for that bench immediately (do NOT wait for all flash to finish).
#
# Dataflow LLM uses the campaign GPU endpoint. Multiple benches can run
# concurrently (Vitis csim/csynth/cosim parallelism); LLM requests share the
# single vLLM endpoint.
#
# Skills: dataflow uses flash_no_RMW_m_axi_skill_entries.json (default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
pc2_setup_vitis_env
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
POLL_SEC="${POST_FLASH_POLL_SEC:-60}"
MAX_PARALLEL="${C2HLS_DATAFLOW_MAX_PARALLEL:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --poll-sec) shift; POLL_SEC="$1"; shift ;;
    --max-parallel) shift; MAX_PARALLEL="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

FLOW_DIR="${CAMPAIGN_ROOT}/flow"
STATE_DIR="${FLOW_DIR}/streaming_dataflow"
mkdir -p "${STATE_DIR}/logs" "${CAMPAIGN_ROOT}/reports"
STATE_JSON="${STATE_DIR}/state.json"
FLASH_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"
DATAFLOW_BUNDLE="${C2HLS_ROOT}/artifacts/pc2/dataflow_selected_bundle/$(basename "${CAMPAIGN_ROOT}")"

export C2HLS_RUN_COSIM="${C2HLS_RUN_COSIM:-1}"
export C2HLS_COSIM_REQUIRED="${C2HLS_COSIM_REQUIRED:-0}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-c2hlsc_stream_cosim}"
export C2HLS_POST_FLASH_MATRIX_ROOT="${CAMPAIGN_ROOT}"
# Nest HLS scratch: c2hls_tmp/<campaign_dirname>/<bench>/hls_*
export C2HLS_TMP_RUN="$(basename "${CAMPAIGN_ROOT}")"
# Explicit dataflow skills file (no_RMW overlay entries).
export C2HLS_DATAFLOW_SKILL_ENTRIES_JSON="${C2HLS_DATAFLOW_SKILL_ENTRIES_JSON:-${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/flash_no_RMW_m_axi_skill_entries.json}"

# RAG scrape (optional) — mirrors starter exports when set.
SCRAPE_CORPUS="${C2HLS_RAG_SCRAPE_CORPUS:-${C2HLS_ROOT}/artifacts/rag/knowledge_repo}"
RAG_DATAFLOW_ARGS=()
if [[ "${C2HLS_RAG:-0}" == "1" ]] || [[ "${C2HLS_RAG_ENABLE:-0}" == "1" ]]; then
  RAG_DATAFLOW_ARGS+=(--rag --rag-mode "${C2HLS_RAG_MODE:-everywhere}")
  if [[ "${C2HLS_RAG_SCRAPE:-0}" == "1" ]]; then
    RAG_DATAFLOW_ARGS+=(--scrape --scrape-corpus "${SCRAPE_CORPUS}")
  fi
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] streaming flash→dataflow watcher"
echo "campaign=${CAMPAIGN_ROOT}"
echo "max_parallel=${MAX_PARALLEL} poll_sec=${POLL_SEC}"
echo "dataflow_skills=${C2HLS_DATAFLOW_SKILL_ENTRIES_JSON}"
echo "tmp_run=${C2HLS_TMP_RUN} rag_args=${RAG_DATAFLOW_ARGS[*]:-none}"

# Bootstrap empty state
if [[ ! -f "${STATE_JSON}" ]]; then
  echo '{"started":{},"done":{},"failed":{}}' > "${STATE_JSON}"
fi

wait_for_endpoint() {
  local ep deadline
  ep="${CAMPAIGN_ROOT}/llm_endpoint.json"
  deadline=$((SECONDS + 14400))
  while (( SECONDS < deadline )); do
    if [[ -f "${ep}" ]]; then
      OPENAI_BASE_URL="$("${PY}" -c "import json;print(json.load(open('${ep}'))['url'])")"
      export OPENAI_BASE_URL
      export CHATHLS_API_BASE="${OPENAI_BASE_URL}"
      if curl -sf --max-time 5 "${OPENAI_BASE_URL}/models" >/dev/null; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] endpoint ready ${OPENAI_BASE_URL}"
        return 0
      fi
    fi
    sleep 30
  done
  echo "ERROR: timed out waiting for ${ep}" >&2
  return 1
}

wait_for_endpoint

rebuild_matrix() {
  "${PY}" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("${CAMPAIGN_ROOT}")
rows = []
for cell in sorted((root / "variants").glob("*/*/*")):
    if not cell.is_dir():
        continue
    bench = cell.parent.name
    has_sel = any(cell.glob(f"{bench}_selected.cpp")) or any(cell.glob(f"{bench}_final.cpp"))
    has_res = (cell / f"{bench}_multistep_results.json").is_file()
    if not (has_sel or has_res or (cell / "reference_validation.json").is_file()):
        continue
    rows.append({
        "bench": bench,
        "cell_dir": str(cell.resolve()),
        "status": "unknown",
        "model": cell.name,
        "variant": cell.parent.parent.name,
    })
(root / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(len(rows))
PY
}

flash_ready_benches() {
  "${PY}" - <<'PY'
import json
from pathlib import Path
import os

root = Path(os.environ["CAMPAIGN_ROOT"])
ready = []
for cell in sorted((root / "variants").glob("*/*/*")):
    if not cell.is_dir():
        continue
    bench = cell.parent.name
    sel = list(cell.glob(f"{bench}_selected.cpp")) + list(cell.glob(f"{bench}_final.cpp"))
    res_path = cell / f"{bench}_multistep_results.json"
    ok = False
    if res_path.is_file():
        try:
            d = json.loads(res_path.read_text())
            ok = bool(d.get("success")) or bool(d.get("passed_optimization")) or bool(d.get("phase") == "flash" and d.get("success") is not False and sel)
            # Prefer explicit success; also accept selected kernel present after flash.
            if d.get("success") is True:
                ok = True
            elif sel and d.get("success") is not False and (d.get("final_report") or d.get("steps")):
                ok = True
        except Exception:
            ok = bool(sel)
    elif sel:
        ok = True
    if ok:
        ready.append(bench)
print("\n".join(sorted(set(ready))))
PY
}

export CAMPAIGN_ROOT

reap_finished() {
  local pid bench st
  for st in "${STATE_DIR}"/pids/*.pid; do
    [[ -e "${st}" ]] || continue
    bench="$(basename "${st}" .pid)"
    pid="$(cat "${st}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      rc_file="${STATE_DIR}/logs/${bench}.rc"
      rc=1
      [[ -f "${rc_file}" ]] && rc="$(cat "${rc_file}")"
      "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${STATE_JSON}")
d = json.loads(p.read_text())
d.setdefault("started", {}).pop("${bench}", None)
if int("${rc}") == 0:
    d.setdefault("done", {})["${bench}"] = True
    d.setdefault("failed", {}).pop("${bench}", None)
else:
    d.setdefault("failed", {})["${bench}"] = int("${rc}")
p.write_text(json.dumps(d, indent=2) + "\n")
print("reaped ${bench} rc=${rc}")
PY
      rm -f "${st}"
    fi
  done
}

start_dataflow_bench() {
  local bench="$1"
  mkdir -p "${STATE_DIR}/pids"
  local log="${STATE_DIR}/logs/${bench}.log"
  local rc_file="${STATE_DIR}/logs/${bench}.rc"
  (
    set +e
    "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_dataflow.py" --pc2 \
      --matrix-root "${CAMPAIGN_ROOT}" \
      --benches "${bench}" \
      --force \
      --results-suffix "${C2HLS_POST_FLASH_RESULTS_SUFFIX}" \
      --prompt-policy system_skills \
      --contract-turns "${C2HLS_DATAFLOW_CONTRACT_ROUNDS}" \
      "${RAG_DATAFLOW_ARGS[@]}" \
      >"${log}" 2>&1
    echo $? > "${rc_file}"
  ) &
  local pid=$!
  echo "${pid}" > "${STATE_DIR}/pids/${bench}.pid"
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${STATE_JSON}")
d = json.loads(p.read_text())
d.setdefault("started", {})["${bench}"] = ${pid}
p.write_text(json.dumps(d, indent=2) + "\n")
PY
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] started dataflow ${bench} pid=${pid}"
}

campaign_done() {
  local status
  status="$("${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
if not p.is_file():
    print("missing")
else:
    print(json.loads(p.read_text()).get("campaign_status", "unknown"))
PY
)"
  case "${status}" in
    complete|completed|failed|aborted) return 0 ;;
    *) return 1 ;;
  esac
}

# Main loop
idle_rounds=0
while true; do
  rebuild_matrix >/dev/null || true
  # Refresh flash_selected incrementally (best-effort)
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/export_flash_selected_bundle.py" --pc2 \
    --matrix-root "${CAMPAIGN_ROOT}" \
    --out-root "${C2HLS_ROOT}/artifacts/pc2/flash_selected_bundle" \
    >/dev/null 2>&1 || true

  reap_finished

  mapfile -t READY < <(flash_ready_benches)
  running="$("${PY}" -c "import json;print(len(json.load(open('${STATE_JSON}')).get('started',{})))")"
  for bench in "${READY[@]:-}"; do
    [[ -n "${bench}" ]] || continue
    already="$("${PY}" - <<PY
import json
from pathlib import Path
d=json.loads(Path("${STATE_JSON}").read_text())
b="${bench}"
print("yes" if b in d.get("started",{}) or b in d.get("done",{}) or b in d.get("failed",{}) else "no")
PY
)"
    if [[ "${already}" == "yes" ]]; then
      continue
    fi
    if (( running >= MAX_PARALLEL )); then
      break
    fi
    start_dataflow_bench "${bench}"
    running=$((running + 1))
    idle_rounds=0
  done

  if campaign_done; then
    reap_finished
    running="$("${PY}" -c "import json;print(len(json.load(open('${STATE_JSON}')).get('started',{})))")"
    if (( running == 0 )); then
      idle_rounds=$((idle_rounds + 1))
      # After campaign complete and no running dataflow, finish once we've
      # attempted all currently-ready benches (or idled a few polls).
      if (( idle_rounds >= 3 )); then
        break
      fi
    else
      idle_rounds=0
    fi
  fi
  sleep "${POLL_SEC}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting final dataflow bundle"
mkdir -p "${DATAFLOW_BUNDLE}"
"${PY}" "${C2HLS_ROOT}/scripts/pc2/export_post_flash_dataflow_csynth_bundle.py" \
  --matrix-root "${CAMPAIGN_ROOT}" \
  --flash-bundle-root "${FLASH_BUNDLE}" \
  --kernel-bundle "${DATAFLOW_BUNDLE}" \
  --force \
  || true

ln -sfn "${FLASH_BUNDLE}" "${CAMPAIGN_ROOT}/flash_selected"
ln -sfn "${DATAFLOW_BUNDLE}" "${CAMPAIGN_ROOT}/dataflow_selected"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] streaming watcher done"
echo "flash_selected=${FLASH_BUNDLE}"
echo "dataflow_selected=${DATAFLOW_BUNDLE}"
echo "state=${STATE_JSON}"
