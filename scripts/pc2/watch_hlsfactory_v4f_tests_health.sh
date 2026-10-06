#!/usr/bin/env bash
# Health check for HLSFactory v4f test sequence. Prints STATUS=ok|warn|fail.
# Usage: ./scripts/pc2/watch_hlsfactory_v4f_tests_health.sh [SEQ_ROOT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

SEQ_ROOT="${1:-}"
if [[ -z "${SEQ_ROOT}" ]]; then
  SEQ_ROOT="$(ls -1dt "${C2HLS_ROOT}"/artifacts/pc2/hlsfactory_ds_v4f_tests_* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${SEQ_ROOT}" || ! -d "${SEQ_ROOT}" ]]; then
  echo "STATUS=fail reason=no_seq_root"
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3
MANIFEST="${SEQ_ROOT}/sequence_manifest.json"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_DIR="${SEQ_ROOT}/watch"
mkdir -p "${LOG_DIR}"
OUT="${LOG_DIR}/health_latest.txt"
JSON_OUT="${LOG_DIR}/health_latest.json"

{
  echo "=== health ${STAMP} ==="
  echo "seq_root=${SEQ_ROOT}"
} > "${OUT}"

STATUS="ok"
REASONS=()

if [[ ! -f "${MANIFEST}" ]]; then
  STATUS="fail"
  REASONS+=("missing_manifest")
fi

# Proxy checks
for FLAVOR in skills noskills bare; do
  EP_FILE="${SEQ_ROOT}/proxy_${FLAVOR}/llm_endpoint.json"
  if [[ ! -f "${EP_FILE}" ]]; then
    STATUS="fail"
    REASONS+=("proxy_${FLAVOR}_missing")
    echo "proxy_${FLAVOR}=MISSING" >> "${OUT}"
    continue
  fi
  URL="$("${PY}" -c "import json;print(json.load(open('${EP_FILE}'))['url'])" 2>/dev/null || true)"
  if [[ -z "${URL}" ]]; then
    STATUS="fail"
    REASONS+=("proxy_${FLAVOR}_bad_url")
    echo "proxy_${FLAVOR}=BAD_URL" >> "${OUT}"
    continue
  fi
  if curl -sf --max-time 8 "${URL}/models" >/dev/null 2>&1; then
    echo "proxy_${FLAVOR}=UP ${URL}" >> "${OUT}"
  else
    STATUS="fail"
    REASONS+=("proxy_${FLAVOR}_down")
    echo "proxy_${FLAVOR}=DOWN ${URL}" >> "${OUT}"
  fi
done

# Campaign / selection progress
"${PY}" - <<PY >> "${OUT}"
import json
from pathlib import Path
from datetime import datetime, timezone

seq = Path("${SEQ_ROOT}")
manifest = seq / "sequence_manifest.json"
doc = json.loads(manifest.read_text()) if manifest.is_file() else {}
tests = doc.get("tests") or {}
summary = {"stamp": "${STAMP}", "seq_root": str(seq), "tests": {}}
problems = []
for tname, tmeta in sorted(tests.items()):
    flavors = (tmeta or {}).get("flavors") or {}
    tsum = {"flavors": {}}
    for fl, meta in sorted(flavors.items()):
        root = Path(str((meta or {}).get("campaign_root") or ""))
        entry = {"campaign_root": str(root), "exists": root.is_dir()}
        if not root.is_dir():
            problems.append(f"{tname}/{fl}:missing_campaign")
            tsum["flavors"][fl] = entry
            continue
        camp = {}
        cj = root / "campaign.json"
        if cj.is_file():
            try:
                camp = json.loads(cj.read_text())
            except Exception as e:
                problems.append(f"{tname}/{fl}:bad_campaign_json:{e}")
        status = camp.get("campaign_status", "missing")
        entry["campaign_status"] = status
        entry["selection_complete"] = bool(camp.get("selection_complete"))
        entry["post_watcher_job_id"] = camp.get("post_watcher_job_id")
        entry["test_mode"] = camp.get("test_mode")
        # count flash selected / df rankings
        ranked_flash = 0
        ranked_df = 0
        selected = 0
        variants = root / "variants"
        if variants.is_dir():
            for cell in variants.glob("*/*/*"):
                if not cell.is_dir():
                    continue
                bench = cell.parent.name
                if any(cell.glob(f"{bench}_selected.cpp")) or any(cell.glob(f"{bench}_final.cpp")):
                    selected += 1
                if (cell / f"{bench}_flash_candidate_ranking.json").is_file():
                    ranked_flash += 1
                if (cell / f"{bench}_dataflow_candidate_ranking.json").is_file():
                    ranked_df += 1
        entry["selected_cells"] = selected
        entry["flash_ranked"] = ranked_flash
        entry["dataflow_ranked"] = ranked_df
        # watch log tail hint
        wlog = root / "flow" / "post_flash_dataflow_watcher.log"
        entry["post_log_exists"] = wlog.is_file()
        if wlog.is_file():
            try:
                tail = wlog.read_text(errors="replace").splitlines()[-3:]
                entry["post_log_tail"] = tail
            except Exception:
                pass
        if status in {"failed", "aborted"}:
            problems.append(f"{tname}/{fl}:campaign_{status}")
        if status in {"complete", "completed"} and selected == 0 and not entry["selection_complete"]:
            problems.append(f"{tname}/{fl}:complete_with_zero_selected")
        # Detect the RAG_MODE misconfig that wiped the first launch.
        ev = root / "flow" / "events.jsonl"
        if ev.is_file():
            try:
                text = ev.read_text(errors="replace")
                rag_hits = text.count("RAG_MODE set but")
                entry["rag_mode_fail_events"] = rag_hits
                if rag_hits >= 3:
                    problems.append(f"{tname}/{fl}:rag_mode_codegen_failures={rag_hits}")
            except Exception:
                pass
        tsum["flavors"][fl] = entry
        print(f"test={tname} flavor={fl} status={status} sel={selected} flash_rank={ranked_flash} df_rank={ranked_df} selection_complete={entry['selection_complete']}")
    summary["tests"][tname] = tsum

# queue snapshot
import subprocess
sq = subprocess.run(["squeue", "-u", "${USER}", "-h", "-o", "%i|%j|%T|%M|%R"], capture_output=True, text=True)
jobs = [ln for ln in sq.stdout.splitlines() if ln.strip()]
# Only count HLSFactory jobs for hollow-queue detection (ignore unrelated GRPO etc.)
hf_jobs = [ln for ln in jobs if "bphf" in ln]
print(f"squeue_jobs={len(jobs)}")
for ln in jobs[:40]:
    print("  " + ln)
summary["squeue_count"] = len(jobs)
summary["hlsfactory_squeue_count"] = len(hf_jobs)
# If all active campaigns still need selection but no bphf jobs remain → fail
needs_work = False
for tname, tmeta in (summary.get("tests") or {}).items():
    for fl, e in (tmeta.get("flavors") or {}).items():
        st = e.get("campaign_status")
        if st in {"running", "complete", "completed"} and not e.get("selection_complete") and int(e.get("selected_cells") or 0) == 0:
            needs_work = True
        elif st == "running" and not e.get("selection_complete"):
            needs_work = True
if needs_work and len(hf_jobs) == 0:
    problems.append("no_hlsfactory_jobs_while_selection_incomplete")
summary["problems"] = problems
Path("${JSON_OUT}").write_text(json.dumps(summary, indent=2) + "\n")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  " + p)
# expose to bash via marker file
Path("${LOG_DIR}/problems.txt").write_text("\n".join(problems) + ("\n" if problems else ""))
PY

# Slurm: if zero jobs AND no selection_complete on active test → warn/fail
SQ_N="$(squeue -u "$USER" -h | wc -l | tr -d ' ')"
echo "squeue_count=${SQ_N}" >> "${OUT}"

PROBLEMS_FILE="${LOG_DIR}/problems.txt"
if [[ -s "${PROBLEMS_FILE}" ]]; then
  STATUS="fail"
  while IFS= read -r line; do
    [[ -n "${line}" ]] && REASONS+=("${line}")
  done < "${PROBLEMS_FILE}"
fi

# Dead proxies already marked fail. Also: if sequence launch log shows ERROR recently.
LAUNCH_LOG="${SEQ_ROOT}/launch.log"
if [[ -f "${LAUNCH_LOG}" ]]; then
  if rg -q "ERROR:|Traceback|FATAL" "${LAUNCH_LOG}" 2>/dev/null; then
    # only fail if last 30 lines contain error AND not clearly past
    if tail -n 40 "${LAUNCH_LOG}" | rg -q "ERROR:|Traceback|FATAL"; then
      if [[ "${STATUS}" == "ok" ]]; then STATUS="warn"; fi
      REASONS+=("launch_log_errors")
    fi
  fi
  echo "--- launch.log tail ---" >> "${OUT}"
  tail -n 15 "${LAUNCH_LOG}" >> "${OUT}" || true
fi

# Orchestrator still running?
ORCH_PID_FILE="${SEQ_ROOT}/orchestrator.pid"
if [[ -f "${ORCH_PID_FILE}" ]]; then
  OPID="$(cat "${ORCH_PID_FILE}")"
  if kill -0 "${OPID}" 2>/dev/null; then
    echo "orchestrator_pid=${OPID} alive" >> "${OUT}"
  else
    echo "orchestrator_pid=${OPID} DEAD" >> "${OUT}"
    # If selection not complete for all tests, this is a problem
    if [[ "${STATUS}" != "fail" ]]; then STATUS="warn"; fi
    REASONS+=("orchestrator_dead")
  fi
fi

echo "STATUS=${STATUS}" >> "${OUT}"
echo "REASONS=${REASONS[*]:-none}" >> "${OUT}"
echo "STATUS=${STATUS} reasons=${REASONS[*]:-none}"
cat "${OUT}"
# exit 0 even on fail so watcher can decide; STATUS line is authoritative
exit 0
