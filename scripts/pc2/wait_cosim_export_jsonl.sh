#!/usr/bin/env bash
# Poll cosim run roots until all cells finish, then export fixed-cosim flash JSONL.
#
#   ./scripts/pc2/wait_cosim_export_jsonl.sh \
#     --flash-stamp 20260628_fixed_cosim_flash_r2_pipelined \
#     --selected-cosim-stamp fixed_cosim_flash_20260628 \
#     --phase-b-cosim-stamp fixed_cosim_flash_phase_b_20260628 \
#     --output misc/hlsfactory_fixed_cosim_flash_u280_20260628.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
FLASH_STAMP=""
SELECTED_STAMP=""
PHASE_B_STAMP=""
JSONL_OUT=""
BASELINE_JSONL="${C2HLS_ROOT}/misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"
COSIM_WALLTIME="${PC2_COSIM_PIPELINE_WALLTIME:-13:00:00}"
POLL_SEC="${PC2_PIPELINE_POLL_SEC:-60}"
LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_cosim_export_jsonl.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flash-stamp) FLASH_STAMP="$2"; shift 2 ;;
    --selected-cosim-stamp) SELECTED_STAMP="$2"; shift 2 ;;
    --phase-b-cosim-stamp) PHASE_B_STAMP="$2"; shift 2 ;;
    --output) JSONL_OUT="$2"; shift 2 ;;
    --baseline-jsonl) BASELINE_JSONL="$2"; shift 2 ;;
    --cosim-walltime) COSIM_WALLTIME="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${FLASH_STAMP}" || -z "${SELECTED_STAMP}" || -z "${PHASE_B_STAMP}" || -z "${JSONL_OUT}" ]]; then
  echo "ERROR: --flash-stamp, --selected-cosim-stamp, --phase-b-cosim-stamp, --output required" >&2
  exit 2
fi

SELECTED_ROOT="${C2HLS_ROOT}/artifacts/pc2/flash_cosim/${SELECTED_STAMP}"
PHASE_B_ROOT="${C2HLS_ROOT}/artifacts/pc2/flash_cosim/${PHASE_B_STAMP}"
mkdir -p "$(dirname "${LOG}")" "$(dirname "${JSONL_OUT}")"

plog() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

plog "waiting for cosim selected=${SELECTED_ROOT} phase_b=${PHASE_B_ROOT} walltime=${COSIM_WALLTIME}"
"${PY}" - "${SELECTED_ROOT}" "${PHASE_B_ROOT}" "${COSIM_WALLTIME}" "${POLL_SEC}" <<'PY' | tee -a "${LOG}"
import json, subprocess, sys, time
from pathlib import Path

run_roots = [Path(sys.argv[1]), Path(sys.argv[2])]
walltime = sys.argv[3]
poll_sec = int(sys.argv[4])

def walltime_sec(spec: str) -> int:
    if "-" in spec:
        days, rest = spec.split("-", 1)
        h, m, s = rest.split(":")
        return int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    h, m, s = spec.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

max_wait = walltime_sec(walltime) + 3600
deadline = time.time() + max_wait

def job_active(job_id: str) -> bool:
    if not job_id or job_id in {"null", "None"}:
        return False
    proc = subprocess.run(
        ["squeue", "-h", "-j", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())

def load_job_ids(run_root: Path) -> list[str]:
    log = run_root / "submissions" / "individual_jobs.log"
    if not log.is_file():
        return []
    ids = []
    for line in log.read_text().splitlines():
        parts = line.split()
        if parts:
            ids.append(parts[0])
    return ids

def count_results(run_root: Path) -> tuple[int, int]:
    manifest = run_root / "manifest.json"
    if not manifest.is_file():
        return 0, 0
    cells = json.loads(manifest.read_text()).get("cells", [])
    total = len(cells)
    done = 0
    cells_dir = run_root / "cells"
    for cell in cells:
        cid = cell.get("cell_id")
        if cid and (cells_dir / cid / "cosim_result.json").is_file():
            done += 1
    return done, total

while time.time() < deadline:
    all_ok = True
    for run_root in run_roots:
        done, total = count_results(run_root)
        jobs = load_job_ids(run_root)
        active = sum(1 for j in jobs if job_active(j))
        print(f"cosim {run_root.name}: results {done}/{total} active_jobs={active}", flush=True)
        if total == 0 or done < total or active > 0:
            all_ok = False
    if all_ok:
        print("all cosim runs complete", flush=True)
        break
    time.sleep(poll_sec)
else:
    raise SystemExit("TIMEOUT waiting for cosim runs")
PY

if [[ ! -f "${BASELINE_JSONL}" ]]; then
  plog "ERROR: missing baseline JSONL: ${BASELINE_JSONL}"
  exit 2
fi

plog "export JSONL flash_stamp=${FLASH_STAMP} -> ${JSONL_OUT}"
"${PY}" "${C2HLS_ROOT}/misc/export_pc2_fixed_cosim_flash_jsonl.py" \
  --baseline-jsonl "${BASELINE_JSONL}" \
  --flash-stamp "${FLASH_STAMP}" \
  --selected-cosim-root "${SELECTED_ROOT}" \
  --phase-b-cosim-root "${PHASE_B_ROOT}" \
  --output "${JSONL_OUT}" | tee -a "${LOG}"

plog "done summary=${JSONL_OUT%.jsonl}.summary.json"
