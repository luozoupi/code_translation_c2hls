#!/usr/bin/env bash
# Poll multistep cosim run until complete, then export JSONL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
MULTISTEP_STAMP=""
COSIM_STAMP=""
JSONL_OUT=""
BASELINE_JSONL="${C2HLS_ROOT}/misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"
COSIM_WALLTIME="${PC2_COSIM_PIPELINE_WALLTIME:-13:00:00}"
POLL_SEC="${PC2_PIPELINE_POLL_SEC:-60}"
LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_cosim_export_jsonl.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multistep-stamp) MULTISTEP_STAMP="$2"; shift 2 ;;
    --cosim-stamp) COSIM_STAMP="$2"; shift 2 ;;
    --output) JSONL_OUT="$2"; shift 2 ;;
    --baseline-jsonl) BASELINE_JSONL="$2"; shift 2 ;;
    --cosim-walltime) COSIM_WALLTIME="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${MULTISTEP_STAMP}" || -z "${COSIM_STAMP}" || -z "${JSONL_OUT}" ]]; then
  echo "ERROR: --multistep-stamp, --cosim-stamp, --output required" >&2
  exit 2
fi

COSIM_ROOT="${C2HLS_ROOT}/artifacts/pc2/multistep_cosim/${COSIM_STAMP}"
mkdir -p "$(dirname "${LOG}")" "$(dirname "${JSONL_OUT}")"

plog() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

plog "waiting for multistep cosim root=${COSIM_ROOT} walltime=${COSIM_WALLTIME}"
"${PY}" - "${COSIM_ROOT}" "${COSIM_WALLTIME}" "${POLL_SEC}" <<'PY' | tee -a "${LOG}"
import json, subprocess, sys, time
from pathlib import Path

run_root = Path(sys.argv[1])
walltime = sys.argv[2]
poll_sec = int(sys.argv[3])

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
    proc = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True, check=False)
    return bool(proc.stdout.strip())

def load_job_ids(run_root: Path) -> list[str]:
    log = run_root / "submissions" / "individual_jobs.log"
    if not log.is_file():
        return []
    return [line.split()[0] for line in log.read_text().splitlines() if line.split()]

def count_results(run_root: Path) -> tuple[int, int]:
    manifest = run_root / "manifest.json"
    if not manifest.is_file():
        return 0, 0
    cells = json.loads(manifest.read_text()).get("cells", [])
    total = len(cells)
    done = sum(
        1 for cell in cells
        if (run_root / "cells" / cell.get("cell_id", "") / "cosim_result.json").is_file()
    )
    return done, total

while time.time() < deadline:
    done, total = count_results(run_root)
    jobs = load_job_ids(run_root)
    active = sum(1 for j in jobs if job_active(j))
    print(f"cosim {run_root.name}: results {done}/{total} active_jobs={active}", flush=True)
    if total > 0 and done >= total and active == 0:
        print("multistep cosim complete", flush=True)
        break
    time.sleep(poll_sec)
else:
    raise SystemExit("TIMEOUT waiting for multistep cosim runs")
PY

plog "export JSONL multistep_stamp=${MULTISTEP_STAMP} -> ${JSONL_OUT}"
"${PY}" "${C2HLS_ROOT}/misc/export_pc2_fixed_cosim_multistep_jsonl.py" \
  --baseline-jsonl "${BASELINE_JSONL}" \
  --multistep-stamp "${MULTISTEP_STAMP}" \
  --cosim-root "${COSIM_ROOT}" \
  --output "${JSONL_OUT}" | tee -a "${LOG}"

plog "done summary=${JSONL_OUT%.jsonl}.summary.json"
