#!/usr/bin/env bash
# End-to-end fixed-corpus flash campaign:
#   1) optional benchmarks_cosim gold refresh
#   2) 5-variant LLM flash (csynth+csim, record-flow artifacts)
#   3) full-size cosim: selected + phase_b (separate Slurm jobs)
#   4) export / validate combined JSONL
#
# Usage (login node, repo root):
#   ./scripts/pc2/run_fixed_cosim_flash_full_pipeline.sh
#   ./scripts/pc2/run_fixed_cosim_flash_full_pipeline.sh --flash-stamp 20260627_fixed_cosim_flash
#   ./scripts/pc2/run_fixed_cosim_flash_full_pipeline.sh --dry-run
#   ./scripts/pc2/run_fixed_cosim_flash_full_pipeline.sh --skip-flash --flash-stamp 20260626_fixed_cosim_flash
#
# Timeouts (defaults match user spec):
#   Flash GPU+compute Slurm walltime .............. 12:00:00
#   Flash Vitis/LLM per-step watchdog ............. 12h (43200s)
#   Cosim Slurm walltime .......................... 13:00:00
#   Cosim Vitis watchdog (C2HLS_COSIM_TIMEOUT) .... 12h (43200s)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
PIPELINE_LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/fixed_cosim_flash_full.log"

FLASH_WALLTIME="${PC2_FIXED_COSIM_FLASH_PIPELINE_WALLTIME:-12:00:00}"
FLASH_VITIS_TIMEOUT_SEC="${C2HLS_FLASH_PIPELINE_VITIS_TIMEOUT_SEC:-43200}"
COSIM_SLURM_WALLTIME="${PC2_COSIM_PIPELINE_WALLTIME:-13:00:00}"
COSIM_TIMEOUT_SEC="${C2HLS_COSIM_PIPELINE_TIMEOUT_SEC:-43200}"
POLL_SEC="${PC2_PIPELINE_POLL_SEC:-60}"

FLASH_STAMP="${C2HLS_FLASH_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_flash}"
DRY_RUN=0
SKIP_PREPARE=0
SKIP_FLASH=0
SKIP_COSIM=0
SKIP_JSONL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flash-stamp) shift; FLASH_STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --skip-flash) SKIP_FLASH=1; shift ;;
    --skip-cosim) SKIP_COSIM=1; shift ;;
    --skip-jsonl) SKIP_JSONL=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

DATE_PREFIX="$(printf '%s' "${FLASH_STAMP}" | grep -oE '[0-9]{8}' | head -1)"
if [[ -z "${DATE_PREFIX}" ]]; then
  echo "ERROR: --flash-stamp must contain YYYYMMDD (got: ${FLASH_STAMP})" >&2
  exit 2
fi

SELECTED_COSIM_STAMP="fixed_cosim_flash_${DATE_PREFIX}"
PHASE_B_COSIM_STAMP="fixed_cosim_flash_phase_b_${DATE_PREFIX}"
ARTIFACT_GLOB="flash_fixed_cosim_*_${FLASH_STAMP}"
SELECTED_COSIM_ROOT="${C2HLS_ROOT}/artifacts/pc2/flash_cosim/${SELECTED_COSIM_STAMP}"
PHASE_B_COSIM_ROOT="${C2HLS_ROOT}/artifacts/pc2/flash_cosim/${PHASE_B_COSIM_STAMP}"
JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_flash_u280_${DATE_PREFIX}.jsonl"
BASELINE_JSONL="${C2HLS_ROOT}/misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"

VARIANT_SESSIONS=(
  flash_fixed_cosim_nav_o
  flash_fixed_cosim_aav_n
  flash_fixed_cosim_nav_n
  flash_fixed_cosim_noskills
  flash_fixed_cosim_aav_o
)

mkdir -p "$(dirname "${PIPELINE_LOG}")"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

plog() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

plog "=== fixed cosim flash full pipeline ==="
plog "flash_stamp=${FLASH_STAMP}"
plog "selected_cosim_stamp=${SELECTED_COSIM_STAMP}"
plog "phase_b_cosim_stamp=${PHASE_B_COSIM_STAMP}"
plog "flash_walltime=${FLASH_WALLTIME} flash_vitis_timeout=${FLASH_VITIS_TIMEOUT_SEC}s"
plog "cosim_walltime=${COSIM_SLURM_WALLTIME} cosim_timeout=${COSIM_TIMEOUT_SEC}s"
plog "jsonl_out=${JSONL_OUT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  plog "dry-run: would run prepare=${SKIP_PREPARE} flash=${SKIP_FLASH} cosim=${SKIP_COSIM} jsonl=${SKIP_JSONL}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 0 — refresh benchmarks_cosim gold (= hls_baseline_cosim)
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREPARE}" -eq 0 ]]; then
  plog "phase 0: refresh benchmarks_cosim (gold_hls_source from hls_baseline_cosim)"
  "${PY}" "${C2HLS_ROOT}/scripts/prepare_hlsfactory_cosim_benchmarks.py"
else
  plog "phase 0: skipped (--skip-prepare)"
fi

# ---------------------------------------------------------------------------
# Phase 1 — 5-variant LLM flash (auto-stop on compute success)
# ---------------------------------------------------------------------------
if [[ "${SKIP_FLASH}" -eq 0 ]]; then
  plog "phase 1: submit 5-variant flash matrix"
  export PC2_FIXED_COSIM_FLASH_WALLTIME="${FLASH_WALLTIME}"
  export PC2_FORCE_WALLTIME="${FLASH_WALLTIME}"
  export C2HLS_SYNTH_TIMEOUT="${FLASH_VITIS_TIMEOUT_SEC}"
  export C2HLS_CSIM_TIMEOUT="${FLASH_VITIS_TIMEOUT_SEC}"
  export C2HLS_LLM_TIMEOUT="${FLASH_VITIS_TIMEOUT_SEC}"
  export C2HLS_FLASH_FIXED_COSIM_STAMP="${FLASH_STAMP}"
  "${SCRIPT_DIR}/start_fixed_cosim_flash_matrix.sh" \
    --stamp "${FLASH_STAMP}" \
    --auto-stop-on-complete

  plog "phase 1: waiting for all 5 flash sessions (max ${FLASH_WALLTIME} walltime each)"
  "${PY}" - "${FLASH_STAMP}" "${FLASH_WALLTIME}" "${POLL_SEC}" "${C2HLS_ROOT}" \
    "${VARIANT_SESSIONS[@]}" <<'PY'
import json, os, re, subprocess, sys, time
from pathlib import Path

flash_stamp = sys.argv[1]
walltime = sys.argv[2]
poll_sec = int(sys.argv[3])
repo = Path(sys.argv[4])
sessions = sys.argv[5:]

def walltime_sec(spec: str) -> int:
    # HH:MM:SS or D-HH:MM:SS
    if "-" in spec:
        days, rest = spec.split("-", 1)
        h, m, s = rest.split(":")
        return int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    h, m, s = spec.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

max_wait = walltime_sec(walltime) + 3600  # walltime + 1h queue buffer
deadline = time.time() + max_wait
variant_dirs = {
    "flash_fixed_cosim_nav_o": repo / f"artifacts/pc2/flash_fixed_cosim_nav_o_{flash_stamp}",
    "flash_fixed_cosim_aav_n": repo / f"artifacts/pc2/flash_fixed_cosim_aav_n_{flash_stamp}",
    "flash_fixed_cosim_nav_n": repo / f"artifacts/pc2/flash_fixed_cosim_nav_n_{flash_stamp}",
    "flash_fixed_cosim_noskills": repo / f"artifacts/pc2/flash_fixed_cosim_noskills_{flash_stamp}",
    "flash_fixed_cosim_aav_o": repo / f"artifacts/pc2/flash_fixed_cosim_aav_o_{flash_stamp}",
}

def session_state(session_id: str) -> str:
    path = repo / "artifacts/pc2/sessions" / session_id / "session.json"
    if not path.is_file():
        return "missing"
    data = json.loads(path.read_text())
    return str(data.get("compute_state") or "unknown")

def matrix_ready(artifact_dir: Path) -> bool:
    matrix = artifact_dir / "matrix.json"
    if not matrix.is_file():
        return False
    rows = json.loads(matrix.read_text())
    return bool(rows)

pending = set(sessions)
while pending and time.time() < deadline:
    done = []
    for sid in sorted(pending):
        state = session_state(sid)
        art = variant_dirs.get(sid)
        art_ok = art is not None and matrix_ready(art)
        if state == "completed" and art_ok:
            done.append(sid)
            print(f"flash session ready: {sid} compute_state=completed matrix.json ok")
        else:
            print(f"flash session pending: {sid} compute_state={state} matrix={'ok' if art_ok else 'wait'}")
    for sid in done:
        pending.discard(sid)
    if pending:
        time.sleep(poll_sec)

if pending:
    raise SystemExit(
        f"TIMEOUT waiting for flash sessions: {', '.join(sorted(pending))} "
        f"(>{max_wait}s)"
    )
print("all flash sessions complete")
PY
else
  plog "phase 1: skipped (--skip-flash)"
fi

# ---------------------------------------------------------------------------
# Phase 2 — full-size cosim (selected + phase_b)
# ---------------------------------------------------------------------------
if [[ "${SKIP_COSIM}" -eq 0 ]]; then
  export PC2_COSIM_WALLTIME="${COSIM_SLURM_WALLTIME}"
  export C2HLS_COSIM_TIMEOUT="${COSIM_TIMEOUT_SEC}"
  export C2HLS_FLASH_COSIM_FULL_SIZE=1

  plog "phase 2a: submit selected cosim stamp=${SELECTED_COSIM_STAMP}"
  C2HLS_FLASH_COSIM_STAMP="${SELECTED_COSIM_STAMP}" \
    "${SCRIPT_DIR}/submit_flash_cosim_all.sh" \
    --stamp "${SELECTED_COSIM_STAMP}" \
    --artifact-glob "${ARTIFACT_GLOB}" \
    --full-size \
    --individual

  plog "phase 2b: submit phase_b cosim stamp=${PHASE_B_COSIM_STAMP}"
  C2HLS_FLASH_COSIM_STAMP="${PHASE_B_COSIM_STAMP}" \
    C2HLS_FLASH_COSIM_KERNEL=phase_b \
    "${SCRIPT_DIR}/submit_flash_cosim_all.sh" \
    --stamp "${PHASE_B_COSIM_STAMP}" \
    --artifact-glob "${ARTIFACT_GLOB}" \
    --kernel-source phase_b \
    --full-size \
    --individual

  plog "phase 2: waiting for cosim runs to finish"
  "${PY}" - "${SELECTED_COSIM_ROOT}" "${PHASE_B_COSIM_ROOT}" "${COSIM_SLURM_WALLTIME}" "${POLL_SEC}" <<'PY'
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

# Allow parallel cosim batches: max walltime + 1h buffer per root, take overall max.
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
        print(f"cosim {run_root.name}: results {done}/{total} active_jobs={active}")
        if total == 0 or done < total or active > 0:
            all_ok = False
    if all_ok:
        print("all cosim runs complete")
        break
    time.sleep(poll_sec)
else:
    raise SystemExit("TIMEOUT waiting for cosim runs")
PY
else
  plog "phase 2: skipped (--skip-cosim)"
fi

# ---------------------------------------------------------------------------
# Phase 3 — export JSONL
# ---------------------------------------------------------------------------
if [[ "${SKIP_JSONL}" -eq 0 ]]; then
  plog "phase 3: export JSONL -> ${JSONL_OUT}"
  if [[ ! -f "${BASELINE_JSONL}" ]]; then
    echo "ERROR: missing baseline JSONL: ${BASELINE_JSONL}" >&2
    echo "Run baseline cosim JSONL export first (build_hlsfactory_fixed_cosim_jsonl.py)." >&2
    exit 2
  fi
  "${PY}" "${C2HLS_ROOT}/misc/export_pc2_fixed_cosim_flash_jsonl.py" \
    --baseline-jsonl "${BASELINE_JSONL}" \
    --flash-stamp "${FLASH_STAMP}" \
    --selected-cosim-root "${SELECTED_COSIM_ROOT}" \
    --phase-b-cosim-root "${PHASE_B_COSIM_ROOT}" \
    --output "${JSONL_OUT}"
  plog "phase 3: done summary=${JSONL_OUT%.jsonl}.summary.json"
else
  plog "phase 3: skipped (--skip-jsonl)"
fi

plog "=== pipeline complete ==="
