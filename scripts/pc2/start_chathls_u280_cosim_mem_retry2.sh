#!/usr/bin/env bash
# ChatHLS-only U280 full-size cosim RETRY #2:
#   - gemm_ncubed: xelab SIGSEGV → C2HLS_COSIM_XELAB_MT_OFF=1 + more RAM
#   - kernel_symm: Slurm OOM @ 256G → largemem 1024G
#   - kernel_syrk: re-run with same mitigations (prior clean COSIM FAIL may persist)
#
# Leaves gemm (still running on job 2030824) alone.
#
# Usage:
#   ./scripts/pc2/start_chathls_u280_cosim_mem_retry2.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "usage: $0 [--dry-run]"
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-7-00:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-604800}"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export C2HLS_FLASH_COSIM_KERNEL=selected
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready}"
export PC2_COSIM_ARRAY_MAX_PARALLEL="${PC2_COSIM_ARRAY_MAX_PARALLEL:-3}"
# largemem nodes have ~1.5T; symm already hit 256G OOM.
# Force largemem by default (login shells often export PC2_COMPUTE_PARTITION=normal).
export PC2_COSIM_MEM="${PC2_COSIM_MEM:-1024G}"
export PC2_COSIM_CPUS="${PC2_COSIM_CPUS:-16}"
if [[ -z "${PC2_FORCE_PARTITION:-}" ]]; then
  export PC2_COMPUTE_PARTITION=largemem
else
  export PC2_COMPUTE_PARTITION="${PC2_FORCE_PARTITION}"
fi

# xelab SIGSEGV mitigations (absolute-path xelab → setup + patch -mt off)
export C2HLS_COSIM_TRACE_LEVEL="${C2HLS_COSIM_TRACE_LEVEL:-none}"
export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_COSIM_EXTRA_ARGS="${C2HLS_COSIM_EXTRA_ARGS:--disable_deadlock_detection}"

RETRY_BENCHES=(gemm_ncubed kernel_symm kernel_syrk)

CHATHLS_SESSION="${CHATHLS_U280_SESSION:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260717-001649}"
CHATHLS_CSV="${CHATHLS_SESSION}/final_latency_csynth.csv"

STAMP_BASE="$(date -u +%Y%m%d_%H%M%S)"
BATCH_ROOT="${C2HLS_ROOT}/artifacts/pc2/u280_compare_cosim_${STAMP_BASE}"
mkdir -p "${BATCH_ROOT}"
INDEX_JSON="${BATCH_ROOT}/launch_index.json"
RETRY_LIST_FILE="${BATCH_ROOT}/retry_benches.txt"
printf '%s\n' "${RETRY_BENCHES[@]}" > "${RETRY_LIST_FILE}"

echo "=== ChatHLS U280 cosim MEMORY/SIGSEGV RETRY #2 ==="
echo "batch_root=${BATCH_ROOT}"
echo "retry_benches=${RETRY_BENCHES[*]}"
echo "PC2_COSIM_MEM=${PC2_COSIM_MEM} PC2_COSIM_CPUS=${PC2_COSIM_CPUS}"
echo "partition=${PC2_COMPUTE_PARTITION}"
echo "C2HLS_COSIM_XELAB_MT_OFF=${C2HLS_COSIM_XELAB_MT_OFF}"
echo "C2HLS_COSIM_EXTRA_ARGS=${C2HLS_COSIM_EXTRA_ARGS}"
echo "dry_run=${DRY_RUN}"

CHATHLS_ART="${BATCH_ROOT}/chathls_u280_hybrid_20260717_001649_memretry2"
"${C2HLS_PYTHON:-python3}" - "${CHATHLS_CSV}" "${CHATHLS_ART}" "${RETRY_LIST_FILE}" <<'PY'
import csv, json, shutil, sys
from pathlib import Path

csv_path, art, retry_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
retry = {ln.strip() for ln in retry_path.read_text().splitlines() if ln.strip()}
art.mkdir(parents=True, exist_ok=True)
rows = []
for r in csv.DictReader(csv_path.open()):
    bench = (r.get("bench") or "").strip()
    if bench not in retry:
        continue
    if str(r.get("passed_optimization", "")).strip().lower() != "true":
        print(f"skip {bench}: passed_optimization!=true", file=sys.stderr)
        continue
    run_dir = Path((r.get("run_dir") or "").strip())
    src = run_dir / "project" / f"{bench}.cpp"
    if not src.is_file():
        arts = sorted((run_dir / "artifacts").glob("optimization-round-*.cpp")) if (run_dir / "artifacts").is_dir() else []
        src = arts[-1] if arts else None
    if src is None or not Path(src).is_file():
        print(f"skip {bench}: no kernel cpp", file=sys.stderr)
        continue
    c2_bench = f"chathls_{bench}"
    cell = art / "cells" / c2_bench / "chathls_native"
    cell.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, cell / f"{c2_bench}_selected.cpp")
    shutil.copy2(src, cell / f"{c2_bench}_final.cpp")
    rows.append({
        "bench": c2_bench,
        "cell_dir": str(cell.resolve()),
        "status": "ok",
        "model": "chathls-deepseek-u280",
        "variant": "chathls_native",
        "matrix_family": "chathls_u280_hybrid_memretry2",
        "mode": "chathls",
    })
(art / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
missing = sorted(retry - {r["bench"].removeprefix("chathls_") for r in rows})
if missing:
    raise SystemExit(f"missing retry benches after build: {missing}")
print(f"chathls memretry2 artifact cells={len(rows)} root={art}")
for r in rows:
    print(f"  {r['bench']}")
PY

NAME="chathls_u280_memretry2"
STAMP="${STAMP_BASE}_${NAME}"
ART_PATH="${CHATHLS_ART}"
ART_BASE="$(basename "${ART_PATH}")"
ART_PARENT="$(dirname "${ART_PATH}")"

export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_ROOT="${BATCH_ROOT}/flash_cosim"
export C2HLS_FLASH_COSIM_RUN_ROOT="${BATCH_ROOT}/flash_cosim/${STAMP}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  C2HLS_FLASH_COSIM_STAMP="${STAMP}" \
  C2HLS_FLASH_COSIM_ROOT="${BATCH_ROOT}/flash_cosim" \
  C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT}" \
  C2HLS_FLASH_COSIM_FULL_SIZE=1 \
    "${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/build_flash_cosim_manifest.py" \
      --stamp "${STAMP}" \
      --artifact-glob "${ART_BASE}" \
      --artifact "${ART_BASE}" \
      --full-size \
      --kernel-source selected \
      --run-root "${BATCH_ROOT}/flash_cosim" \
      --dry-run
  echo "dry-run complete batch_root=${BATCH_ROOT}"
  exit 0
fi

"${C2HLS_PYTHON:-python3}" - "${ART_PARENT}" "${ART_BASE}" "${STAMP}" "${BATCH_ROOT}/flash_cosim" <<'PY'
import json, os, sys
from pathlib import Path

art_parent, art_base, stamp, cosim_root = sys.argv[1:5]
os.environ["C2HLS_FLASH_COSIM_STAMP"] = stamp
os.environ["C2HLS_FLASH_COSIM_ROOT"] = cosim_root
os.environ["C2HLS_FLASH_COSIM_FULL_SIZE"] = "1"
os.environ["C2HLS_FLASH_COSIM_KERNEL"] = "selected"
sys.path.insert(0, os.environ.get("C2HLS_ROOT", "."))
from scripts.pc2.flash_cosim_lib import discover_cells, write_manifest, cosim_run_root

cells = discover_cells(
    artifacts_root=Path(art_parent),
    artifact_glob=art_base,
    artifact_filter={art_base},
    matrix_status="ok",
    kernel_source="selected",
)
run_root = cosim_run_root(stamp)
path = write_manifest(
    run_root,
    cells,
    extra={
        "artifact_glob": art_base,
        "cosim_size_mode": "full",
        "kernel_source": "selected",
        "compare_label": art_base,
        "retry_reason": "sigsegv_mt_off_and_oom_more_ram",
        "pc2_cosim_mem": os.environ.get("PC2_COSIM_MEM", ""),
        "c2hls_cosim_xelab_mt_off": os.environ.get("C2HLS_COSIM_XELAB_MT_OFF", ""),
        "c2hls_cosim_extra_args": os.environ.get("C2HLS_COSIM_EXTRA_ARGS", ""),
    },
)
benches = []
for c in cells:
    bench = getattr(c, "bench", None)
    if bench is None and isinstance(c, dict):
        bench = c.get("bench")
    benches.append(bench)
print(json.dumps({"run_root": str(run_root), "cells": len(cells), "manifest": str(path), "benches": benches}))
PY

RUN_ROOT="${BATCH_ROOT}/flash_cosim/${STAMP}"
mkdir -p "${RUN_ROOT}/submissions" "${RUN_ROOT}/slurm"
"${SCRIPT_DIR}/verify_flash_cosim.sh" "${RUN_ROOT}"

CELL_COUNT="$(
  "${C2HLS_PYTHON:-python3}" -c "import json;print(len(json.load(open('${RUN_ROOT}/manifest.json'))['cells']))"
)"
if [[ "${CELL_COUNT}" -ne "${#RETRY_BENCHES[@]}" ]]; then
  echo "ERROR: expected ${#RETRY_BENCHES[@]} cells, got ${CELL_COUNT}" >&2
  exit 2
fi
LAST_INDEX=$((CELL_COUNT - 1))

JOB_ID="$(
  sbatch --parsable \
    --job-name="u280cosim-ch-memretry2" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --account="${PC2_SLURM_ACCOUNT}" \
    --cpus-per-task="${PC2_COSIM_CPUS}" \
    --mem="${PC2_COSIM_MEM}" \
    --time="${PC2_COSIM_WALLTIME}" \
    --array="0-${LAST_INDEX}%${PC2_COSIM_ARRAY_MAX_PARALLEL}" \
    --chdir="${C2HLS_ROOT}" \
    --output="${RUN_ROOT}/slurm/cosim-%A_%a.out" \
    --error="${RUN_ROOT}/slurm/cosim-%A_%a.err" \
    --export=ALL,C2HLS_ROOT,C2HLS_SITE,C2HLS_FLASH_COSIM_RUN_ROOT="${RUN_ROOT}",C2HLS_FLASH_COSIM_STAMP="${STAMP}",C2HLS_COSIM_TIMEOUT,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_BENCHMARKS_ROOT,C2HLS_FLASH_COSIM_KERNEL=selected,C2HLS_COSIM_TRACE_LEVEL,C2HLS_COSIM_XELAB_MT_OFF,C2HLS_COSIM_EXTRA_ARGS \
    "${SCRIPT_DIR}/cosim_array.sbatch.sh"
)"

{
  echo "name=${NAME}"
  echo "stamp=${STAMP}"
  echo "run_root=${RUN_ROOT}"
  echo "job_id=${JOB_ID}"
  echo "cell_count=${CELL_COUNT}"
  echo "array=0-${LAST_INDEX}%${PC2_COSIM_ARRAY_MAX_PARALLEL}"
  echo "pc2_cosim_mem=${PC2_COSIM_MEM}"
  echo "pc2_cosim_cpus=${PC2_COSIM_CPUS}"
  echo "pc2_partition=${PC2_COMPUTE_PARTITION}"
  echo "c2hls_cosim_xelab_mt_off=${C2HLS_COSIM_XELAB_MT_OFF}"
  echo "c2hls_cosim_extra_args=${C2HLS_COSIM_EXTRA_ARGS}"
  echo "retry_benches=${RETRY_BENCHES[*]}"
  echo "prior_memretry=artifacts/pc2/u280_compare_cosim_20260721_011927"
  echo "submitted_at=$(date -Is)"
} > "${RUN_ROOT}/submissions/array_job.txt"

"${C2HLS_PYTHON:-python3}" - "${INDEX_JSON}" "${NAME}" "${RUN_ROOT}" "${JOB_ID}" "${CELL_COUNT}" "${ART_PATH}" "${PC2_COSIM_MEM}" <<'PY'
import json, sys
from pathlib import Path
idx, name, run_root, job_id, cells, art, mem = sys.argv[1:8]
doc = {
    "batch_root": str(Path(run_root).parents[1]),
    "kind": "chathls_u280_cosim_memretry2",
    "pc2_cosim_mem": mem,
    "pc2_partition": "largemem",
    "mitigations": {
        "C2HLS_COSIM_XELAB_MT_OFF": "1",
        "C2HLS_COSIM_EXTRA_ARGS": "-disable_deadlock_detection",
        "ulimit_stack": "unlimited",
    },
    "retry_benches": ["gemm_ncubed", "kernel_symm", "kernel_syrk"],
    "left_alone_running": ["gemm"],
    "prior_memretry": "artifacts/pc2/u280_compare_cosim_20260721_011927",
    "targets": {
        name: {
            "run_root": run_root,
            "job_id": job_id,
            "cell_count": int(cells),
            "artifact": art,
        }
    },
}
Path(idx).write_text(json.dumps(doc, indent=2) + "\n")
PY

echo ""
echo "submitted ${NAME}: job_id=${JOB_ID} cells=${CELL_COUNT} mem=${PC2_COSIM_MEM} partition=${PC2_COMPUTE_PARTITION}"
echo "run_root=${RUN_ROOT}"
echo "launch_index=${INDEX_JSON}"
cat "${INDEX_JSON}"
echo ""
echo "Monitor: squeue -j ${JOB_ID}"
echo "Report:  ${C2HLS_PYTHON:-python3} scripts/pc2/report_chathls_u280_cosim_mem_retry.py --run-root ${RUN_ROOT}"
