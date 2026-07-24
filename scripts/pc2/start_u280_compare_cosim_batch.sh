#!/usr/bin/env bash
# Submit full-size Vitis cosim for U280 compare set:
#   1) ChatHLS hybrid-u280-split-20260717-001649 (optimized kernels)
#   2) c2hls DeepSeek RAG2+skills
#   3) c2hls GLM RAG2+skills
#   4) c2hls GLM skills (no RAG)
#
# Walltimes (both default 7 days):
#   PC2_COSIM_WALLTIME   Slurm --time for each array task / job
#   C2HLS_COSIM_TIMEOUT  Vitis cosim process timeout (seconds)
#
# Usage:
#   ./scripts/pc2/start_u280_compare_cosim_batch.sh [--dry-run]
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

# Both cosim job walltime (Slurm) and cosim process timeout = 7 days.
export PC2_COSIM_WALLTIME="${PC2_COSIM_WALLTIME:-7-00:00:00}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-604800}"
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export C2HLS_FLASH_COSIM_KERNEL=selected
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready}"
export PC2_COSIM_ARRAY_MAX_PARALLEL="${PC2_COSIM_ARRAY_MAX_PARALLEL:-16}"

CHATHLS_SESSION="${CHATHLS_U280_SESSION:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260717-001649}"
CHATHLS_CSV="${CHATHLS_SESSION}/final_latency_csynth.csv"

GLM_SEQ="${C2HLS_ROOT}/artifacts/pc2/glm_u280_seq_20260718_021929/sequence_state.json"
DS_RAG2="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260717_195140_rag2_skills"
GLM_RAG2="$(
  "${C2HLS_PYTHON:-python3}" -c "import json;print(json.load(open('${GLM_SEQ}'))['campaigns']['rag2_skills']['campaign_root'])"
)"
GLM_SKILLS="$(
  "${C2HLS_PYTHON:-python3}" -c "import json;print(json.load(open('${GLM_SEQ}'))['campaigns']['skills']['campaign_root'])"
)"

STAMP_BASE="$(date -u +%Y%m%d_%H%M%S)"
BATCH_ROOT="${C2HLS_ROOT}/artifacts/pc2/u280_compare_cosim_${STAMP_BASE}"
mkdir -p "${BATCH_ROOT}"
INDEX_JSON="${BATCH_ROOT}/launch_index.json"

echo "=== U280 compare cosim batch ==="
echo "batch_root=${BATCH_ROOT}"
echo "PC2_COSIM_WALLTIME=${PC2_COSIM_WALLTIME}  (Slurm job walltime)"
echo "C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT}  (cosim process timeout sec)"
echo "cosim_benches_root=${C2HLS_COSIM_BENCHMARKS_ROOT}"
echo "full_size=1 kernel=selected dry_run=${DRY_RUN}"

# --- prepare ChatHLS synthetic artifact (selected kernels from run dirs) ---
CHATHLS_ART="${BATCH_ROOT}/chathls_u280_hybrid_20260717_001649"
"${C2HLS_PYTHON:-python3}" - "${CHATHLS_CSV}" "${CHATHLS_ART}" <<'PY'
import csv, json, shutil, sys
from pathlib import Path

csv_path, art = Path(sys.argv[1]), Path(sys.argv[2])
art.mkdir(parents=True, exist_ok=True)
rows = []
for r in csv.DictReader(csv_path.open()):
    bench = (r.get("bench") or "").strip()
    if not bench:
        continue
    # Only benches that passed optimization (have a meaningful csynth design).
    if str(r.get("passed_optimization", "")).strip().lower() != "true":
        continue
    run_dir = Path((r.get("run_dir") or "").strip())
    src = run_dir / "project" / f"{bench}.cpp"
    if not src.is_file():
        # fallback: latest optimization-round
        arts = sorted((run_dir / "artifacts").glob("optimization-round-*.cpp")) if (run_dir / "artifacts").is_dir() else []
        src = arts[-1] if arts else None
    if src is None or not Path(src).is_file():
        print(f"skip {bench}: no kernel cpp", file=sys.stderr)
        continue
    c2_bench = f"chathls_{bench}"
    cell = art / "cells" / c2_bench / "chathls_native"
    cell.mkdir(parents=True, exist_ok=True)
    dst = cell / f"{c2_bench}_selected.cpp"
    shutil.copy2(src, dst)
    # also as final for resolve_cell_final_cpp fallback
    shutil.copy2(src, cell / f"{c2_bench}_final.cpp")
    rows.append({
        "bench": c2_bench,
        "cell_dir": str(cell.resolve()),
        "status": "ok",
        "model": "chathls-deepseek-u280",
        "variant": "chathls_native",
        "matrix_family": "chathls_u280_hybrid",
        "mode": "chathls",
    })
(art / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
print(f"chathls artifact cells={len(rows)} root={art}")
PY

# Fix campaign matrix statuses to ok when selected kernel exists (discover defaults to status=ok).
for CAMP in "${DS_RAG2}" "${GLM_RAG2}" "${GLM_SKILLS}"; do
  "${C2HLS_PYTHON:-python3}" - "${CAMP}" <<'PY'
import json, sys
from pathlib import Path
from flash_flow_artifacts import resolve_cell_final_cpp

camp = Path(sys.argv[1])
mat = camp / "matrix.json"
rows = json.loads(mat.read_text())
changed = 0
for row in rows:
    bench = row.get("bench", "")
    cell = Path(row.get("cell_dir") or "")
    kernel = resolve_cell_final_cpp(cell, bench) if cell.is_dir() else None
    new = "ok" if kernel is not None else "missing"
    if row.get("status") != new:
        row["status"] = new
        changed += 1
mat.write_text(json.dumps(rows, indent=2) + "\n")
ok = sum(1 for r in rows if r.get("status") == "ok")
print(f"{camp.name}: matrix ok={ok}/{len(rows)} rewritten={changed}")
PY
done

TARGETS=(
  "chathls_u280|${CHATHLS_ART}"
  "ds_rag2_skills|${DS_RAG2}"
  "glm_rag2_skills|${GLM_RAG2}"
  "glm_skills|${GLM_SKILLS}"
)

LAUNCH_DOC='{"batch_root":"'"${BATCH_ROOT}"'","pc2_cosim_walltime":"'"${PC2_COSIM_WALLTIME}"'","c2hls_cosim_timeout_s":'"${C2HLS_COSIM_TIMEOUT}"',"targets":{}}'
echo "${LAUNCH_DOC}" > "${INDEX_JSON}"

for entry in "${TARGETS[@]}"; do
  NAME="${entry%%|*}"
  ART_PATH="${entry#*|}"
  ART_BASE="$(basename "${ART_PATH}")"
  STAMP="${STAMP_BASE}_${NAME}"
  echo ""
  echo "=== target=${NAME} artifact=${ART_BASE} stamp=${STAMP} ==="

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
    continue
  fi

  export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
  export C2HLS_FLASH_COSIM_ROOT="${BATCH_ROOT}/flash_cosim"
  export C2HLS_FLASH_COSIM_RUN_ROOT="${BATCH_ROOT}/flash_cosim/${STAMP}"

  # Point discover at the specific artifact dir's parent (artifacts/pc2 or BATCH_ROOT).
  ART_PARENT="$(dirname "${ART_PATH}")"
  # build manifest by temporarily using ART_PARENT as PC2 artifacts via env override in python
  "${C2HLS_PYTHON:-python3}" - "${ART_PARENT}" "${ART_BASE}" "${STAMP}" "${BATCH_ROOT}/flash_cosim" <<'PY'
import json, os, sys
from pathlib import Path

art_parent, art_base, stamp, cosim_root = sys.argv[1:5]
os.environ["C2HLS_FLASH_COSIM_STAMP"] = stamp
os.environ["C2HLS_FLASH_COSIM_ROOT"] = cosim_root
os.environ["C2HLS_FLASH_COSIM_FULL_SIZE"] = "1"
os.environ["C2HLS_FLASH_COSIM_KERNEL"] = "selected"
# keep cosim benches root from outer env
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
    },
)
print(json.dumps({"run_root": str(run_root), "cells": len(cells), "manifest": str(path)}))
PY

  RUN_ROOT="${BATCH_ROOT}/flash_cosim/${STAMP}"
  mkdir -p "${RUN_ROOT}/submissions" "${RUN_ROOT}/slurm"
  "${SCRIPT_DIR}/verify_flash_cosim.sh" "${RUN_ROOT}"

  CELL_COUNT="$(
    "${C2HLS_PYTHON:-python3}" -c "import json;print(len(json.load(open('${RUN_ROOT}/manifest.json'))['cells']))"
  )"
  if [[ "${CELL_COUNT}" -le 0 ]]; then
    echo "ERROR: zero cosim cells for ${NAME}" >&2
    exit 2
  fi
  LAST_INDEX=$((CELL_COUNT - 1))

  JOB_ID="$(
    sbatch --parsable \
      --job-name="u280cosim-${NAME}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --account="${PC2_SLURM_ACCOUNT}" \
      --cpus-per-task="${PC2_COSIM_CPUS:-8}" \
      --mem="${PC2_COSIM_MEM:-32G}" \
      --time="${PC2_COSIM_WALLTIME}" \
      --array="0-${LAST_INDEX}%${PC2_COSIM_ARRAY_MAX_PARALLEL}" \
      --chdir="${C2HLS_ROOT}" \
      --output="${RUN_ROOT}/slurm/cosim-%A_%a.out" \
      --error="${RUN_ROOT}/slurm/cosim-%A_%a.err" \
      --export=ALL,C2HLS_ROOT,C2HLS_SITE,C2HLS_FLASH_COSIM_RUN_ROOT="${RUN_ROOT}",C2HLS_FLASH_COSIM_STAMP="${STAMP}",C2HLS_COSIM_TIMEOUT,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_BENCHMARKS_ROOT,C2HLS_FLASH_COSIM_KERNEL=selected \
      "${SCRIPT_DIR}/cosim_array.sbatch.sh"
  )"
  {
    echo "name=${NAME}"
    echo "stamp=${STAMP}"
    echo "run_root=${RUN_ROOT}"
    echo "job_id=${JOB_ID}"
    echo "cell_count=${CELL_COUNT}"
    echo "array=0-${LAST_INDEX}%${PC2_COSIM_ARRAY_MAX_PARALLEL}"
    echo "pc2_cosim_walltime=${PC2_COSIM_WALLTIME}"
    echo "c2hls_cosim_timeout_s=${C2HLS_COSIM_TIMEOUT}"
    echo "submitted_at=$(date -Is)"
  } > "${RUN_ROOT}/submissions/array_job.txt"

  "${C2HLS_PYTHON:-python3}" - "${INDEX_JSON}" "${NAME}" "${RUN_ROOT}" "${JOB_ID}" "${CELL_COUNT}" "${ART_PATH}" <<'PY'
import json, sys
from pathlib import Path
idx, name, run_root, job_id, cells, art = sys.argv[1:7]
doc = json.loads(Path(idx).read_text())
doc.setdefault("targets", {})[name] = {
    "run_root": run_root,
    "job_id": job_id,
    "cell_count": int(cells),
    "artifact": art,
}
Path(idx).write_text(json.dumps(doc, indent=2) + "\n")
PY

  echo "submitted ${NAME}: job_id=${JOB_ID} cells=${CELL_COUNT} run_root=${RUN_ROOT}"
done

echo ""
echo "=== launch index ==="
echo "${INDEX_JSON}"
cat "${INDEX_JSON}"
