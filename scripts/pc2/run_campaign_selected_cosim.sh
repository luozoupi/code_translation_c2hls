#!/usr/bin/env bash
# Cosim flash-selected kernels for one campaign: one sbatch per cell (NO arrays).
# Requires C2HLS_COSIM_XELAB_MT_OFF=1 (default ON) so hls_eval patches xelab -mt off.
#
# Usage:
#   ./scripts/pc2/run_campaign_selected_cosim.sh --campaign-root <dir> [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
FORCE=0
POLL_SEC="${SELECTED_COSIM_POLL_SEC:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --force) FORCE=1; shift ;;
    --poll-sec) shift; POLL_SEC="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" || ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required (existing dir)" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

export C2HLS_COSIM_XELAB_MT_OFF="${C2HLS_COSIM_XELAB_MT_OFF:-1}"
export C2HLS_FLASH_COSIM_KERNEL=selected
export C2HLS_FLASH_COSIM_FULL_SIZE="${C2HLS_FLASH_COSIM_FULL_SIZE:-1}"
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/benchmarks_cosim}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_COSIM_TRACE_LEVEL="${C2HLS_COSIM_TRACE_LEVEL:-none}"

STAMP="$(basename "${CAMPAIGN_ROOT}")_selected_cosim"
export C2HLS_FLASH_COSIM_STAMP="${STAMP}"
export C2HLS_FLASH_COSIM_ROOT="${CAMPAIGN_ROOT}/flash_selected_cosim"
RUN_ROOT="${C2HLS_FLASH_COSIM_ROOT}/${STAMP}"
mkdir -p "${RUN_ROOT}/slurm" "${RUN_ROOT}/submissions"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] building selected-cosim manifest under ${RUN_ROOT}"
echo "C2HLS_COSIM_XELAB_MT_OFF=${C2HLS_COSIM_XELAB_MT_OFF}"

"${PY}" - "${CAMPAIGN_ROOT}" "${RUN_ROOT}" "${STAMP}" <<'PY'
import json
import os
import sys
from pathlib import Path

campaign_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
stamp = sys.argv[3]
sys.path.insert(0, os.environ.get("C2HLS_ROOT", "."))
from scripts.pc2.flash_cosim_lib import (
    CosimCell,
    cosim_benchmarks_root,
    make_cell_id,
    write_manifest,
)
from flash_flow_artifacts import resolve_cell_kernel_cpp

matrix_path = campaign_root / "matrix.json"
if not matrix_path.is_file():
    raise SystemExit(f"missing matrix.json: {matrix_path}")
rows = json.loads(matrix_path.read_text())
if not isinstance(rows, list):
    raise SystemExit("matrix.json must be a list of rows")

art_base = campaign_root.name
cells = []
index = 0
for row in rows:
    bench = row.get("bench") or ""
    cell_dir = Path(row.get("cell_dir") or "")
    if not bench or not cell_dir.is_dir():
        continue
    kernel = resolve_cell_kernel_cpp(cell_dir, bench, "selected")
    if kernel is None:
        continue
    bench_dir = cosim_benchmarks_root() / bench
    supports = False
    meta_path = bench_dir / "metadata.json"
    if meta_path.is_file():
        try:
            supports = bool(json.loads(meta_path.read_text()).get("supports_cosim"))
        except (OSError, json.JSONDecodeError):
            supports = False
    if not supports:
        continue
    setup_tag = cell_dir.name
    cell_id = make_cell_id(art_base, bench, setup_tag)
    cells.append(
        CosimCell(
            index=index,
            cell_id=cell_id,
            artifact_dir=str(campaign_root),
            artifact_basename=art_base,
            artifact_stamp=stamp,
            matrix_family="hlsfactory_flash_selected",
            bench=bench,
            setup_tag=setup_tag,
            variant=str(row.get("variant") or ""),
            mode="flash",
            model=str(row.get("model") or ""),
            curation_focus="",
            skills_json="",
            cell_dir=str(cell_dir),
            final_cpp=str(kernel.resolve()),
            kernel_source="selected",
            source_matrix_status=str(row.get("status") or ""),
            supports_cosim=True,
        )
    )
    index += 1

path = write_manifest(
    run_root,
    cells,
    extra={
        "campaign_root": str(campaign_root),
        "kernel_source": "selected",
        "cosim_size_mode": "full",
        "c2hls_cosim_xelab_mt_off": os.environ.get("C2HLS_COSIM_XELAB_MT_OFF", "1"),
    },
)
print(json.dumps({"run_root": str(run_root), "cells": len(cells), "manifest": str(path)}))
PY

CELL_COUNT="$(
  "${PY}" -c "import json;print(len(json.load(open('${RUN_ROOT}/manifest.json'))['cells']))"
)"
if [[ "${CELL_COUNT}" -le 0 ]]; then
  echo "WARNING: zero selected-cosim cells; skipping submit" >&2
  echo "0" > "${RUN_ROOT}/submissions/job_ids.txt"
  exit 0
fi

JOB_IDS_FILE="${RUN_ROOT}/submissions/job_ids.txt"
: > "${JOB_IDS_FILE}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submitting ${CELL_COUNT} individual cosim jobs (no array)"
for ((i = 0; i < CELL_COUNT; i++)); do
  CELL_ID="$(
    "${PY}" -c "import json;print(json.load(open('${RUN_ROOT}/manifest.json'))['cells'][${i}]['cell_id'])"
  )"
  SAFE_ID="$(echo "${CELL_ID}" | tr '/:' '__' | cut -c1-80)"
  JOB_ID="$(
    sbatch --parsable \
      --job-name="hfselc-${i}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --cpus-per-task="${PC2_COSIM_CPUS:-8}" \
      --mem="${PC2_COSIM_MEM:-32G}" \
      --time="${PC2_COSIM_WALLTIME:-48:00:00}" \
      --chdir="${C2HLS_ROOT}" \
      --output="${RUN_ROOT}/slurm/cosim-${SAFE_ID}-%j.out" \
      --error="${RUN_ROOT}/slurm/cosim-${SAFE_ID}-%j.err" \
      --export=ALL,C2HLS_ROOT,C2HLS_SITE,C2HLS_FLASH_COSIM_RUN_ROOT="${RUN_ROOT}",C2HLS_FLASH_COSIM_STAMP="${STAMP}",C2HLS_FLASH_COSIM_CELL_ID="${CELL_ID}",C2HLS_COSIM_TIMEOUT,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_BENCHMARKS_ROOT,C2HLS_FLASH_COSIM_KERNEL=selected,C2HLS_COSIM_XELAB_MT_OFF,C2HLS_COSIM_TRACE_LEVEL \
      "${SCRIPT_DIR}/cosim_one.sbatch.sh" \
      $([[ "${FORCE}" -eq 1 ]] && echo --force || true)
  )"
  echo "${JOB_ID}" >> "${JOB_IDS_FILE}"
  echo "  submitted i=${i} cell_id=${CELL_ID} job=${JOB_ID}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for ${CELL_COUNT} selected-cosim jobs"
while true; do
  pending=0
  while read -r jid; do
    [[ -z "${jid}" ]] && continue
    if squeue -j "${jid}" -h 2>/dev/null | grep -q .; then
      pending=$((pending + 1))
    fi
  done < "${JOB_IDS_FILE}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] selected_cosim_pending=${pending}/${CELL_COUNT}"
  if [[ "${pending}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] selected cosim jobs finished; run_root=${RUN_ROOT}"
