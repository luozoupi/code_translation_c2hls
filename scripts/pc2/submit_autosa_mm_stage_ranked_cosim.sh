#!/usr/bin/env bash
# Rank 20260918 stage-box winners and fire-and-forget ranked cosim Slurm jobs.
#
# Writes only under --out (must not contain autosa_mm_variant_sweep_20260918).
# Already-submitted buckets (recorded in flow/job_ids.txt) are skipped.
# --all-candidates submits one job per remaining ranked replicate.
#
# Usage:
#   ./scripts/pc2/submit_autosa_mm_stage_ranked_cosim.sh \
#     --sweep-root artifacts/pc2/autosa_mm_variant_sweep_20260918 \
#     --out artifacts/pc2/autosa_mm_stage_ranked_cosim_20260921
#   ./scripts/pc2/submit_autosa_mm_stage_ranked_cosim.sh ... --all-candidates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

SWEEP_ROOT=""
OUT_ROOT=""
DRY_RUN=0
ALL_CANDIDATES=0
BUCKETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep-root) shift; SWEEP_ROOT="$1"; shift ;;
    --out) shift; OUT_ROOT="$1"; shift ;;
    --bucket) shift; BUCKETS+=("$1"); shift ;;
    --all-candidates) ALL_CANDIDATES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${SWEEP_ROOT}" || ! -d "${SWEEP_ROOT}" ]]; then
  echo "ERROR: --sweep-root required and must be a directory" >&2
  exit 2
fi
if [[ -z "${OUT_ROOT}" ]]; then
  echo "ERROR: --out required" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

SWEEP_ROOT="$("${PY}" -c "from pathlib import Path; print(Path('${SWEEP_ROOT}').resolve())")"
OUT_ROOT="$("${PY}" -c "from pathlib import Path; print(Path('${OUT_ROOT}').resolve())")"

"${PY}" - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
from autosa_mm_stage_ranked_cosim import refuse_frozen_write
refuse_frozen_write(Path("${OUT_ROOT}"))
print("out_ok=${OUT_ROOT}")
PY

FLOW_DIR="${OUT_ROOT}/flow"
mkdir -p "${FLOW_DIR}/slurm" "${FLOW_DIR}/logs"
JOB_IDS="${FLOW_DIR}/job_ids.txt"
touch "${JOB_IDS}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] staging bench + rankings"

"${PY}" "${SCRIPT_DIR}/autosa_mm_stage_ranked_cosim.py" \
  --sweep-root "${SWEEP_ROOT}" \
  --out "${OUT_ROOT}" \
  --stage-bench \
  --write-rankings

if [[ "${ALL_CANDIDATES}" -eq 1 ]]; then
  CAND_JOB_IDS="${FLOW_DIR}/candidate_job_ids.txt"
  touch "${CAND_JOB_IDS}"
  WORK_LIST="$("${PY}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${SCRIPT_DIR}")
from autosa_mm_stage_ranked_cosim import load_sweep_buckets, remaining_candidates

out = Path("${OUT_ROOT}")
job_ids = Path("${FLOW_DIR}") / "candidate_job_ids.txt"
already = set()
if job_ids.is_file():
    for line in job_ids.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            already.add(parts[1])
requested = [b for b in """${BUCKETS[*]}""".split() if b]
buckets = load_sweep_buckets(Path("${SWEEP_ROOT}"), require_kernel=True)
if requested:
    buckets = [b for b in buckets if b["bucket_id"] in requested]
left = remaining_candidates(buckets, out)
n = 0
for cand in left:
    cid = cand.get("cell_id") or ""
    if not cid or cid in already:
        if cid in already:
            print(f"ALREADY\t{cid}", file=sys.stderr)
        continue
    print(f"{cand['bucket_id']}\t{cid}")
    n += 1
print(f"remaining_work={n}", file=sys.stderr)
PY
)"
  if [[ -z "${WORK_LIST}" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] nothing new to submit"
    exit 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submitting per-candidate ranked cosim"
  while IFS=$'\t' read -r BUCKET CANDIDATE; do
    [[ -z "${CANDIDATE:-}" || "${BUCKET}" == ALREADY ]] && continue
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "[dry-run] would sbatch ranked cosim candidate=${CANDIDATE}"
      continue
    fi
    JOB_ID="$(
      sbatch --parsable \
        --chdir="${C2HLS_ROOT}" \
        --job-name="${PC2_BATCH_JOB_PREFIX:-vsc}-${CANDIDATE}" \
        --output="${FLOW_DIR}/slurm/${CANDIDATE}-%j.out" \
        --error="${FLOW_DIR}/slurm/${CANDIDATE}-%j.err" \
        --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
        --partition="${PC2_COMPUTE_PARTITION}" \
        --cpus-per-task="${PC2_COSIM_CPUS:-8}" \
        --mem="${PC2_COSIM_MEM:-32G}" \
        --time="${PC2_COSIM_WALLTIME:-12:00:00}" \
        --export="ALL,C2HLS_STAGE_COSIM_OUT=${OUT_ROOT},C2HLS_STAGE_COSIM_SWEEP=${SWEEP_ROOT},C2HLS_STAGE_COSIM_CANDIDATE=${CANDIDATE},C2HLS_STAGE_COSIM_FORCE=1,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT:-43200},C2HLS_COSIM_TRACE_LEVEL=${C2HLS_COSIM_TRACE_LEVEL:-none},C2HLS_COSIM_BENCHMARKS_ROOT=${OUT_ROOT}/bench" \
        --wrap="bash ${SCRIPT_DIR}/run_autosa_mm_stage_ranked_cosim_bench.sh"
    )"
    JOB_ID="${JOB_ID%%;*}"
    echo "${JOB_ID} ${CANDIDATE}" >> "${CAND_JOB_IDS}"
    echo "  submitted stage ranked cosim ${CANDIDATE} job=${JOB_ID}"
  done <<< "${WORK_LIST}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] per-candidate ranked cosim submit done (async; not waiting)"
  echo "jobs=${CAND_JOB_IDS}"
  exit 0
fi

WORK_LIST="$("${PY}" - <<PY
import json
import sys
from pathlib import Path

out = Path("${OUT_ROOT}")
job_ids = Path("${JOB_IDS}")
already = set()
if job_ids.is_file():
    for line in job_ids.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            already.add(parts[1])

requested = [b for b in """${BUCKETS[*]}""".split() if b]
summary_path = out / "rank_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
rows = []
for entry in summary:
    bid = entry.get("bucket_id") or ""
    if not bid:
        continue
    if requested and bid not in requested:
        continue
    n = int(entry.get("n_candidates") or 0)
    if n < 1:
        print(f"SKIP\t{bid}\t(no candidates)", file=sys.stderr)
        continue
    if bid in already:
        print(f"ALREADY\t{bid}", file=sys.stderr)
        continue
    rows.append(bid)
for bid in rows:
    print(bid)
PY
)"

if [[ -z "${WORK_LIST}" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] nothing new to submit"
  exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submitting stage ranked cosim"

while IFS= read -r BUCKET; do
  [[ -z "${BUCKET}" || "${BUCKET}" == ALREADY || "${BUCKET}" == SKIP ]] && continue
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would sbatch ranked cosim bucket=${BUCKET}"
    continue
  fi
  JOB_ID="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --job-name="${PC2_BATCH_JOB_PREFIX:-vsc}-${BUCKET}" \
      --output="${FLOW_DIR}/slurm/${BUCKET}-%j.out" \
      --error="${FLOW_DIR}/slurm/${BUCKET}-%j.err" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task="${PC2_COSIM_CPUS:-8}" \
      --mem="${PC2_COSIM_MEM:-32G}" \
      --time="${PC2_COSIM_WALLTIME:-48:00:00}" \
      --export="ALL,C2HLS_STAGE_COSIM_OUT=${OUT_ROOT},C2HLS_STAGE_COSIM_SWEEP=${SWEEP_ROOT},C2HLS_STAGE_COSIM_BUCKET=${BUCKET},C2HLS_STAGE_COSIM_FORCE=1,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT:-43200},C2HLS_COSIM_TRACE_LEVEL=${C2HLS_COSIM_TRACE_LEVEL:-none},C2HLS_COSIM_BENCHMARKS_ROOT=${OUT_ROOT}/bench" \
      --wrap="bash ${SCRIPT_DIR}/run_autosa_mm_stage_ranked_cosim_bench.sh"
  )"
  JOB_ID="${JOB_ID%%;*}"
  echo "${JOB_ID} ${BUCKET}" >> "${JOB_IDS}"
  echo "  submitted stage ranked cosim ${BUCKET} job=${JOB_ID}"
done <<< "${WORK_LIST}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage ranked cosim submit done (async; not waiting)"
echo "jobs=${JOB_IDS}"
