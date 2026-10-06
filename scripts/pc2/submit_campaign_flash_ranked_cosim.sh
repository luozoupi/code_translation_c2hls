#!/usr/bin/env bash
# Rank flash-side candidates and fire-and-forget ranked cosim Slurm jobs.
#
# Default: every row in matrix.json (campaign-complete path).
# Per-bench: --bench NAME (repeatable) discovers the cell under variants/ and
# submits immediately — does NOT require campaign-complete or a full matrix.
# Already-submitted benches (recorded in job_ids.txt) are skipped.
#
# Usage:
#   ./scripts/pc2/submit_campaign_flash_ranked_cosim.sh --campaign-root DIR [--dry-run]
#   ./scripts/pc2/submit_campaign_flash_ranked_cosim.sh --campaign-root DIR \
#       --bench hlsfactory_correlation --bench hlsfactory_gramschmidt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CAMPAIGN_ROOT=""
DRY_RUN=0
BENCHES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-root) shift; CAMPAIGN_ROOT="$1"; shift ;;
    --bench) shift; BENCHES+=("$1"); shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${CAMPAIGN_ROOT}" || ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "ERROR: --campaign-root required" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3

FLOW_DIR="${CAMPAIGN_ROOT}/flow/flash_ranked_cosim"
mkdir -p "${FLOW_DIR}/slurm" "${FLOW_DIR}/logs"
JOB_IDS="${FLOW_DIR}/job_ids.txt"
RANK_SUMMARY="${FLOW_DIR}/rank_summary.json"
touch "${JOB_IDS}"
if [[ ! -f "${RANK_SUMMARY}" ]]; then
  echo "[]" > "${RANK_SUMMARY}"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ranking + submitting flash ranked cosim"

# Build work list: either explicit --bench args, or all matrix.json rows.
WORK_LIST="$("${PY}" - <<PY
import json
import sys
from pathlib import Path

root = Path("${CAMPAIGN_ROOT}")
flow = Path("${FLOW_DIR}")
job_ids = Path("${JOB_IDS}")
already = set()
if job_ids.is_file():
    for line in job_ids.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            already.add(parts[1])

requested = [b for b in """${BENCHES[*]}""".split() if b]

def prefer_cell(bench: str):
    best = None
    best_score = None
    for cell in sorted((root / "variants").glob(f"*/{bench}/*")):
        if not cell.is_dir():
            continue
        ready = (
            (cell / f"{bench}_multistep_results.json").is_file()
            or any(cell.glob("*_final.cpp"))
            or any(cell.glob("*_selected.cpp"))
        )
        if not ready:
            continue
        score = (0 if "deepseek" in cell.name else 1, 0 if "devstral" not in cell.name else 2, cell.name)
        if best_score is None or score < best_score:
            best_score = score
            best = cell
    return best

rows = []
if requested:
    for bench in requested:
        cell = prefer_cell(bench)
        if cell is None:
            print(f"SKIP\t{bench}\t(no ready cell)", file=sys.stderr)
            continue
        rows.append({"bench": bench, "cell_dir": str(cell.resolve())})
else:
    matrix = root / "matrix.json"
    if not matrix.is_file():
        raise SystemExit(f"missing {matrix} (pass --bench for per-bench submit before matrix exists)")
    for row in json.loads(matrix.read_text(encoding="utf-8")):
        bench = row.get("bench") or ""
        cell = Path(row.get("cell_dir") or "")
        if not bench or not cell.is_dir():
            continue
        rows.append({"bench": bench, "cell_dir": str(cell.resolve())})

for row in rows:
    bench = row["bench"]
    if bench in already:
        print(f"ALREADY\t{bench}", file=sys.stderr)
        continue
    print(f"{bench}\t{row['cell_dir']}")
PY
)"

if [[ -z "${WORK_LIST}" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] nothing new to submit"
  exit 0
fi

# Rank + merge into rank_summary.json, then sbatch each.
while IFS=$'\t' read -r BENCH CELL_DIR; do
  [[ -z "${BENCH}" || "${BENCH}" == ALREADY || "${BENCH}" == SKIP ]] && continue
  RANKED_JSON="$("${PY}" - <<PY
import json
import sys
from pathlib import Path
sys.path.insert(0, "${SCRIPT_DIR}")
from flash_df_candidate_rank import rank_and_promote

bench = "${BENCH}"
cell = Path("${CELL_DIR}")
ranked = rank_and_promote(cell, bench, side="flash")
entry = {
    "bench": bench,
    "cell_dir": str(cell),
    "rank1_id": ranked[0]["id"] if ranked else None,
    "n_candidates": len(ranked),
}
summary_path = Path("${RANK_SUMMARY}")
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
if not isinstance(summary, list):
    summary = []
summary = [r for r in summary if r.get("bench") != bench]
summary.append(entry)
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(entry))
PY
)"
  RANK1="$("${PY}" -c "import json,sys; print(json.loads(sys.argv[1]).get('rank1_id') or '')" "${RANKED_JSON}")"
  short="${BENCH#hlsfactory_}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would sbatch ranked cosim flash ${BENCH} rank1=${RANK1}"
    continue
  fi
  JOB_ID="$(
    sbatch --parsable \
      --chdir="${C2HLS_ROOT}" \
      --job-name="${PC2_BATCH_JOB_PREFIX:-bphfrc}-frc-${short}" \
      --output="${FLOW_DIR}/slurm/${BENCH}-%j.out" \
      --error="${FLOW_DIR}/slurm/${BENCH}-%j.err" \
      --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
      --partition="${PC2_COMPUTE_PARTITION}" \
      --cpus-per-task="${PC2_COSIM_CPUS:-8}" \
      --mem="${PC2_COSIM_MEM:-32G}" \
      --time="${PC2_COSIM_WALLTIME:-48:00:00}" \
      --export="ALL,BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT},C2HLS_RANKED_COSIM_CELL_DIR=${CELL_DIR},C2HLS_RANKED_COSIM_BENCH=${BENCH},C2HLS_RANKED_COSIM_SIDE=flash,C2HLS_RANKED_COSIM_FORCE=1,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT:-43200},C2HLS_COSIM_TRACE_LEVEL=${C2HLS_COSIM_TRACE_LEVEL:-none},C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_COSIM_BENCHMARKS_ROOT:-${C2HLS_ROOT}/benchmarks_cosim}" \
      --wrap="bash ${SCRIPT_DIR}/run_ranked_cosim_bench.sh"
  )"
  JOB_ID="${JOB_ID%%;*}"
  echo "${JOB_ID} ${BENCH}" >> "${JOB_IDS}"
  echo "  submitted flash ranked cosim ${BENCH} job=${JOB_ID} rank1=${RANK1}"
done <<< "${WORK_LIST}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] flash ranked cosim submit done (async; not waiting)"
echo "jobs=${JOB_IDS}"
