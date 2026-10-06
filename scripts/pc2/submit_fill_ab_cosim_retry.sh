#!/usr/bin/env bash
# Resubmit only cosim_small fill_ab jobs (after gold_kernel TB link fix).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

FILL_ROOT="${C2HLS_ROOT}/artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730"
JOBS="${FILL_ROOT}/jobs.jsonl"
mkdir -p "${FILL_ROOT}/slurm"

# Collect cosim_small indices
mapfile -t COSIM_IDX < <("${C2HLS_PYTHON:-python3}" - <<'PY'
import json
from pathlib import Path
p=Path("/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730/jobs.jsonl")
for line in p.read_text().splitlines():
    j=json.loads(line)
    if j["metric_kind"]=="cosim_small":
        print(j["index"])
PY
)

if [[ ${#COSIM_IDX[@]} -lt 1 ]]; then
  echo "ERROR: no cosim_small jobs" >&2
  exit 2
fi

ARRAY_SPEC="$(IFS=,; echo "${COSIM_IDX[*]}")"
echo "cosim_small indices: ${ARRAY_SPEC}"

export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-21600}"
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_COSIM_EXTRA_ARGS="${C2HLS_COSIM_EXTRA_ARGS:--disable_deadlock_detection}"
export C2HLS_FILL_AB_FORCE=1

WALL="${PC2_FILL_AB_WALLTIME:-6:00:00}"
MEM="${PC2_FILL_AB_MEM:-32G}"
CPUS="${PC2_FILL_AB_CPUS:-8}"
PARTITION="${PC2_FILL_AB_PARTITION:-normal}"

JOBID="$(sbatch --parsable \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${WALL}" \
  --array="${ARRAY_SPEC}%16" \
  --job-name="fill-ab-cosim" \
  --output="${FILL_ROOT}/slurm/fill-cosim-%A_%a.out" \
  --error="${FILL_ROOT}/slurm/fill-cosim-%A_%a.err" \
  --export=ALL,C2HLS_ROOT,C2HLS_SITE=pc2,C2HLS_PART,C2HLS_CLOCK_NS,C2HLS_COSIM_TIMEOUT,C2HLS_COSIM_TRACE_LEVEL,C2HLS_COSIM_XELAB_MT_OFF,C2HLS_COSIM_EXTRA_ARGS,C2HLS_FILL_AB_FORCE \
  "${SCRIPT_DIR}/fill_ab_array.sbatch.sh")"

echo "${JOBID}" | tee "${FILL_ROOT}/slurm/cosim_retry_jobid.txt"
echo "submitted cosim retry array_job=${JOBID} tasks=${ARRAY_SPEC}"
echo "When done: ${C2HLS_PYTHON:-python3} scripts/pc2/report_fill_ab_timeout.py"
