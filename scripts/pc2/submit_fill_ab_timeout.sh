#!/usr/bin/env bash
# Prepare copies and submit fill_ab Slurm array (csynth_full + cosim_small in parallel via one array).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

FILL_ROOT="${C2HLS_ROOT}/artifacts/pc2/reports/hlsfactory_flash_cosim_vs_gold_latrag_off/fill_ab_20260730"
mkdir -p "${FILL_ROOT}/slurm"

export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-21600}"
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_COSIM_EXTRA_ARGS="${C2HLS_COSIM_EXTRA_ARGS:--disable_deadlock_detection}"

echo "=== prepare fill_ab copies ==="
"${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/prepare_fill_ab_timeout.py"

NJOBS="$(wc -l < "${FILL_ROOT}/jobs.jsonl" | tr -d ' ')"
if [[ "${NJOBS}" -lt 1 ]]; then
  echo "ERROR: no jobs" >&2
  exit 2
fi
MAX_IDX=$((NJOBS - 1))

WALL="${PC2_FILL_AB_WALLTIME:-6:00:00}"
MEM="${PC2_FILL_AB_MEM:-32G}"
CPUS="${PC2_FILL_AB_CPUS:-8}"
PARTITION="${PC2_FILL_AB_PARTITION:-normal}"

echo "=== submit array 0-${MAX_IDX} partition=${PARTITION} ==="
JOBID="$(sbatch --parsable \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${WALL}" \
  --array="0-${MAX_IDX}%16" \
  --job-name="fill-ab" \
  --output="${FILL_ROOT}/slurm/fill-%A_%a.out" \
  --error="${FILL_ROOT}/slurm/fill-%A_%a.err" \
  --export=ALL,C2HLS_ROOT,C2HLS_SITE=pc2,C2HLS_PART,C2HLS_CLOCK_NS,C2HLS_COSIM_TIMEOUT,C2HLS_COSIM_TRACE_LEVEL,C2HLS_COSIM_XELAB_MT_OFF,C2HLS_COSIM_EXTRA_ARGS \
  "${SCRIPT_DIR}/fill_ab_array.sbatch.sh")"

echo "${JOBID}" | tee "${FILL_ROOT}/slurm/array_jobid.txt"
echo "submitted array_job=${JOBID} tasks=0-${MAX_IDX}"
echo "Report later: ${C2HLS_PYTHON:-python3} scripts/pc2/report_fill_ab_timeout.py"
