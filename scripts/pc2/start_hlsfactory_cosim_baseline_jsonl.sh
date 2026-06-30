#!/usr/bin/env bash
# Submit csynth+csim JSONL generation for benchmarks_cosim/hls_baseline_cosim.cpp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_HLSFACTORY_BASELINE_STAMP:-20260616_benchmarks}"
OUT_JSONL="${C2HLS_HLSFACTORY_BASELINE_JSONL:-${C2HLS_ROOT}/misc/hlsfactory_cosim_baseline_u280_${STAMP}.jsonl}"
PARTITION="${PC2_COMPUTE_PARTITION:-normal}"
WALLTIME="${PC2_BASELINE_JSONL_WALLTIME:-8:00:00}"
CPUS="${PC2_COMPUTE_CPUS:-8}"
MEM="${PC2_COMPUTE_MEM:-32G}"

export C2HLS_HLSFACTORY_BASELINE_STAMP="${STAMP}"
export C2HLS_HLSFACTORY_BASELINE_JSONL="${OUT_JSONL}"

mkdir -p artifacts/pc2/baseline_cosim/slurm

SBATCH_ARGS=(
  --job-name="hlsf-cosim-bl"
  --chdir="${C2HLS_ROOT}"
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${WALLTIME}"
  --export=ALL,C2HLS_ROOT,C2HLS_SITE,C2HLS_HLSFACTORY_BASELINE_STAMP,C2HLS_HLSFACTORY_BASELINE_JSONL
  --output="${C2HLS_ROOT}/artifacts/pc2/baseline_cosim/slurm/cosim-baseline-jsonl-%j.out"
  --error="${C2HLS_ROOT}/artifacts/pc2/baseline_cosim/slurm/cosim-baseline-jsonl-%j.err"
)

if [[ -n "${PC2_SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="${PC2_SLURM_ACCOUNT}")
fi

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/hlsfactory_cosim_baseline_jsonl.sbatch.sh")"
pc2_log "submitted hlsfactory cosim baseline jsonl job ${JOB_ID} stamp=${STAMP} out=${OUT_JSONL}"
echo "${JOB_ID}"
