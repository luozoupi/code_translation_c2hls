#!/usr/bin/env bash
# Submit one standalone cosim Slurm job for a manifest cell_id.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CELL_ID="${1:-}"
RUN_ROOT="${2:-${C2HLS_FLASH_COSIM_RUN_ROOT:-}}"
if [[ -z "${CELL_ID}" || -z "${RUN_ROOT}" ]]; then
  echo "usage: $0 <cell_id> [cosim_run_root]" >&2
  exit 2
fi

MANIFEST="${RUN_ROOT}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing manifest: ${MANIFEST}" >&2
  exit 2
fi

export C2HLS_FLASH_COSIM_RUN_ROOT="${RUN_ROOT}"
export C2HLS_FLASH_COSIM_CELL_ID="${CELL_ID}"

WALLTIME="${PC2_COSIM_WALLTIME:-4:00:00}"
CPUS="${PC2_COSIM_CPUS:-8}"
MEM="${PC2_COSIM_MEM:-32G}"
PARTITION="${PC2_COSIM_PARTITION:-${PC2_COMPUTE_PARTITION:-normal}}"

SBATCH_ARGS=(
  --job-name="cosim-${CELL_ID:0:40}"
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${WALLTIME}"
  --export=ALL,C2HLS_ROOT,C2HLS_SITE,C2HLS_FLASH_COSIM_RUN_ROOT,C2HLS_FLASH_COSIM_CELL_ID,C2HLS_COSIM_TIMEOUT
)

if [[ -n "${PC2_SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="${PC2_SLURM_ACCOUNT}")
fi

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/cosim_one.sbatch.sh" | awk '{print $NF}')"
mkdir -p "${RUN_ROOT}/submissions"
echo "${JOB_ID} ${CELL_ID}" >> "${RUN_ROOT}/submissions/individual_jobs.log"
echo "submitted cell_id=${CELL_ID} job_id=${JOB_ID}"
