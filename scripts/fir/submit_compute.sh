#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}"

gpu_dep="$(fir_session_py get gpu_job_id 2>/dev/null || true)"
dep_args=()

if [[ -z "${gpu_dep}" || "${gpu_dep}" == "None" || "${gpu_dep}" == "null" ]]; then
  fir_log "ERROR: refuse compute submit — no gpu_job_id (GPU must be submitted and running first)"
  exit 1
fi

if ! fir_job_is_running "${gpu_dep}"; then
  fir_log "ERROR: refuse compute submit — gpu job ${gpu_dep} is not RUNNING yet"
  exit 1
fi

dep_args=(--dependency="after:${gpu_dep}")

account_args=()
compute_account="${FIR_COMPUTE_SLURM_ACCOUNT:-${FIR_SLURM_ACCOUNT:-}}"
if [[ -n "${compute_account}" ]]; then
  account_args=(--account="${compute_account}")
fi

job_id="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,FIR_SESSION_ID="${FIR_SESSION_ID:-}" \
    --job-name="c2hls-fir-vitis-${FIR_SESSION_ID:-default}" \
    --output="${FIR_SESSION_DIR}/slurm-compute-%j.out" \
    --error="${FIR_SESSION_DIR}/slurm-compute-%j.err" \
    "${account_args[@]}" \
    ${FIR_COMPUTE_PARTITION:+--partition="${FIR_COMPUTE_PARTITION}"} \
    --cpus-per-task="${FIR_COMPUTE_CPUS}" \
    --mem="${FIR_COMPUTE_MEM}" \
    --time="${FIR_WALLTIME}" \
    "${dep_args[@]}" \
    "${SCRIPT_DIR}/compute_worker.sbatch.sh"
)"
fir_session_py set compute_job_id "${job_id}"
fir_session_py set compute_state queued
fir_log "submitted compute job ${job_id} partition=${FIR_COMPUTE_PARTITION} walltime=${FIR_WALLTIME} dependency=after:${gpu_dep}"
echo "${job_id}"
