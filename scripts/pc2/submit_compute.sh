#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}"

gpu_dep="$(pc2_session_py get gpu_job_id 2>/dev/null || true)"
dep_args=()

if pc2_session_is_borrowed_gpu; then
  if ! pc2_llm_ready; then
    pc2_log "ERROR: refuse compute submit — borrowed LLM endpoint is not healthy"
    exit 1
  fi
else
  if [[ -z "${gpu_dep}" || "${gpu_dep}" == "None" || "${gpu_dep}" == "null" ]]; then
    pc2_log "ERROR: refuse compute submit — no gpu_job_id (GPU must be submitted and running first)"
    exit 1
  fi

  if ! pc2_job_is_running "${gpu_dep}"; then
    pc2_log "ERROR: refuse compute submit — gpu job ${gpu_dep} is not RUNNING yet"
    exit 1
  fi

  dep_args=(--dependency="after:${gpu_dep}")
fi

account_args=()
if [[ -n "${PC2_SLURM_ACCOUNT}" ]]; then
  account_args=(--account="${PC2_SLURM_ACCOUNT}")
fi

job_id="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,PC2_SESSION_ID="${PC2_SESSION_ID:-}",PC2_ENDPOINT_FILE="${PC2_ENDPOINT_FILE}" \
    --job-name="c2hls-vitis-${PC2_SESSION_ID:-default}" \
    --output="${PC2_SESSION_DIR}/slurm-compute-%j.out" \
    --error="${PC2_SESSION_DIR}/slurm-compute-%j.err" \
    "${account_args[@]}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task="${PC2_COMPUTE_CPUS}" \
    --mem="${PC2_COMPUTE_MEM}" \
    --time="${PC2_WALLTIME}" \
    "${dep_args[@]}" \
    "${SCRIPT_DIR}/compute_worker.sbatch.sh"
)"
pc2_session_py set compute_job_id "${job_id}"
pc2_session_py set compute_state queued
pc2_log "submitted compute job ${job_id} partition=${PC2_COMPUTE_PARTITION} walltime=${PC2_WALLTIME} dependency=after:${gpu_dep}"
echo "${job_id}"
