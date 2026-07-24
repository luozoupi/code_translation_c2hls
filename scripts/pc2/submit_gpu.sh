#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${PC2_SESSION_DIR}"
rm -f "${PC2_ENDPOINT_FILE}"

account_args=()
if [[ -n "${PC2_SLURM_ACCOUNT}" ]]; then
  account_args=(--account="${PC2_SLURM_ACCOUNT}")
fi

job_id="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,PC2_SESSION_ID="${PC2_SESSION_ID:-}",PC2_ENDPOINT_FILE="${PC2_ENDPOINT_FILE}" \
    --job-name="c2hls-llm-${PC2_SESSION_ID:-default}" \
    --output="${PC2_SESSION_DIR}/slurm-gpu-%j.out" \
    --error="${PC2_SESSION_DIR}/slurm-gpu-%j.err" \
    "${account_args[@]}" \
    --partition="${PC2_GPU_PARTITION}" \
    --nodes="${PC2_GPU_NODES}" \
    --cpus-per-task="${PC2_GPU_CPUS_PER_TASK}" \
    --mem="${PC2_GPU_MEM}" \
    --gres="gpu:h100:${PC2_GPU_GPUS}" \
    --time="${PC2_WALLTIME}" \
    "${SCRIPT_DIR}/gpu_serve.sbatch.sh"
)"
pc2_session_py set gpu_job_id "${job_id}"
pc2_session_py set gpu_state queued
pc2_log "submitted gpu job ${job_id} partition=${PC2_GPU_PARTITION} walltime=${PC2_WALLTIME}"
echo "${job_id}"
