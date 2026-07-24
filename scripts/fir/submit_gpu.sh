#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p "${FIR_SESSION_DIR}"
rm -f "${FIR_ENDPOINT_FILE}"

account_args=()
if [[ -n "${FIR_SLURM_ACCOUNT}" ]]; then
  account_args=(--account="${FIR_SLURM_ACCOUNT}")
fi

job_id="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,FIR_SESSION_ID="${FIR_SESSION_ID:-}" \
    --job-name="c2hls-fir-llm-${FIR_SESSION_ID:-default}" \
    --output="${FIR_SESSION_DIR}/slurm-gpu-%j.out" \
    --error="${FIR_SESSION_DIR}/slurm-gpu-%j.err" \
    "${account_args[@]}" \
    --partition="${FIR_GPU_PARTITION}" \
    --nodes="${FIR_GPU_NODES}" \
    --cpus-per-task="${FIR_GPU_CPUS_PER_TASK}" \
    --mem="${FIR_GPU_MEM}" \
    --gpus-per-node="h100:${FIR_GPU_GPUS}" \
    --time="${FIR_WALLTIME}" \
    "${SCRIPT_DIR}/gpu_serve.sbatch.sh"
)"
fir_session_py set gpu_job_id "${job_id}"
fir_session_py set gpu_state queued
fir_log "submitted gpu job ${job_id} partition=${FIR_GPU_PARTITION} walltime=${FIR_WALLTIME}"
echo "${job_id}"
