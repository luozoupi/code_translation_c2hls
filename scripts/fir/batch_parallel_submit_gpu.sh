#!/usr/bin/env bash
# Submit GPU job for a Fir batch_parallel campaign (shared vLLM).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
export FIR_BATCH_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
_fir_configure_session_paths
export FIR_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
export FIR_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"
if [[ "${PRESERVE_ENDPOINT:-0}" != "1" ]]; then
  rm -f "${FIR_ENDPOINT_FILE}"
fi

cd "${C2HLS_ROOT}"

account_args=()
if [[ -n "${FIR_SLURM_ACCOUNT:-}" ]]; then
  account_args=(--account="${FIR_SLURM_ACCOUNT}")
fi

job_tag="${FIR_JOB_TAG:-$(basename "${CAMPAIGN_ROOT}")}"
job_prefix="$(fir_batch_job_prefix "${CAMPAIGN_ROOT}")"

job_id="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",FIR_BATCH_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
    --job-name="${job_prefix}-gpu-${job_tag}" \
    --output="${CAMPAIGN_ROOT}/slurm-gpu-%j.out" \
    --error="${CAMPAIGN_ROOT}/slurm-gpu-%j.err" \
    "${account_args[@]}" \
    --partition="${FIR_GPU_PARTITION}" \
    --nodes="${FIR_GPU_NODES}" \
    --cpus-per-task="${FIR_GPU_CPUS_PER_TASK}" \
    --mem="${FIR_GPU_MEM}" \
    --gpus-per-node="h100:${FIR_GPU_GPUS}" \
    --time="${FIR_WALLTIME:-12:00:00}" \
    "${SCRIPT_DIR}/gpu_serve.sbatch.sh"
)"
job_id="${job_id%%;*}"
echo "${job_id}"
