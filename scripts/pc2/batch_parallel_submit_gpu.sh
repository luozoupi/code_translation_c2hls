#!/usr/bin/env bash
# Submit GPU job for a batch_parallel campaign (vLLM only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CAMPAIGN_ROOT="${BATCH_PARALLEL_CAMPAIGN_ROOT:?set BATCH_PARALLEL_CAMPAIGN_ROOT}"
export PC2_SESSION_ID="${PC2_SESSION_ID:-$(basename "${CAMPAIGN_ROOT}")}"
_pc2_configure_session_paths
export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/coordinator.log"
mkdir -p "${CAMPAIGN_ROOT}/flow" "${PC2_SESSION_DIR}"

cd "${C2HLS_ROOT}"
pc2_session_py init --reset >/dev/null
rm -f "${PC2_ENDPOINT_FILE}"

account_args=()
if [[ -n "${PC2_SLURM_ACCOUNT:-}" ]]; then
  account_args=(--account="${PC2_SLURM_ACCOUNT}")
fi

job_id="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --export=ALL,BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",PC2_SESSION_ID="${PC2_SESSION_ID}",PC2_ENDPOINT_FILE="${PC2_ENDPOINT_FILE}" \
    --job-name="bp-llm-${PC2_SESSION_ID}" \
    --output="${CAMPAIGN_ROOT}/slurm-gpu-%j.out" \
    --error="${CAMPAIGN_ROOT}/slurm-gpu-%j.err" \
    "${account_args[@]}" \
    --partition="${PC2_GPU_PARTITION}" \
    --nodes="${PC2_GPU_NODES}" \
    --cpus-per-task="${PC2_GPU_CPUS_PER_TASK}" \
    --mem="${PC2_GPU_MEM}" \
    --gres="gpu:h100:${PC2_GPU_GPUS}" \
    --time="${PC2_WALLTIME:-12:00:00}" \
    "${SCRIPT_DIR}/gpu_serve.sbatch.sh"
)"
echo "${job_id}"
