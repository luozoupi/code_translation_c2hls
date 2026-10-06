#!/usr/bin/env bash
# Start AutoSA DSE flash + streaming-dataflow batch_parallel (90-skill aav_n).
#
# Policy: GPU borrow OFF, batch_park ON, park_grace_s=5400, cosim OFF.
# Streaming: as soon as a bench has flash_selected, start its dataflow (do not
# wait for all 35 flash results).
#
# Usage:
#   ./scripts/pc2/start_autosa_dse_flash_dataflow_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_autosa_dse_flash_dataflow_batch_parallel.sh --stamp 20260713_autosa_dse_fd_stream
#
# Artifacts: artifacts/pc2/batch_parallel_autosa_dse_fd_<stamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --borrow-gpu)
      echo "ERROR: this campaign requires --no-borrow-gpu (GPU borrow off)" >&2
      exit 2
      ;;
    --no-borrow-gpu) shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_dse_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-autosa_dse_aav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_dse_fd}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpautfd}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-72:00:00}"
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_PHASEB_FROM_GOLD=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-7200}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_DATAFLOW_MAX_PARALLEL="${C2HLS_DATAFLOW_MAX_PARALLEL:-8}"

EXTRA_ARGS=(--no-borrow-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "=== AutoSA DSE flash+streaming-dataflow batch_parallel ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "gpu_borrow=off park_policy=on park_grace_s=5400"
echo "skills=90 aav_n turns=4 cosim=off PHASEB_FROM_GOLD=0"
echo "streaming=per-bench flash→dataflow max_parallel=${C2HLS_DATAFLOW_MAX_PARALLEL}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_STAMP="${STAMP}" \
    "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
    --stamp "${STAMP}" \
    "${EXTRA_ARGS[@]}"
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
WATCH_LOG="${CAMPAIGN_ROOT}/flow/stream_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpautfd-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=16 \
    --mem=128G \
    --time=96:00:00 \
    --wrap="bash ${SCRIPT_DIR}/wait_autosa_dse_flash_stream_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} --max-parallel ${C2HLS_DATAFLOW_MAX_PARALLEL} >> ${WATCH_LOG} 2>&1"
)"

echo "submitted streaming flash→dataflow watcher job ${POST_JOB}"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"
