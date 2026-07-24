#!/usr/bin/env bash
# Start MachSuite flash+dataflow batch_parallel (90-skill aav_n, csim+csynth+cosim).
#
# Policy: GPU borrow OFF, batch_park ON, park_grace_s=5400 (+1h vs 1800 default).
# Repair rounds: turns=4 for flash; dataflow uses 4 repair + 4 contract rounds.
#
# Usage:
#   ./scripts/pc2/start_tier_b_machsuite_flash_dataflow_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_tier_b_machsuite_flash_dataflow_batch_parallel.sh --stamp 20260710_machsuite_fd
#
# Artifacts: artifacts/pc2/batch_parallel_machsuite_fd_<stamp>/
# After flash completes, a watcher exports flash_selected, runs dataflow+cosim,
# then exports dataflow_selected.
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

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_machsuite_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-tier_b_aav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_machsuite_fd}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpmachfd}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
POST_WALLTIME="${PC2_MACHSUITE_POST_WALLTIME:-7-00:00:00}"

EXTRA_ARGS=(--no-borrow-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "=== MachSuite flash+dataflow batch_parallel ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "gpu_borrow=off park_policy=on park_grace_s=5400"
echo "skills=90 aav_n turns=4 cosim=on max_repair_attempt=${C2HLS_MAX_REPAIR_ATTEMPT}"
echo "post_watcher_walltime=${POST_WALLTIME}"

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
WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpmachfd-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time="${POST_WALLTIME}" \
    --wrap="bash ${SCRIPT_DIR}/wait_machsuite_flash_then_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
)"

echo "submitted post flash→dataflow watcher job ${POST_JOB}"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"
