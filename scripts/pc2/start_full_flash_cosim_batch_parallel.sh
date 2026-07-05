#!/usr/bin/env bash
# Start a full 28-bench flash + RTL cosim batch_parallel campaign.
#
# Usage:
#   ./scripts/pc2/start_full_flash_cosim_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_full_flash_cosim_batch_parallel.sh --stamp 20260702_bp_full_aav_n_park
#   BATCH_PARALLEL_CONFIG=scripts/pc2/batch_parallel_full_aav_n_park.json \
#     PC2_BATCH_PARALLEL_WALLTIME=24:00:00 \
#     ./scripts/pc2/start_full_flash_cosim_batch_parallel.sh
#
# Slurm job names: bpfcosim-{synth,cosim,gpu}-*
# Artifacts: artifacts/pc2/batch_parallel_<stamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_full_aav_n_park.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-aav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpfcosim}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-24:00:00}"

exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" "$@"
