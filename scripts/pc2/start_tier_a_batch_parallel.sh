#!/usr/bin/env bash
# Start a tier_A_ready flash batch_parallel campaign (parallel gold gate + synth/csim).
#
# Usage:
#   ./scripts/pc2/start_tier_a_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_tier_a_batch_parallel.sh --stamp 20260701_tier_a_bp
#   ./scripts/pc2/start_tier_a_batch_parallel.sh --borrow-gpu   # reuse active vLLM, no new gpu_h100
#   C2HLS_TIER_A_FLASH_BENCHES="bench1,bench2" ./scripts/pc2/start_tier_a_batch_parallel.sh
#
# Config: BATCH_PARALLEL_CONFIG (default batch_parallel_tier_a_flash.json)
# Artifacts: artifacts/pc2/batch_parallel_complex_<stamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_tier_a_flash.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-tier_a_90}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_complex}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpcplx}"
export PC2_WALLTIME="${PC2_TIER_A_FLASH_WALLTIME:-12:00:00}"

exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" "$@"
