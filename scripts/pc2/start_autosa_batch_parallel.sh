#!/usr/bin/env bash
# Start AutoSA 21-kernel flash batch_parallel (nav_n, 90 skills no avoids, csim+csynth only).
#
# Usage:
#   ./scripts/pc2/start_autosa_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_autosa_batch_parallel.sh --stamp 20260708_autosa_nav_n
#
# Env overrides:
#   C2HLS_CSIM_TIMEOUT=1800   (30 min, default)
#   C2HLS_SYNTH_TIMEOUT=14400 (4 h, default)
#   PC2_BATCH_PARALLEL_WALLTIME=8:00:00
#
# Artifacts: artifacts/pc2/batch_parallel_autosa_<stamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
TRY_BORROW=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --borrow-gpu) TRY_BORROW=1; shift ;;
    --no-borrow-gpu) TRY_BORROW=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

echo "=== preparing autosa_ready corpus (21 kernels) ==="
"${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py"

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_21.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-autosa_nav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpautosa}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-8:00:00}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"

EXTRA_ARGS=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

if [[ "${TRY_BORROW}" -eq 1 ]]; then
  CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/batch_parallel_autosa_${STAMP}"
  export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  export PC2_SESSION_DIR="${CAMPAIGN_ROOT}"
  export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
  export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"
  mkdir -p "${CAMPAIGN_ROOT}/flow"
  if "${SCRIPT_DIR}/borrow_gpu.sh" 2>/dev/null; then
    echo "borrowed existing GPU endpoint; submitting campaign without new gpu_h100 job"
    exec env BATCH_PARALLEL_STAMP="${STAMP}" \
      "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
      --stamp "${STAMP}" \
      --borrow-gpu \
      "${EXTRA_ARGS[@]}"
  fi
  echo "no borrowable GPU found; submitting dedicated gpu_h100 job with batch_park policy"
fi

exec env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"
