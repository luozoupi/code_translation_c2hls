#!/usr/bin/env bash
# Full hlsfactory_* flash batch_parallel on Fir (no cosim, shared/borrowed GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_full_hlsfactory.json}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
WALLTIME="${FIR_BATCH_PARALLEL_WALLTIME:-16:00:00}"
COMPUTE_WALLTIME="${FIR_COMPUTE_WALLTIME:-${WALLTIME}}"
BORROW=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift; CONFIG="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --walltime) shift; WALLTIME="$1"; shift ;;
    --no-borrow-gpu) BORROW=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

export BATCH_PARALLEL_CONFIG="${CONFIG}"
export FIR_BATCH_PARALLEL_WALLTIME="${WALLTIME}"
export FIR_COMPUTE_WALLTIME="${COMPUTE_WALLTIME}"
export BATCH_PARALLEL_STAMP="${STAMP}"

args=(--config "${CONFIG}" --stamp "${STAMP}")
[[ "${BORROW}" -eq 1 ]] && args+=(--borrow-gpu)
[[ "${DRY_RUN}" -eq 1 ]] && args+=(--dry-run)

exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" "${args[@]}"
