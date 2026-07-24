#!/usr/bin/env bash
# Fir 0-shot flash on hlsfactory corpus (28 parallel compute nodes).
#
#   ./scripts/fir/start_zero_shot_cosim_full_campaign.sh --variant phaseb
#   ./scripts/fir/start_zero_shot_cosim_full_campaign.sh --variant direct
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

VARIANT="${C2HLS_ZERO_SHOT_VARIANT:-phaseb}"
STAMP="${C2HLS_ZERO_SHOT_STAMP:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) shift; VARIANT="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${STAMP}" ]]; then
  STAMP="$(date -u +%Y%m%d_%H%M%S)_abs_zero_shot_${VARIANT}"
fi

case "${VARIANT}" in
  phaseb|direct) ;;
  *) echo "Unknown variant: ${VARIANT} (expected phaseb or direct)" >&2; exit 2 ;;
esac

CONFIG="${SCRIPT_DIR}/batch_parallel_zero_shot_${VARIANT}.json"
export BATCH_PARALLEL_CONFIG="${CONFIG}"
export BATCH_PARALLEL_STAMP="${STAMP}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="abs_zero_shot_cosim_${VARIANT}"
export FIR_BATCH_JOB_PREFIX="$([[ "${VARIANT}" == phaseb ]] && echo firzsa || echo firzsd)"

export FIR_GPU_PARTITION=gpubase_bynode_b1
export FIR_SLURM_ACCOUNT=def-zhenman_gpu
export FIR_COMPUTE_SLURM_ACCOUNT=def-zhenman
export FIR_WALLTIME=3:00:00
export FIR_FORCE_WALLTIME=3:00:00
export FIR_COMPUTE_WALLTIME=18:00:00
export FIR_BATCH_PARALLEL_WALLTIME=18:00:00
export FIR_GPU_PRESUBMIT_SEC=600
export C2HLS_TURNS="${C2HLS_TURNS:-1}"
export C2HLS_SYNTH_TIMEOUT=14400
export C2HLS_CSIM_TIMEOUT=600
export C2HLS_COSIM_TIMEOUT=57600
export C2HLS_RUN_COSIM=1

echo "=== Fir absolute 0-shot flash (${VARIANT}, no repair) ==="
echo "config=${CONFIG}"
echo "stamp=${STAMP}"
echo "gpu: ${FIR_GPU_PARTITION} walltime=${FIR_WALLTIME} presubmit=${FIR_GPU_PRESUBMIT_SEC}s policy=always_on borrow=off"
echo "compute: 28 nodes x 1 worker, ${FIR_COMPUTE_WALLTIME}, 24 CPU / 128G"
echo "repair: turns=${C2HLS_TURNS} C2HLS_DISABLE_REPAIR=1 correctness_repair=off quality_repair=0"

echo "timeouts: csynth=${C2HLS_SYNTH_TIMEOUT}s csim=${C2HLS_CSIM_TIMEOUT}s cosim=${C2HLS_COSIM_TIMEOUT}s"

ARGS=(--config "${CONFIG}" --stamp "${STAMP}" --no-borrow-gpu)
[[ "${DRY_RUN}" -eq 1 ]] && ARGS+=(--dry-run)

exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" "${ARGS[@]}"
