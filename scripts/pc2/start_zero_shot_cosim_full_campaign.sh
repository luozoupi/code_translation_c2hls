#!/usr/bin/env bash
# Full 28-bench 0-shot flash on benchmarks_cosim/ via batch_parallel (max parallelism).
#
# Per variant (phaseb | direct):
#   1 GPU node  (gpu_b1, 3h walltime, rolling renew at 10m left)
#   28 synth compute nodes (16 CPU, 64G each, 1 worker, 18h)
#   28 cosim compute nodes (16 CPU, 64G each, 1 worker, 18h)
#
#   ./scripts/pc2/start_zero_shot_cosim_full_campaign.sh --variant phaseb --dry-run
#   ./scripts/pc2/start_zero_shot_cosim_full_campaign.sh --variant direct
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
VARIANT="${C2HLS_ZERO_SHOT_VARIANT:-phaseb}"
STAMP="${C2HLS_ZERO_SHOT_STAMP:-$(date +%Y%m%d)_zero_shot_cosim_${VARIANT}}"
PY="${C2HLS_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --variant) shift; VARIANT="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${VARIANT}" in
  phaseb|direct) ;;
  *)
    echo "Unknown variant: ${VARIANT} (expected phaseb or direct)" >&2
    exit 2
    ;;
esac

CONFIG="${SCRIPT_DIR}/batch_parallel_zero_shot_${VARIANT}.json"
export BATCH_PARALLEL_CONFIG="${CONFIG}"
export BATCH_PARALLEL_VARIANT="${VARIANT}"
export BATCH_PARALLEL_STAMP="${STAMP}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="zero_shot_cosim_${VARIANT}"
export PC2_BATCH_JOB_PREFIX="$([[ "${VARIANT}" == phaseb ]] && echo bpzsp || echo bpzsd)"
export PC2_SLURM_ACCOUNT="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}"
export PC2_GPU_PARTITION=gpu_b1
export PC2_GPU_WALLTIME=3:00:00
export PC2_BATCH_PARALLEL_WALLTIME=18:00:00
export PC2_FORCE_WALLTIME=18:00:00
export PC2_BORROW_GPU=0
export C2HLS_TURNS="${C2HLS_TURNS:-4}"
export C2HLS_SYNTH_TIMEOUT=7200
export C2HLS_CSIM_TIMEOUT=600
export C2HLS_COSIM_TIMEOUT=57600

OUT_DIR="artifacts/pc2/zero_shot_cosim_${VARIANT}_${STAMP}"

echo "=== 0-shot flash cosim (batch_parallel) variant=${VARIANT} ==="
echo "config=${CONFIG}"
echo "stamp=${STAMP}"
echo "artifacts=${OUT_DIR}"
echo "gpu: partition=${PC2_GPU_PARTITION} walltime=${PC2_GPU_WALLTIME} renew_before=600s policy=always_on borrow=off park=off"
echo "compute: 28 synth + 28 cosim nodes, 16 CPU / 64G each, walltime=${PC2_BATCH_PARALLEL_WALLTIME}"
echo "timeouts: csynth=${C2HLS_SYNTH_TIMEOUT}s csim=${C2HLS_CSIM_TIMEOUT}s cosim=${C2HLS_COSIM_TIMEOUT}s"

ARGS=(--config "${CONFIG}" --stamp "${STAMP}" --variant "${VARIANT}" --no-borrow-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  ARGS+=(--dry-run)
fi

exec "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" "${ARGS[@]}"
