#!/usr/bin/env bash
# Pipelined multistep pilot: 5 benches, aav_n, extended walltime.
#
#   ./scripts/pc2/start_multistep_fixed_cosim_pipelined.sh --dry-run
#   ./scripts/pc2/start_multistep_fixed_cosim_pipelined.sh --pilot
#   ./scripts/pc2/start_multistep_fixed_cosim_pipelined.sh  # full 28 benches
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
PILOT=0
AUTO_STOP=1
VARIANT="${C2HLS_MULTISTEP_VARIANT:-aav_n}"
STAMP="${C2HLS_MULTISTEP_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_multistep}"
PY="${C2HLS_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --pilot) PILOT=1; shift ;;
    --variant) shift; VARIANT="$1"; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export C2HLS_MULTISTEP_VARIANT="${VARIANT}"

if [[ "${PILOT}" -eq 1 ]]; then
  export PC2_FORCE_WALLTIME="${PC2_MULTISTEP_PILOT_WALLTIME:-12:00:00}"
  SESSION_ID="multistep_pipelined_cosim_pilot_${VARIANT}"
  EXTRA_ARGS="--pilot --variant ${VARIANT}"
else
  export PC2_FORCE_WALLTIME="${PC2_MULTISTEP_FULL_WALLTIME:-48:00:00}"
  SESSION_ID="multistep_pipelined_cosim_${VARIANT}"
  EXTRA_ARGS="--variant ${VARIANT}"
fi

export PC2_COMPUTE_CPUS=64
export PC2_COMPUTE_MEM=256G
export C2HLS_PIPELINED_SYNTH_WORKERS="${C2HLS_PIPELINED_SYNTH_WORKERS:-4}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_MULTISTEP_FIXED_COSIM_STAMP="${STAMP}"

echo "multistep pipelined variant=${VARIANT} stamp=${STAMP} pilot=${PILOT} walltime=${PC2_FORCE_WALLTIME}"
echo "synth_workers=${C2HLS_PIPELINED_SYNTH_WORKERS} compute=${PC2_COMPUTE_CPUS}cpu/${PC2_COMPUTE_MEM}"

"${PY}" scripts/pc2/run_multistep_fixed_cosim_pipelined.py --pc2 ${EXTRA_ARGS} --stamp "${STAMP}" --dry-run

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
  exit 0
fi

export PC2_SLURM_ACCOUNT="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}"

WORKER_CMD="C2HLS_MULTISTEP_FIXED_COSIM_STAMP=${STAMP} ${PY} scripts/pc2/run_multistep_fixed_cosim_pipelined.py --pc2 ${EXTRA_ARGS} --stamp ${STAMP}"
START_ARGS=(--session-id "${SESSION_ID}" --worker-cmd "${WORKER_CMD}")
if [[ "${AUTO_STOP}" -eq 1 ]]; then
  START_ARGS+=(--auto-stop-on-complete)
fi
"${SCRIPT_DIR}/start_session.sh" "${START_ARGS[@]}"
