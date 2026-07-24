#!/usr/bin/env bash
# Fir flash smoke via supervised GPU+compute session (auto-stops GPU 2 min after worker).
#
# Usage:
#   ./scripts/fir/run_flash_smoke_session.sh --dry-run
#   ./scripts/fir/run_flash_smoke_session.sh --submit
#   ./scripts/fir/run_flash_smoke_session.sh --submit --benches hlsfactory_2mm,hlsfactory_lu,hlsfactory_3mm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
SESSION_ID="${FIR_FLASH_SMOKE_SESSION_ID:-fir_flash_smoke}"
STAMP="${C2HLS_FIR_FLASH_SMOKE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHES="${C2HLS_FIR_FLASH_SMOKE_BENCHES:-hlsfactory_2mm,hlsfactory_lu,hlsfactory_3mm}"
DRY_RUN=0
SUBMIT=0
AUTO_STOP=1
COSIM=0

usage() {
  cat <<EOF
Usage: $0 [--dry-run | --submit] [options]

Modes (exactly one required):
  --dry-run    Preflight + manifest plan; no Slurm jobs
  --submit     Submit one Fir GPU+compute session

Options:
  --stamp STAMP       Artifact stamp (default: date-based)
  --benches A,B       Comma-separated benchmark names
  --cosim             90-skill flash with RTL cosim + LLM repair (longer walltime)
  --no-auto-stop      Keep GPU running after compute worker succeeds
  -h, --help          Show this help

Defaults: benches=${BENCHES}
          session_id=${SESSION_ID}
          auto_stop_gpu_delay=\${FIR_AUTO_STOP_DELAY_SEC:-120}s after compute completes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --cosim) COSIM=1; shift ;;
    --no-auto-stop) AUTO_STOP=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 && "${SUBMIT}" -eq 1 ]]; then
  echo "ERROR: use --dry-run or --submit, not both" >&2
  exit 2
fi
if [[ "${DRY_RUN}" -eq 0 && "${SUBMIT}" -eq 0 ]]; then
  echo "ERROR: specify --dry-run or --submit" >&2
  usage >&2
  exit 2
fi

if [[ "${COSIM}" -eq 1 ]]; then
  export C2HLS_FIR_FLASH_COSIM=1
  export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-7200}"
  export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
  export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-57600}"
  export C2HLS_TURNS="${C2HLS_TURNS:-4}"
  export C2HLS_QUALITY_REPAIR_TURNS="${C2HLS_QUALITY_REPAIR_TURNS:-2}"
  export FIR_FORCE_WALLTIME="${FIR_FORCE_WALLTIME:-18:00:00}"
  FIR_WALLTIME="${FIR_FORCE_WALLTIME}"
fi

export C2HLS_FIR_FLASH_SMOKE_STAMP="${STAMP}"

worker_cmd=(
  "${PY}" scripts/fir/run_flash_smoke_batch.py
  --fir
  --benches "${BENCHES}"
  --stamp "${STAMP}"
)
if [[ "${COSIM}" -eq 1 ]]; then
  worker_cmd+=(--cosim)
fi
worker_cmd_str="${worker_cmd[*]}"

echo "Fir flash smoke stamp=${STAMP} walltime=${FIR_WALLTIME}"
echo "benches=${BENCHES} session_id=${SESSION_ID} cosim=${COSIM}"
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${worker_cmd[@]}" --dry-run
  echo "dry-run ok"
  exit 0
fi

start_args=(--session-id "${SESSION_ID}" --worker-cmd "${worker_cmd_str}")
if [[ "${AUTO_STOP}" -eq 1 ]]; then
  start_args+=(--auto-stop-on-complete)
else
  start_args+=(--no-auto-stop)
fi

echo "=== submitting Fir session ${SESSION_ID} ==="
echo "worker: ${worker_cmd_str}"
echo "gpu policy: scancel gpu ${FIR_AUTO_STOP_DELAY_SEC:-120}s after compute completes"
"${SCRIPT_DIR}/start_session.sh" "${start_args[@]}"

echo ""
echo "Submitted Fir flash smoke (1 GPU + 1 compute when GPU starts)."
if [[ "${COSIM}" -eq 1 ]]; then
  echo "Artifacts: artifacts/fir/flash_cosim_${STAMP}/"
else
  echo "Artifacts: artifacts/fir/flash_smoke_${STAMP}/"
fi
echo "Monitor: tail -f artifacts/fir/sessions/${SESSION_ID}/watch.log"
