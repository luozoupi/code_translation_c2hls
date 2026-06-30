#!/usr/bin/env bash
# Serial multistep pilot: 5 benches, aav_n, one PC2 session.
#
#   ./scripts/pc2/start_multistep_fixed_cosim_pilot.sh --dry-run
#   ./scripts/pc2/start_multistep_fixed_cosim_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
STAMP="${C2HLS_MULTISTEP_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_multistep_pilot}"
PY="${C2HLS_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_FORCE_WALLTIME="${PC2_MULTISTEP_PILOT_WALLTIME:-12:00:00}"
export C2HLS_MULTISTEP_FIXED_COSIM_STAMP="${STAMP}"

echo "multistep serial pilot stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME}"

"${PY}" scripts/pc2/run_multistep_fixed_cosim_batch.py --pc2 --verify-all
"${PY}" scripts/pc2/run_multistep_fixed_cosim_batch.py --pc2 --pilot --stamp "${STAMP}" --dry-run

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
  exit 0
fi

WORKER_CMD="C2HLS_MULTISTEP_FIXED_COSIM_STAMP=${STAMP} ${PY} scripts/pc2/run_multistep_fixed_cosim_batch.py --pc2 --pilot --stamp ${STAMP}"
"${SCRIPT_DIR}/start_session.sh" \
  --session-id multistep_fixed_cosim_pilot_aav_n \
  --worker-cmd "${WORKER_CMD}" \
  --auto-stop-on-complete
