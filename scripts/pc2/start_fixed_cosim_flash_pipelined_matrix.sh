#!/usr/bin/env bash
# Submit 5 PC2 pipelined flash sessions (codegen + synth workers per variant).
#
# Uses run_flash_fixed_cosim_pipelined.py instead of the serial batch runner.
# Same variants / corpus as start_fixed_cosim_flash_matrix.sh.
#
#   ./scripts/pc2/start_fixed_cosim_flash_pipelined_matrix.sh --dry-run
#   ./scripts/pc2/start_fixed_cosim_flash_pipelined_matrix.sh --stamp 20260627_fixed_cosim_flash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=1
STAMP="${C2HLS_FLASH_FIXED_COSIM_STAMP:-$(date +%Y%m%d_%H%M%S)}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_FORCE_WALLTIME="${PC2_FIXED_COSIM_FLASH_WALLTIME:-12:00:00}"
export C2HLS_PIPELINED_SYNTH_WORKERS="${C2HLS_PIPELINED_SYNTH_WORKERS:-4}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
# Parallel csynth on one compute node — force allocation for pipelined runs
# (override any PC2_COMPUTE_* left in the shell from common.sh defaults).
export PC2_COMPUTE_CPUS=64
export PC2_COMPUTE_MEM=256G
PY="${C2HLS_PYTHON:-python3}"

run_batch() {
  "${PY}" scripts/pc2/run_flash_fixed_cosim_pipelined.py --pc2 "$@"
}

VARIANT_KEYS=(nav_o aav_n nav_n noskills aav_o)
declare -A SESSION_ID=(
  [nav_o]=flash_pipelined_cosim_nav_o
  [aav_n]=flash_pipelined_cosim_aav_n
  [nav_n]=flash_pipelined_cosim_nav_n
  [noskills]=flash_pipelined_cosim_noskills
  [aav_o]=flash_pipelined_cosim_aav_o
)

echo "pipelined fixed-cosim flash matrix stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME} synth_workers=${C2HLS_PIPELINED_SYNTH_WORKERS} synth_timeout=${C2HLS_SYNTH_TIMEOUT}s compute=${PC2_COMPUTE_CPUS}cpu/${PC2_COMPUTE_MEM}"
echo "corpus=benchmarks_cosim runner=pipelined benches=$(ls -1d benchmarks_cosim/hlsfactory_* 2>/dev/null | wc -l)"
echo ""

echo "=== skills preflight ==="
"${PY}" scripts/pc2/run_flash_fixed_cosim_batch.py --pc2 --verify-all
echo ""

echo "=== per-variant dry-run ==="
for key in "${VARIANT_KEYS[@]}"; do
  C2HLS_RECORD_FLOW=1 C2HLS_FLASH_FIXED_COSIM_STAMP="${STAMP}" \
    run_batch --variant "${key}" --stamp "${STAMP}" --dry-run
  echo ""
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: preflight ok, no Slurm jobs submitted"
  exit 0
fi

start_one() {
  local session_id="$1"
  local worker_cmd="$2"
  local extra=()
  if [[ "${AUTO_STOP}" -eq 1 ]]; then
    extra+=(--auto-stop-on-complete)
  fi
  "${SCRIPT_DIR}/start_session.sh" --session-id "${session_id}" --worker-cmd "${worker_cmd}" "${extra[@]}"
}

for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  export PC2_SESSION_ID="${sid}"
  _pc2_configure_session_paths
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    echo "stopping existing session ${sid}"
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  fi
done
unset PC2_SESSION_ID

for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  worker_cmd="C2HLS_RECORD_FLOW=1 C2HLS_FLASH_FIXED_COSIM_STAMP=${STAMP} C2HLS_PIPELINED_STAMP_SUFFIX=0 C2HLS_PIPELINED_SYNTH_WORKERS=${C2HLS_PIPELINED_SYNTH_WORKERS} C2HLS_SYNTH_TIMEOUT=${C2HLS_SYNTH_TIMEOUT} ${PY} scripts/pc2/run_flash_fixed_cosim_pipelined.py --pc2 --variant ${key} --stamp ${STAMP}_pipelined"
  echo "submitting pipelined session ${sid}"
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "Submitted 5 pipelined sessions. Monitor:"
for key in "${VARIANT_KEYS[@]}"; do
  echo "  tail -f artifacts/pc2/sessions/${SESSION_ID[$key]}/watch.log"
done
echo "Artifacts under artifacts/pc2/flash_fixed_cosim_<variant>_${STAMP}_pipelined/"
