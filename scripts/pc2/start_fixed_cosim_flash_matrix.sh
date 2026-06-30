#!/usr/bin/env bash
# Submit 5 PC2 flash sessions on benchmarks_cosim (fixed corpus), one GPU + one compute each.
#
# Variants (explicit skill JSON — not inferred from legacy artifact dirs):
#   nav_o     No avoids (old)   skills.json (55)
#   aav_n     All+avoids (new)  skills_ii_target_miss_solutions_added(90skills).json
#   nav_n     No avoids (new)   skills_ii_target_miss_solutions_added(73skills).json
#   noskills  No skills
#   aav_o     All+avoids (old)  skills.json (55)
#
# All runs set C2HLS_RECORD_FLOW=1 (see flash_fixed_cosim_lib.py).
#
#   ./scripts/pc2/start_fixed_cosim_flash_matrix.sh --dry-run
#   ./scripts/pc2/start_fixed_cosim_flash_matrix.sh --stamp 20260626_150000
# Auto-stops GPU + watch ~120s after compute worker succeeds (same as other flash launchers).
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
PY="${C2HLS_PYTHON:-python3}"

run_batch() {
  "${PY}" scripts/pc2/run_flash_fixed_cosim_batch.py --pc2 "$@"
}

VARIANT_KEYS=(nav_o aav_n nav_n noskills aav_o)
declare -A SESSION_ID=(
  [nav_o]=flash_fixed_cosim_nav_o
  [aav_n]=flash_fixed_cosim_aav_n
  [nav_n]=flash_fixed_cosim_nav_n
  [noskills]=flash_fixed_cosim_noskills
  [aav_o]=flash_fixed_cosim_aav_o
)

echo "fixed-cosim flash matrix stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME}"
echo "corpus=benchmarks_cosim record_flow=1 benches=$(ls -1d benchmarks_cosim/hlsfactory_* 2>/dev/null | wc -l)"
echo ""

echo "=== skills preflight ==="
run_batch --verify-all
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
  worker_cmd="C2HLS_RECORD_FLOW=1 C2HLS_FLASH_FIXED_COSIM_STAMP=${STAMP} ${PY} scripts/pc2/run_flash_fixed_cosim_batch.py --pc2 --variant ${key} --stamp ${STAMP}"
  echo "submitting session ${sid}"
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "Submitted 5 sessions (5 GPU + 5 compute when GPUs start). Monitor:"
for key in "${VARIANT_KEYS[@]}"; do
  echo "  tail -f artifacts/pc2/sessions/${SESSION_ID[$key]}/watch.log"
done
echo "Artifacts under artifacts/pc2/flash_fixed_cosim_<variant>_${STAMP}/"
