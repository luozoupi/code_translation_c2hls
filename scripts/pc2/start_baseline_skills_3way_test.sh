#!/usr/bin/env bash
# Parallel 3-way flash test on all hlsfactory_* kernels (csim + csynth only; no cosim).
#
# Three independent sessions submit in parallel (each gets its own GPU + Vitis job):
#   noskills_new
#   all_new_skills_avoids_global
#   all_new_skills_no_avoids_global
#
# Usage:
#   ./scripts/pc2/start_baseline_skills_3way_test.sh --dry-run
#   ./scripts/pc2/start_baseline_skills_3way_test.sh --auto-stop-on-complete
#   ./scripts/pc2/start_baseline_skills_3way_test.sh --stamp 20260622_120000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
STAMP="${C2HLS_BASELINE_3WAY_STAMP:-$(date +%Y%m%d_%H%M%S)}"

VARIANT_KEYS=(
  noskills_new
  all_new_skills_avoids_global
  all_new_skills_no_avoids_global
)

declare -A STAMP_ENV=(
  [noskills_new]=C2HLS_FLASH_NOSKILLS_NEW_STAMP
  [all_new_skills_avoids_global]=C2HLS_FLASH_ALL_NEW_SKILLS_AVOIDS_GLOBAL_STAMP
  [all_new_skills_no_avoids_global]=C2HLS_FLASH_ALL_NEW_SKILLS_NO_AVOIDS_GLOBAL_STAMP
)

declare -A SESSION_ID=(
  [noskills_new]=flash_noskills_new
  [all_new_skills_avoids_global]=flash_all_new_skills_avoids_global
  [all_new_skills_no_avoids_global]=flash_all_new_skills_no_avoids_global
)

declare -A ARTIFACT_PREFIX=(
  [noskills_new]=flash_noskills_new
  [all_new_skills_avoids_global]=flash_all_new_skills_avoids_global
  [all_new_skills_no_avoids_global]=flash_all_new_skills_no_avoids_global
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

export PC2_FORCE_WALLTIME="${PC2_BASELINE_3WAY_WALLTIME:-12:00:00}"
PY="${C2HLS_PYTHON:-python3}"

echo "baseline 3-way flash test stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME}"
echo "mode: parallel (3 independent sessions, each with own GPU + Vitis job)"
echo "skills: hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills).json (90 skills)"
echo "validation: csim + csynth only (C2HLS_RUN_COSIM=0)"
echo "variants: ${VARIANT_KEYS[*]}"

for key in "${VARIANT_KEYS[@]}"; do
  echo "  dry-run ${key}..."
  C2HLS_FLASH_NEW_SKILLS_STAMP="${STAMP}" \
    "${PY}" scripts/pc2/run_flash_new_skills_batch.py --pc2 --variant "${key}" --dry-run --stamp "${STAMP}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok — no sessions started"
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

# Stop any existing session state before restart (even if watch already exited).
for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    echo "stopping existing session ${sid}..."
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  else
    echo "resetting session ${sid} (no active watch)..."
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" 2>/dev/null || true
  fi
done

for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  stamp_var="${STAMP_ENV[$key]}"
  worker_cmd="${stamp_var}=${STAMP} python3 scripts/pc2/run_flash_new_skills_batch.py --pc2 --variant ${key}"
  echo "starting session ${sid} (parallel)..."
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "All 3 sessions submitted in parallel (stamp=${STAMP}). Monitor:"
for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  echo "  tail -f artifacts/pc2/sessions/${sid}/watch.log"
done
echo ""
echo "Artifacts:"
for key in "${VARIANT_KEYS[@]}"; do
  echo "  artifacts/pc2/${ARTIFACT_PREFIX[$key]}_${STAMP}/"
done
