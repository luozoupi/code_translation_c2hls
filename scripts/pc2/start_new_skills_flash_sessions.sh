#!/usr/bin/env bash
# Start PC2 flash sessions for the NEW skills matrix only.
# Does NOT touch legacy sessions: flash_noskills, flash_skills,
# flash_all_skills_avoids_global, flash_all_skills_no_avoids_global.
#
# Skills file: skills_ii_target_miss_solutions_added(73skills).json or (90skills).json per variant.
#
# Usage:
#   ./scripts/pc2/start_new_skills_flash_sessions.sh --dry-run
#   ./scripts/pc2/start_new_skills_flash_sessions.sh --auto-stop-on-complete
#   ./scripts/pc2/start_new_skills_flash_sessions.sh --variants bn_skills_new_2_2,all_new_skills_no_avoids_global
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
STAMP="${C2HLS_FLASH_NEW_SKILLS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
VARIANTS_CSV="${C2HLS_FLASH_NEW_VARIANTS:-noskills_new,bn_skills_new_2_2,bn_skills_new_4_2,bn_skills_new_6_2,all_new_skills_avoids_global,all_new_skills_no_avoids_global}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --variants) shift; VARIANTS_CSV="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Watch uses PC2_WATCH_INTERVAL_SEC from common.sh (default 60s). Do not lengthen
# here — long polling is for manual/agent tailing of watch.log, not supervision.

export PC2_FORCE_WALLTIME="${PC2_NEW_SKILLS_WALLTIME:-12:00:00}"
PY="${C2HLS_PYTHON:-python3}"

IFS=',' read -r -a VARIANT_KEYS <<< "${VARIANTS_CSV}"

echo "new-skills flash matrix stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME} watch_interval=${PC2_WATCH_INTERVAL_SEC:-60}s"
echo "skills: hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(73skills|90skills).json (per variant)"
echo "variants: ${VARIANTS_CSV}"
echo "(legacy flash_* sessions are NOT modified)"

for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  echo "  dry-run ${key}..."
  C2HLS_FLASH_NEW_SKILLS_STAMP="${STAMP}" \
    "${PY}" scripts/pc2/run_flash_new_skills_batch.py --pc2 --variant "${key}" --dry-run --stamp "${STAMP}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: all variants ok, no Slurm jobs submitted"
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

declare -A STAMP_ENV=(
  [noskills_new]=C2HLS_FLASH_NOSKILLS_NEW_STAMP
  [bn_skills_new_2_2]=C2HLS_FLASH_BN_SKILLS_NEW_2_2_STAMP
  [bn_skills_new_4_2]=C2HLS_FLASH_BN_SKILLS_NEW_4_2_STAMP
  [bn_skills_new_6_2]=C2HLS_FLASH_BN_SKILLS_NEW_6_2_STAMP
  [all_new_skills_avoids_global]=C2HLS_FLASH_ALL_NEW_SKILLS_AVOIDS_GLOBAL_STAMP
  [all_new_skills_no_avoids_global]=C2HLS_FLASH_ALL_NEW_SKILLS_NO_AVOIDS_GLOBAL_STAMP
)

declare -A SESSION_ID=(
  [noskills_new]=flash_noskills_new
  [bn_skills_new_2_2]=flash_bn_skills_new_2_2
  [bn_skills_new_4_2]=flash_bn_skills_new_4_2
  [bn_skills_new_6_2]=flash_bn_skills_new_6_2
  [all_new_skills_avoids_global]=flash_all_new_skills_avoids_global
  [all_new_skills_no_avoids_global]=flash_all_new_skills_no_avoids_global
)

for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  sid="${SESSION_ID[$key]:-}"
  stamp_var="${STAMP_ENV[$key]:-}"
  if [[ -z "${sid}" || -z "${stamp_var}" ]]; then
    echo "unknown variant: ${key}" >&2
    exit 2
  fi
  export PC2_SESSION_ID="${sid}"
  _pc2_configure_session_paths
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  fi
done
unset PC2_SESSION_ID

for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  sid="${SESSION_ID[$key]}"
  stamp_var="${STAMP_ENV[$key]}"
  worker_cmd="${stamp_var}=${STAMP} python3 scripts/pc2/run_flash_new_skills_batch.py --pc2 --variant ${key}"
  echo "starting session ${sid}..."
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "New-skills sessions submitted (stamp=${STAMP}). Monitor:"
for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  sid="${SESSION_ID[$key]}"
  echo "  tail -f artifacts/pc2/sessions/${sid}/watch.log"
done
