#!/usr/bin/env bash
# Start two independent PC2 flash sessions with global skill injection:
#   flash_all_skills_avoids_global      — full library + avoid rules
#   flash_all_skills_no_avoids_global   — full library, no avoids
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
STAMP="${C2HLS_DUAL_GLOBAL_SKILLS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_FORCE_WALLTIME="${PC2_DUAL_FLASH_WALLTIME:-12:00:00}"

AVOIDS_CMD="C2HLS_FLASH_ALL_SKILLS_AVOIDS_GLOBAL_STAMP=${STAMP} python3 scripts/pc2/run_flash_all_skills_avoids_global_batch.py --pc2"
NO_AVOIDS_CMD="C2HLS_FLASH_ALL_SKILLS_NO_AVOIDS_GLOBAL_STAMP=${STAMP} python3 scripts/pc2/run_flash_all_skills_no_avoids_global_batch.py --pc2"

echo "dual global-skills flash stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME} watch_interval=${PC2_WATCH_INTERVAL_SEC:-60}s"
echo "  avoids:    artifacts/pc2/sessions/flash_all_skills_avoids_global/watch.log"
echo "  no_avoids: artifacts/pc2/sessions/flash_all_skills_no_avoids_global/watch.log"

"${C2HLS_PYTHON:-python3}" scripts/pc2/run_flash_all_skills_avoids_global_batch.py --pc2 --dry-run --stamp "${STAMP}"
"${C2HLS_PYTHON:-python3}" scripts/pc2/run_flash_all_skills_no_avoids_global_batch.py --pc2 --dry-run --stamp "${STAMP}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: batch scripts ok, no Slurm jobs submitted"
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

for sid in flash_all_skills_avoids_global flash_all_skills_no_avoids_global; do
  export PC2_SESSION_ID="${sid}"
  _pc2_configure_session_paths
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  fi
done
unset PC2_SESSION_ID

start_one flash_all_skills_avoids_global "${AVOIDS_CMD}"
start_one flash_all_skills_no_avoids_global "${NO_AVOIDS_CMD}"

echo ""
echo "Sessions submitted. Monitor:"
echo "  tail -f artifacts/pc2/sessions/flash_all_skills_avoids_global/watch.log"
echo "  tail -f artifacts/pc2/sessions/flash_all_skills_no_avoids_global/watch.log"
