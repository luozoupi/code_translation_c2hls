#!/usr/bin/env bash
# Start two independent PC2 sessions in parallel:
#   flash_noskills  — all hlsfactory_* kernels, skills off
#   flash_skills    — all hlsfactory_* kernels, base skills on
#
# Each session gets its own GPU job, compute job, watch.log, and llm_endpoint.json
# under artifacts/pc2/sessions/<session-id>/.
#
# Usage (login node, repo root):
#   ./scripts/pc2/start_dual_flash_sessions.sh
#   ./scripts/pc2/start_dual_flash_sessions.sh --dry-run
#
# Options:
#   --dry-run              print plan only
#   --auto-stop-on-complete  stop each session 120s after its worker succeeds
#   --stamp STAMP          shared artifact stamp (default: now)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
STAMP="${C2HLS_DUAL_FLASH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --auto-stop-on-complete)
      AUTO_STOP=1
      shift
      ;;
    --stamp)
      shift
      STAMP="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# 28× hlsfactory benches × ~2 LLM calls + csynth each; allow a long walltime.
export PC2_FORCE_WALLTIME="${PC2_DUAL_FLASH_WALLTIME:-12:00:00}"

NOSKILLS_CMD="C2HLS_FLASH_NOSKILLS_STAMP=${STAMP} python3 scripts/pc2/run_flash_noskills_batch.py --pc2 --all-hlsfactory"
SKILLS_CMD="C2HLS_FLASH_SKILLS_STAMP=${STAMP} python3 scripts/pc2/run_flash_skills_batch.py --pc2 --all-hlsfactory"

BENCH_COUNT="$(
  "${C2HLS_PYTHON:-python3}" -c "
from pathlib import Path
import json
root = Path('${C2HLS_ROOT}') / 'benchmarks'
print(sum(1 for p in root.glob('hlsfactory_*/metadata.json')))
"
)"

echo "dual flash sessions stamp=${STAMP} benches=${BENCH_COUNT} walltime=${PC2_FORCE_WALLTIME} watch_interval=${PC2_WATCH_INTERVAL_SEC:-60}s"
echo "  noskills session: artifacts/pc2/sessions/flash_noskills/watch.log"
echo "  skills session:   artifacts/pc2/sessions/flash_skills/watch.log"
echo "  noskills out:     artifacts/pc2/flash_noskills_${STAMP}/"
echo "  skills out:       artifacts/pc2/flash_skills_${STAMP}/"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: no jobs submitted"
  exit 0
fi

start_one() {
  local session_id="$1"
  local worker_cmd="$2"
  local extra_args=()
  if [[ "${AUTO_STOP}" -eq 1 ]]; then
    extra_args+=(--auto-stop-on-complete)
  fi
  "${SCRIPT_DIR}/start_session.sh" \
    --session-id "${session_id}" \
    --worker-cmd "${worker_cmd}" \
    "${extra_args[@]}"
}

# Clean stale watches for these session ids only (do not touch legacy default session).
for sid in flash_noskills flash_skills; do
  export PC2_SESSION_ID="${sid}"
  _pc2_configure_session_paths
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  fi
done
unset PC2_SESSION_ID

start_one flash_noskills "${NOSKILLS_CMD}"
start_one flash_skills "${SKILLS_CMD}"

echo ""
echo "Both sessions submitted. Monitor separately:"
echo "  tail -f artifacts/pc2/sessions/flash_noskills/watch.log"
echo "  tail -f artifacts/pc2/sessions/flash_skills/watch.log"
echo "  squeue -u \$USER"
