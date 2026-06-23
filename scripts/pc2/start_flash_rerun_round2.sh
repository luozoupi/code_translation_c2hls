#!/usr/bin/env bash
# Re-run legacy (4) + new (6) flash matrices with a fresh artifact stamp.
# Does NOT overwrite prior results: artifacts live under flash_*_<STAMP>/.
#
# Watch interval: 60s (never 30min). Compute walltime: 12h via PC2_FORCE_WALLTIME.
#
# Usage:
#   ./scripts/pc2/start_flash_rerun_round2.sh --dry-run
#   ./scripts/pc2/start_flash_rerun_round2.sh --auto-stop-on-complete
#   ./scripts/pc2/start_flash_rerun_round2.sh --stamp 20260622_120000 --legacy-only
#   ./scripts/pc2/start_flash_rerun_round2.sh --new-only --auto-stop-on-complete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
LEGACY=1
NEW=1
STAMP="${C2HLS_FLASH_RERUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --legacy-only) NEW=0; shift ;;
    --new-only) LEGACY=0; shift ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# Never use 30-minute watch polling for supervision.
export PC2_WATCH_INTERVAL_SEC=60
export PC2_FORCE_WALLTIME="${PC2_FLASH_RERUN_WALLTIME:-12:00:00}"

extra=()
if [[ "${AUTO_STOP}" -eq 1 ]]; then
  extra+=(--auto-stop-on-complete)
fi

echo "flash rerun round stamp=${STAMP}"
echo "  watch_interval=${PC2_WATCH_INTERVAL_SEC}s  walltime=${PC2_FORCE_WALLTIME}"
echo "  legacy=${LEGACY}  new=${NEW}  auto_stop=${AUTO_STOP}"
echo "  prior artifacts (unchanged): flash_*_20260620_*  flash_*_20260621_020847"
echo ""

if [[ "${LEGACY}" -eq 1 ]]; then
  echo "=== legacy dual (noskills + bn 2+2) ==="
  "${SCRIPT_DIR}/start_dual_flash_sessions.sh" --stamp "${STAMP}" "${extra[@]}" $([[ "${DRY_RUN}" -eq 1 ]] && echo --dry-run)
  echo ""
  echo "=== legacy global (all+avoids + no avoids) ==="
  "${SCRIPT_DIR}/start_dual_global_skills_flash_sessions.sh" --stamp "${STAMP}" "${extra[@]}" $([[ "${DRY_RUN}" -eq 1 ]] && echo --dry-run)
  echo ""
fi

if [[ "${NEW}" -eq 1 ]]; then
  echo "=== new skills matrix (6 variants) ==="
  "${SCRIPT_DIR}/start_new_skills_flash_sessions.sh" --stamp "${STAMP}" "${extra[@]}" $([[ "${DRY_RUN}" -eq 1 ]] && echo --dry-run)
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo ""
  echo "dry-run ok — no Slurm jobs submitted"
  exit 0
fi

echo ""
echo "All requested sessions submitted (stamp=${STAMP})."
echo "Artifacts: artifacts/pc2/flash_*_${STAMP}/"
echo "Monitor: ls artifacts/pc2/sessions/flash_*/watch.log"
