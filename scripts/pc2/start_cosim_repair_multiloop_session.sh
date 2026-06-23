#!/usr/bin/env bash
# Start 3 parallel PC2 cosim-repair sessions with multi-loop LLM repair (10 loops).
#
# Separate from start_cosim_repair_session.sh (single-shot). Uses distinct
# session ids and artifact root so prior single-loop runs are not overwritten.
#
# Usage:
#   ./scripts/pc2/start_cosim_repair_multiloop_session.sh --dry-run
#   ./scripts/pc2/start_cosim_repair_multiloop_session.sh --auto-stop-on-complete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
MAX_LOOPS="${C2HLS_COSIM_REPAIR_MAX_LOOPS:-10}"
STAMP="${C2HLS_FLASH_COSIM_REPAIR_MULTILOOP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
COSIM_RUN="${C2HLS_FLASH_COSIM_RUN_ROOT:-${C2HLS_ROOT}/artifacts/pc2/flash_cosim/20260622_110920}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --cosim-run) shift; COSIM_RUN="$1"; shift ;;
    --max-loops) shift; MAX_LOOPS="$1"; shift ;;
    -h|--help)
      cat <<EOF
usage: start_cosim_repair_multiloop_session.sh [--dry-run] [--auto-stop-on-complete]
       [--stamp STAMP] [--cosim-run PATH] [--max-loops N]

Submits 3 independent PC2 sessions (3 GPU + 3 compute jobs total).
Each failing kernel gets up to N LLM diagnose+repair loops (default: 10).
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_FORCE_WALLTIME="${PC2_COSIM_REPAIR_MULTILOOP_WALLTIME:-48:00:00}"
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python3}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi
REPAIR_ROOT="${C2HLS_ROOT}/artifacts/pc2/flash_cosim_repair_multiloop/${STAMP}"
export C2HLS_FLASH_COSIM_REPAIR_ROOT="${C2HLS_ROOT}/artifacts/pc2/flash_cosim_repair_multiloop"
export C2HLS_FLASH_COSIM_REPAIR_STAMP="${STAMP}"
export C2HLS_COSIM_REPAIR_MAX_LOOPS="${MAX_LOOPS}"

declare -A SESSION_ID=(
  [all_avoids_new]=cosim_repair_ml10_all_avoids_new
  [no_avoids_old]=cosim_repair_ml10_no_avoids_old
  [noskills_old]=cosim_repair_ml10_noskills_old
)

declare -A SESSION_LABEL=(
  [all_avoids_new]="All+avoids (new) [${MAX_LOOPS}-loop]"
  [no_avoids_old]="No avoids (old) [${MAX_LOOPS}-loop]"
  [noskills_old]="Noskills (old) [${MAX_LOOPS}-loop]"
)

echo "cosim repair MULTI-LOOP sessions stamp=${STAMP} max_loops=${MAX_LOOPS} walltime=${PC2_FORCE_WALLTIME}"
echo "cosim source: ${COSIM_RUN}"
echo "repair out:   ${REPAIR_ROOT}"
echo ""

MANIFEST_JSON="$("${PY}" scripts/pc2/build_flash_cosim_repair_manifest.py \
  --stamp "${STAMP}" --cosim-run "${COSIM_RUN}" 2>/dev/null || true)"

for key in all_avoids_new no_avoids_old noskills_old; do
  echo "  ${SESSION_LABEL[$key]} → session ${SESSION_ID[$key]}"
  if [[ -n "${MANIFEST_JSON}" ]]; then
    "${PY}" -c "
import json, sys
d = json.loads(sys.argv[1])
s = d['sessions'][sys.argv[2]]
print(f'    failures={s[\"failures\"]} repair_variant={s[\"repair_variant\"]} max_loops={sys.argv[3]}')
" "${MANIFEST_JSON}" "${key}" "${MAX_LOOPS}" 2>/dev/null || true
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo ""
  echo "dry-run: no Slurm jobs submitted (${MANIFEST_JSON:+manifest written})"
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

for key in all_avoids_new no_avoids_old noskills_old; do
  sid="${SESSION_ID[$key]}"
  export PC2_SESSION_ID="${sid}"
  _pc2_configure_session_paths
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  fi
done
unset PC2_SESSION_ID

mkdir -p "${REPAIR_ROOT}"

for key in all_avoids_new no_avoids_old noskills_old; do
  sid="${SESSION_ID[$key]}"
  worker_cmd="C2HLS_FLASH_COSIM_REPAIR_ROOT=${C2HLS_ROOT}/artifacts/pc2/flash_cosim_repair_multiloop C2HLS_FLASH_COSIM_REPAIR_STAMP=${STAMP} C2HLS_FLASH_COSIM_RUN_ROOT=${COSIM_RUN} C2HLS_RUN_COSIM=1 C2HLS_COSIM_REPAIR_MAX_LOOPS=${MAX_LOOPS} ${PY} scripts/pc2/run_flash_cosim_repair_batch.py --pc2 --session ${key} --stamp ${STAMP} --cosim-run ${COSIM_RUN} --max-loops ${MAX_LOOPS}"
  echo "starting session ${sid} (${SESSION_LABEL[$key]})..."
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "3 multi-loop cosim-repair sessions submitted (stamp=${STAMP}, max_loops=${MAX_LOOPS}). Monitor:"
for key in all_avoids_new no_avoids_old noskills_old; do
  sid="${SESSION_ID[$key]}"
  echo "  tail -f artifacts/pc2/sessions/${sid}/watch.log"
done
echo "  squeue -u \$USER"
echo "  repair artifacts: ${REPAIR_ROOT}/"
