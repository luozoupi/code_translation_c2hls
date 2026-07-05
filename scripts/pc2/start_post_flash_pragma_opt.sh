#!/usr/bin/env bash
# Post-pass HLS pragma optimization on flash-final or DATAFLOW kernels.
#
# Usage:
#   ./scripts/pc2/start_post_flash_pragma_opt.sh --show-prompts
#   ./scripts/pc2/start_post_flash_pragma_opt.sh --dry-run --source flash_final
#   ./scripts/pc2/start_post_flash_pragma_opt.sh --submit --source dataflow
#
# Enable auto-chain during flash / DATAFLOW runs:
#   export C2HLS_POST_FLASH_PRAGMA_OPT=1
#   export C2HLS_PRAGMA_OPT_CHAIN_FLASH=1      # after flash final passes
#   export C2HLS_PRAGMA_OPT_CHAIN_DATAFLOW=1   # after DATAFLOW passes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python3"
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-}"
SOURCE="${C2HLS_PRAGMA_OPT_SOURCE:-flash_final}"
SUBMIT=0
DRY=0
FORCE=0
SHOW_PROMPTS=0
BORROW_GPU=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --matrix-root) MATRIX_ROOT="$2"; shift 2 ;;
    --benches) BENCHES="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --show-prompts) SHOW_PROMPTS=1; shift ;;
    --borrow-gpu) BORROW_GPU=1; shift ;;
    --no-borrow-gpu) BORROW_GPU=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "${SHOW_PROMPTS}" -eq 1 ]]; then
  exec "${PY}" scripts/pc2/run_post_flash_pragma_opt.py --show-prompts
fi

ARGS=(--pc2 --matrix-root "${MATRIX_ROOT}" --source "${SOURCE}")
[[ -n "${BENCHES}" ]] && ARGS+=(--benches "${BENCHES}")
[[ "${DRY}" -eq 1 ]] && ARGS+=(--dry-run)
[[ "${FORCE}" -eq 1 ]] && ARGS+=(--force)

if [[ "${SUBMIT}" -eq 1 ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SESSION_ID="post_flash_pragma_opt_${STAMP}"
  WORKER_CMD="${PY} scripts/pc2/run_post_flash_pragma_opt.py ${ARGS[*]}"
  pc2_log "submitting supervised session id=${SESSION_ID}"
  pc2_log "worker: ${WORKER_CMD}"
  BORROW_ARGS=()
  if [[ "${BORROW_GPU}" -eq 1 ]]; then
    BORROW_ARGS=(--borrow-gpu)
  else
    BORROW_ARGS=(--no-borrow-gpu)
  fi
  exec "${SCRIPT_DIR}/start_session.sh" \
    --session-id "${SESSION_ID}" \
    --worker-cmd "${WORKER_CMD}" \
    "${BORROW_ARGS[@]}"
else
  exec "${PY}" scripts/pc2/run_post_flash_pragma_opt.py "${ARGS[@]}"
fi
