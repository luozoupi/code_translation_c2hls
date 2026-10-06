#!/usr/bin/env bash
# Post-DSE stream/I/O step: LLM rewrite into PE modules + hls::stream DATAFLOW.
#
# Usage:
#   ./scripts/pc2/start_post_flash_stream.sh --show-prompts
#   ./scripts/pc2/start_post_flash_stream.sh --dry-run --matrix-root artifacts/pc2/...
#   ./scripts/pc2/start_post_flash_stream.sh --submit --no-borrow-gpu \
#       --matrix-root artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f \
#       --benches autosa_mm
#
# Enable auto-chain during flash (after DSE):
#   export C2HLS_POST_FLASH_STREAM=1
#   export C2HLS_STREAM_CHAIN_FLASH=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-python3}"
if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python3"
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-}"
CELL_DIR="${C2HLS_POST_FLASH_CELL_DIR:-}"
BENCHES="${C2HLS_POST_FLASH_BENCHES:-}"
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
    --cell-dir) CELL_DIR="$2"; shift 2 ;;
    --benches) BENCHES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --show-prompts) SHOW_PROMPTS=1; shift ;;
    --borrow-gpu) BORROW_GPU=1; shift ;;
    --no-borrow-gpu) BORROW_GPU=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "${SHOW_PROMPTS}" -eq 1 ]]; then
  exec "${PY}" scripts/pc2/run_post_flash_stream.py --show-prompts
fi

ARGS=(--pc2)
[[ -n "${MATRIX_ROOT}" ]] && ARGS+=(--matrix-root "${MATRIX_ROOT}")
[[ -n "${CELL_DIR}" ]] && ARGS+=(--cell-dir "${CELL_DIR}")
[[ -n "${BENCHES}" ]] && ARGS+=(--benches "${BENCHES}")
[[ -n "${C2HLS_MODEL:-}" ]] && ARGS+=(--model "${C2HLS_MODEL}")
[[ "${DRY}" -eq 1 ]] && ARGS+=(--dry-run)
[[ "${FORCE}" -eq 1 ]] && ARGS+=(--force)

export C2HLS_POST_FLASH_STREAM=1
export C2HLS_STREAM_CHAIN_FLASH="${C2HLS_STREAM_CHAIN_FLASH:-1}"

if [[ "${SUBMIT}" -eq 1 ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SESSION_ID="post_flash_stream_${STAMP}"
  WORKER_CMD="${PY} scripts/pc2/run_post_flash_stream.py ${ARGS[*]}"
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
  exec "${PY}" scripts/pc2/run_post_flash_stream.py "${ARGS[@]}"
fi
