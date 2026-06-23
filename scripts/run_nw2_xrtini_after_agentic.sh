#!/usr/bin/env bash
set -u

# shellcheck disable=SC1091
source "$(dirname "$0")/bootstrap_site.sh" "$@"
source "$(dirname "$0")/source_local_env.sh"

ROOT="${C2HLS_ROOT}"
if [[ "${C2HLS_SITE:-team}" == "pc2" ]]; then
  PYTHON="${C2HLS_PYTHON:-python3}"
  SITE_FLAG=(--pc2)
else
  PYTHON="${C2HLS_PYTHON:-/home/luo00466/.conda/envs/py310_2/bin/python}"
  SITE_FLAG=()
fi
WAIT_PID="${C2HLS_WAIT_PID:-}"
WAIT_SECONDS="${C2HLS_XRT_WAIT_SECONDS:-300}"
STAMP="${C2HLS_XRTINI_STAMP:-$(date +%Y%m%d_%H%M%S)}"

LOG="${C2HLS_NW2_XRT_QUEUE_LOG:-$ROOT/artifacts/nw2_pipeline_xrtini_after_agentic_${STAMP}.log}"
JSONL="${C2HLS_NW2_XRT_JSONL:-$ROOT/artifacts/nw2_pipeline_hwemu_xrt_debug_off_after_agentic_${STAMP}.jsonl}"
DELTA="${C2HLS_NW2_XRT_DELTA:-${JSONL%.jsonl}_delta.md}"

mkdir -p "$ROOT/artifacts"

{
  echo "[$(date --iso-8601=seconds)] queued nw_2_pipeline xrt.ini hw_emu experiment"
  echo "root=$ROOT"
  echo "wait_pid=$WAIT_PID"
  echo "jsonl=$JSONL"
  echo "delta=$DELTA"

  if [[ -n "$WAIT_PID" ]]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      echo "[$(date --iso-8601=seconds)] waiting for agentic smoke pid $WAIT_PID"
      sleep "$WAIT_SECONDS"
    done
  else
    while pgrep -f "run_requested_agentic_hwemu_smoke.py" >/dev/null; do
      echo "[$(date --iso-8601=seconds)] waiting for agentic smoke process"
      sleep "$WAIT_SECONDS"
    done
  fi

  cd "$ROOT" || exit 2
  echo "[$(date --iso-8601=seconds)] starting xrt.ini experiment"
  C2HLS_NW2_XRT_JSONL="$JSONL" "${PYTHON}" "${SITE_FLAG[@]}" run_nw2_xrtini_hwemu_experiment.py
  run_rc=$?
  echo "[$(date --iso-8601=seconds)] experiment exit=$run_rc"

  if [[ "$run_rc" -ne 0 ]]; then
    exit "$run_rc"
  fi

  "${PYTHON}" export_schema_jsonl.py --validate-jsonl "$JSONL"
  validate_rc=$?
  echo "[$(date --iso-8601=seconds)] schema validation exit=$validate_rc"

  "${PYTHON}" compare_jsonl_to_references.py "$JSONL" --output "$DELTA"
  compare_rc=$?
  echo "[$(date --iso-8601=seconds)] reference comparison exit=$compare_rc"

  if [[ "$validate_rc" -ne 0 ]]; then
    exit "$validate_rc"
  fi
  exit "$compare_rc"
} >> "$LOG" 2>&1
