#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/bootstrap_site.sh" "$@"
source "$(dirname "$0")/source_local_env.sh"
cd "${C2HLS_ROOT}"

if [[ "${C2HLS_SITE:-team}" == "pc2" ]]; then
  PYTHON="${C2HLS_PYTHON:-python3}"
  SITE_FLAG=(--pc2)
else
  PYTHON="${C2HLS_PYTHON:-/home/luo00466/.conda/envs/py310_2/bin/python}"
  SITE_FLAG=()
fi

DIRECT_JSONL="${C2HLS_HWEMU_MATRIX_JSONL:-artifacts/requested_hwemu_matrix.jsonl}"
RERUN_JSONL="${C2HLS_HWEMU_RERUN_JSONL:-artifacts/requested_hwemu_mismatch_rerun.jsonl}"
AGENT_JSONL="${C2HLS_AGENT_JSONL:-artifacts/requested_agentic_hwemu.jsonl}"
LOG="${C2HLS_E2E_RERUN_QUEUE_LOG:-artifacts/requested_direct_rerun_then_e2e.log}"

{
  echo "[$(date -Is)] waiting for direct hw_emu matrix to finish"
  while pgrep -f "run_requested_hwemu_matrix.py" >/dev/null 2>&1; do
    sleep 300
  done

  echo "[$(date -Is)] direct matrix process absent; validating ${DIRECT_JSONL}"
  "${PYTHON}" export_schema_jsonl.py --validate-jsonl "${DIRECT_JSONL}"
  "${PYTHON}" compare_jsonl_to_references.py "${DIRECT_JSONL}" --output artifacts/requested_hwemu_matrix_delta.md

  echo "[$(date -Is)] rerunning direct hw_emu status mismatches into ${RERUN_JSONL}"
  # shellcheck disable=SC1091
  source scripts/setup_emu_env.sh >/dev/null 2>&1
  export C2HLS_HWEMU_RERUN_JSONL="${RERUN_JSONL}"
  export C2HLS_FORCE_RERUN=1
  "${PYTHON}" "${SITE_FLAG[@]}" run_requested_hwemu_mismatch_rerun.py
  unset C2HLS_FORCE_RERUN

  echo "[$(date -Is)] validating mismatch rerun ${RERUN_JSONL}"
  "${PYTHON}" export_schema_jsonl.py --validate-jsonl "${RERUN_JSONL}"
  "${PYTHON}" compare_jsonl_to_references.py "${RERUN_JSONL}" --output artifacts/requested_hwemu_mismatch_rerun_delta.md
  "${PYTHON}" build_hwemu_reference_candidate.py

  echo "[$(date -Is)] launching requested agentic multistep + final hw_emu smoke"
  export C2HLS_HW_EMU_FINAL=1
  "${PYTHON}" "${SITE_FLAG[@]}" run_requested_agentic_hwemu_smoke.py

  echo "[$(date -Is)] validating agentic JSONL ${AGENT_JSONL}"
  "${PYTHON}" export_schema_jsonl.py --validate-jsonl "${AGENT_JSONL}"
  "${PYTHON}" compare_jsonl_to_references.py "${AGENT_JSONL}" --output artifacts/requested_agentic_hwemu_delta.md
  echo "[$(date -Is)] direct-rerun-then-e2e queue complete"
} >> "${LOG}" 2>&1
