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
AGENT_JSONL="${C2HLS_AGENT_JSONL:-artifacts/requested_agentic_hwemu.jsonl}"
LOG="${C2HLS_E2E_QUEUE_LOG:-artifacts/requested_e2e_queue.log}"

{
  echo "[$(date -Is)] waiting for direct hw_emu matrix to finish"
  while pgrep -f "run_requested_hwemu_matrix.py" >/dev/null 2>&1; do
    sleep 300
  done

  echo "[$(date -Is)] direct matrix process absent; validating ${DIRECT_JSONL}"
  "${PYTHON}" export_schema_jsonl.py --validate-jsonl "${DIRECT_JSONL}"
  "${PYTHON}" compare_jsonl_to_references.py "${DIRECT_JSONL}" --output artifacts/requested_hwemu_matrix_delta.md

  echo "[$(date -Is)] launching requested agentic multistep + final hw_emu smoke"
  # shellcheck disable=SC1091
  source scripts/setup_emu_env.sh >/dev/null 2>&1
  export C2HLS_HW_EMU_FINAL=1
  "${PYTHON}" "${SITE_FLAG[@]}" run_requested_agentic_hwemu_smoke.py

  echo "[$(date -Is)] validating agentic JSONL ${AGENT_JSONL}"
  "${PYTHON}" export_schema_jsonl.py --validate-jsonl "${AGENT_JSONL}"
  "${PYTHON}" compare_jsonl_to_references.py "${AGENT_JSONL}" --output artifacts/requested_agentic_hwemu_delta.md
  echo "[$(date -Is)] requested e2e queue complete"
} >> "${LOG}" 2>&1
