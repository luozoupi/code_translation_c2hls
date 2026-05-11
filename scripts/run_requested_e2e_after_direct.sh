#!/usr/bin/env bash
set -euo pipefail

cd /home/luo00466/code_translation-c2hls

DIRECT_JSONL="${C2HLS_HWEMU_MATRIX_JSONL:-artifacts/requested_hwemu_matrix.jsonl}"
AGENT_JSONL="${C2HLS_AGENT_JSONL:-artifacts/requested_agentic_hwemu.jsonl}"
LOG="${C2HLS_E2E_QUEUE_LOG:-artifacts/requested_e2e_queue.log}"

{
  echo "[$(date -Is)] waiting for direct hw_emu matrix to finish"
  while pgrep -f "run_requested_hwemu_matrix.py" >/dev/null 2>&1; do
    sleep 300
  done

  echo "[$(date -Is)] direct matrix process absent; validating ${DIRECT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python export_schema_jsonl.py --validate-jsonl "${DIRECT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python compare_jsonl_to_references.py "${DIRECT_JSONL}" --output artifacts/requested_hwemu_matrix_delta.md

  echo "[$(date -Is)] launching requested agentic multistep + final hw_emu smoke"
  source scripts/setup_emu_env.sh >/dev/null 2>&1
  export C2HLS_HW_EMU_FINAL=1
  export C2HLS_VITIS_SETTINGS=/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
  export C2HLS_VITIS_VERSION=2023.2
  export C2HLS_PART=xcu280-fsvh2892-2L-e
  export C2HLS_CLOCK_NS=3.33
  export C2HLS_FLOW_TARGET=vitis
  export C2HLS_CLAUDE_KEY_FILE=/home/luo00466/claude-api-key.txt
  /home/luo00466/.conda/envs/py310_2/bin/python run_requested_agentic_hwemu_smoke.py

  echo "[$(date -Is)] validating agentic JSONL ${AGENT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python export_schema_jsonl.py --validate-jsonl "${AGENT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python compare_jsonl_to_references.py "${AGENT_JSONL}" --output artifacts/requested_agentic_hwemu_delta.md
  echo "[$(date -Is)] requested e2e queue complete"
} >> "${LOG}" 2>&1
