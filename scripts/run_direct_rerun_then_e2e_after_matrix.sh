#!/usr/bin/env bash
set -euo pipefail

cd /home/luo00466/code_translation-c2hls

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
  /home/luo00466/.conda/envs/py310_2/bin/python export_schema_jsonl.py --validate-jsonl "${DIRECT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python compare_jsonl_to_references.py "${DIRECT_JSONL}" --output artifacts/requested_hwemu_matrix_delta.md

  echo "[$(date -Is)] rerunning direct hw_emu status mismatches into ${RERUN_JSONL}"
  source scripts/setup_emu_env.sh >/dev/null 2>&1
  export C2HLS_VITIS_SETTINGS=/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
  export C2HLS_VITIS_VERSION=2023.2
  export C2HLS_DEVICE_PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
  export C2HLS_HWEMU_RERUN_JSONL="${RERUN_JSONL}"
  export C2HLS_FORCE_RERUN=1
  /home/luo00466/.conda/envs/py310_2/bin/python run_requested_hwemu_mismatch_rerun.py
  unset C2HLS_FORCE_RERUN

  echo "[$(date -Is)] validating mismatch rerun ${RERUN_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python export_schema_jsonl.py --validate-jsonl "${RERUN_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python compare_jsonl_to_references.py "${RERUN_JSONL}" --output artifacts/requested_hwemu_mismatch_rerun_delta.md
  /home/luo00466/.conda/envs/py310_2/bin/python build_hwemu_reference_candidate.py

  echo "[$(date -Is)] launching requested agentic multistep + final hw_emu smoke"
  export C2HLS_HW_EMU_FINAL=1
  export C2HLS_PART=xcu280-fsvh2892-2L-e
  export C2HLS_CLOCK_NS=3.33
  export C2HLS_FLOW_TARGET=vitis
  export C2HLS_CLAUDE_KEY_FILE=/home/luo00466/claude-api-key.txt
  /home/luo00466/.conda/envs/py310_2/bin/python run_requested_agentic_hwemu_smoke.py

  echo "[$(date -Is)] validating agentic JSONL ${AGENT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python export_schema_jsonl.py --validate-jsonl "${AGENT_JSONL}"
  /home/luo00466/.conda/envs/py310_2/bin/python compare_jsonl_to_references.py "${AGENT_JSONL}" --output artifacts/requested_agentic_hwemu_delta.md
  echo "[$(date -Is)] direct-rerun-then-e2e queue complete"
} >> "${LOG}" 2>&1
