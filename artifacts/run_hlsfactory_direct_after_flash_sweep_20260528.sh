#!/usr/bin/env bash
set -euo pipefail

cd /home/luo00466/code_translation-c2hls

WAIT_LOG=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_after_flash_sweep_20260528.queue.log
DIRECT_LOG=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_after_flash_sweep_20260528.log
STAMP=hlsfactory_direct_after_flash_sweep_20260528

echo "[$(date -Is)] waiting for active HLSFactory agentic sweep to finish" | tee -a "${WAIT_LOG}"
while pgrep -af "[r]un_agentic_sweep.py" >/dev/null; do
  pgrep -af "[r]un_agentic_sweep.py" | tee -a "${WAIT_LOG}" >/dev/null
  sleep 60
done

echo "[$(date -Is)] sweep process absent; starting direct HLSFactory Vitis reference run" | tee -a "${WAIT_LOG}"

export C2HLS_VITIS_SETTINGS=/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
export C2HLS_VITIS_VERSION=2023.2
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_FLOW_TARGET=vitis
export C2HLS_TMP_ROOT=/home/luo00466/tmp
export C2HLS_VITIS_USER_HOME=/home/luo00466/tmp/vitis_user_home_hlsfactory_direct_after_flash_20260528
export C2HLS_SYNTH_TIMEOUT=1200
export C2HLS_CSIM_TIMEOUT=300
export C2HLS_COSIM_TIMEOUT=1800
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_HLSFACTORY_DIRECT_STAMP="${STAMP}"
export C2HLS_HLSFACTORY_DIRECT_CSIM=1
export C2HLS_HLSFACTORY_DIRECT_COSIM=1
export C2HLS_HLSFACTORY_DIRECT_JSONL=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_reference_${STAMP}.jsonl
export C2HLS_HLSFACTORY_DIRECT_SUMMARY=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_reference_${STAMP}.summary.json
export C2HLS_HLSFACTORY_DIRECT_MD=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_reference_${STAMP}.md

python3 run_hlsfactory_direct_reference.py 2>&1 | tee "${DIRECT_LOG}"

echo "[$(date -Is)] direct HLSFactory run complete" | tee -a "${WAIT_LOG}"
