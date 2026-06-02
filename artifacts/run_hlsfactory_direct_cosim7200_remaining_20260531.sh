#!/usr/bin/env bash
set -euo pipefail

cd /home/luo00466/code_translation-c2hls

STAMP=hlsfactory_direct_cosim7200_remaining_20260531
LOG=/home/luo00466/code_translation-c2hls/artifacts/${STAMP}.log
MERGE_LOG=/home/luo00466/code_translation-c2hls/artifacts/${STAMP}.merge.log

export C2HLS_VITIS_SETTINGS=/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh
export C2HLS_VITIS_VERSION=2023.2
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_FLOW_TARGET=vitis
export C2HLS_TMP_ROOT=/mnt/data/luo00466/tmp
export C2HLS_VITIS_USER_HOME=/mnt/data/luo00466/tmp/vitis_user_home_hlsfactory_direct_cosim7200_20260531
export C2HLS_SYNTH_TIMEOUT=1200
export C2HLS_CSIM_TIMEOUT=300
export C2HLS_COSIM_TIMEOUT=7200
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_HLSFACTORY_DIRECT_STAMP="${STAMP}"
export C2HLS_HLSFACTORY_DIRECT_CSIM=0
export C2HLS_HLSFACTORY_DIRECT_COSIM=1
export C2HLS_HLSFACTORY_BENCHES=hlsfactory_2mm,hlsfactory_correlation,hlsfactory_covariance,hlsfactory_fdtd_2d,hlsfactory_floyd_warshall,hlsfactory_gramschmidt,hlsfactory_heat_3d,hlsfactory_jacobi_2d,hlsfactory_lu,hlsfactory_seidel_2d,hlsfactory_symm,hlsfactory_syr2k,hlsfactory_syrk
export C2HLS_HLSFACTORY_DIRECT_JSONL=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_reference_${STAMP}.jsonl
export C2HLS_HLSFACTORY_DIRECT_SUMMARY=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_reference_${STAMP}.summary.json
export C2HLS_HLSFACTORY_DIRECT_MD=/home/luo00466/code_translation-c2hls/artifacts/hlsfactory_direct_reference_${STAMP}.md

python3 run_hlsfactory_direct_reference.py 2>&1 | tee "${LOG}"
python3 artifacts/merge_hlsfactory_cosim7200_20260531.py 2>&1 | tee "${MERGE_LOG}"
