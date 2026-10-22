#!/bin/bash
# DSP 50-65% sets (floor 4512, cap 5865 of 9024) for every test type whose results exceeded 60% DSP:
#   s5065  : flash + skills + DSP floor/cap (stock gates, no refinement)          <- from DSP redo
#   r5065  : refined (original wording) + DSP window 4512:5865                     <- from refined+floor and 70-80% window
#   a5065  : r5065 in Claude agent mode (tools + hls_check, fixed bench)           <- from the agentic ladder (L3 > 60%)
# Fixed bench (C = A*B^T, C poisoned), controlled LLM layer, 128 GB, wall time via study/jobs/ledger.py (1h -> 3h -> 6h).
set -euo pipefail
cd /scratch/asa582/workspaces/c2hls_claude_code
FB=$PWD/study/bench_fixed/n1024; OUT=$PWD/study/dsp5065/runs
INC=/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include
COMMON="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,OUT_ROOT=$OUT,C2HLS_COMPILE_INCLUDE_PATHS=$INC,SEED_CODE=$FB/autosa_mm/hls_baseline.cpp,BENCH_ROOT=$FB"
CC="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat shim_url.txt)"
DS="MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$(cat controlled/ds_proxy_url.txt)"
AG="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat study/shim_agent_study_url.txt),CC_LLM_TIMEOUT=50400"
for r in 1 2 3; do
  for v in s5065 r5065 a5065; do
    case $v in
      s5065) V="FLAVOR=skills,C2HLS_FLASH_MIN_DSP=4512,C2HLS_FLASH_MAX_DSP=5865"; MODELS="cc ds";;
      r5065) V="FLAVOR=skills,REFINED=1,DSP_WINDOW=4512:5865"; MODELS="cc ds";;
      a5065) V="FLAVOR=skills,REFINED=1,DSP_WINDOW=4512:5865"; MODELS="ag";;
    esac
    for m in $MODELS; do
      case $m in cc) M=$CC;; ds) M=$DS;; ag) M=$AG;; esac
      t=st_${v}_${m}_r$r
      python3 study/jobs/ledger.py submit $t --move-glob "$OUT/*_${t}_*" -- \
        --job-name=$t --mem=128G --export=$COMMON,$V,$M,TAG=$t sbatch_flash_cc.sh
    done
  done
done
