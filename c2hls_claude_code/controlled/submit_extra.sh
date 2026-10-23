#!/bin/bash
# Controlled no-skills and DSP-redo variants, 3 replicates per model, interleaved, same settings as refv2/win.
set -euo pipefail
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
R=runs_controlled
SEED=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls/artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/hls_baseline.cpp
INC=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include
COMMON="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,OUT_ROOT=$PWD/$R,C2HLS_COMPILE_INCLUDE_PATHS=$INC,SEED_CODE=$SEED"
for r in 1 2 3; do for v in noskills dspredo; do
  case $v in noskills) V="FLAVOR=noskills";; dspredo) V="FLAVOR=skills,DSP_REDO=1";; esac
  for m in cc ds; do
    if [ $m = cc ]; then MM="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat shim_url.txt)"; else MM="MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$(cat controlled/ds_proxy_url.txt)"; fi
    t=ctl_${m}_${v}_r$r
    j=$(sbatch --parsable --job-name=$t --time=06:00:00 --mem=128G --export=$COMMON,$V,$MM,TAG=$t sbatch_flash_cc.sh 2>/dev/null)
    echo "$t $j" | tee -a $R/jobs.txt
  done
done; done
