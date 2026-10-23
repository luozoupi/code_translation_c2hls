#!/bin/bash
# LUD (N=32) ablation with the controlled method: zero-shot, flash no-skills, flash + skills, neutral refined expertise.
# Both models, fixed LUD bench (L/U poisoned), controlled LLM layer, ledger wall time (1h -> 3h -> 6h).
# usage: submit_ablation.sh "<variants>" "<reps>"     e.g. submit_ablation.sh "zs" "1"
set -euo pipefail
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
LB=$PWD/study/lud/bench/n32; OUT=$PWD/study/lud/runs
INC=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include
COMMON="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,OUT_ROOT=$OUT,C2HLS_COMPILE_INCLUDE_PATHS=$INC,BENCH_ROOT=$LB,BENCH=lud"
CC="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat shim_url.txt)"
DS="MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$(cat controlled/ds_proxy_url.txt)"
for r in $2; do for v in $1; do
  case $v in
    zs) V="FLAVOR=zero_shot";;
    noskills) V="FLAVOR=noskills,SEED_CODE=$LB/lud/hls_baseline.cpp";;
    skills) V="FLAVOR=skills,SEED_CODE=$LB/lud/hls_baseline.cpp";;
    refneu) V="FLAVOR=skills,REFINED=1,REFINED_NEUTRAL=1,SEED_CODE=$LB/lud/hls_baseline.cpp";;
  esac
  for m in cc ds; do
    case $m in cc) M=$CC;; ds) M=$DS;; esac
    t=lud_${v}_${m}_r$r
    python3 study/jobs/ledger.py submit $t --move-glob "$OUT/*_${t}_*" -- --job-name=$t --mem=128G --export=$COMMON,$V,$M,TAG=$t sbatch_flash_cc.sh
  done
done; done
