#!/bin/bash
# Ablation with the plain 90-skills pack (skills_ii_target_miss_solutions_added(90skills).json, 123 skills, no
# gemm_flatten, packaged skills only). mm 1024^3: s90 (flash + pack) and s90dsp (+ DSP floor, the "DSP redo" step);
# LUD N=32: s90. Both models, 3 reps, fixed benches, controlled LLM layer, ledger wall time (1h -> 3h -> 6h).
set -euo pipefail
cd /scratch/asa582/workspaces/c2hls_claude_code
INC=/scratch/asa582/tmp/c2hls/vitis_hls_2023.2_include; OUT=$PWD/study/s90/runs
PACK='skills_ii_target_miss_solutions_added(90skills).json'
FB=$PWD/study/bench_fixed/n1024; LB=$PWD/study/lud/bench/n32
CC="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat shim_url.txt)"
DS="MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$(cat controlled/ds_proxy_url.txt)"
BASE="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,OUT_ROOT=$OUT,C2HLS_COMPILE_INCLUDE_PATHS=$INC,FLAVOR=skills,SKILLS_JSON=$PACK"
MM="BENCH_ROOT=$FB,SEED_CODE=$FB/autosa_mm/hls_baseline.cpp"
LU="BENCH_ROOT=$LB,BENCH=lud,SEED_CODE=$LB/lud/hls_baseline.cpp"
for r in 1 2 3; do
  for v in s90 s90dsp lud_s90; do
    case $v in s90) V="$MM";; s90dsp) V="$MM,DSP_REDO=1";; lud_s90) V="$LU";; esac
    for m in cc ds; do
      case $m in cc) M=$CC;; ds) M=$DS;; esac
      t=${v}_${m}_r$r
      python3 study/jobs/ledger.py submit $t --move-glob "$OUT/*_${t}_*" -- --job-name=$t --mem=128G --export=$BASE,$V,$M,TAG=$t sbatch_flash_cc.sh
    done
  done
done
