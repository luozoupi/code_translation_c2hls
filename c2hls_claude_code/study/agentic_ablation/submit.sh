#!/bin/bash
# Agentic ablation (supervisor's request of 2026-10-13), fixed bench (C = A*B^T, C poisoned), 1024^3, U280, 3.33 ns.
#   Agent mode (Opus 5.5 in Claude Code with Read/Edit/Bash(./hls_check.sh) and real Vitis csim/csynth, 12 checks per
#   session), one tag per rung of the plain ladder, r1-r3:
#     ag_zs       zero-shot prompt                         ag_zsk      zero-shot prompt + skill block (Expertise pack)
#     ag_noskills flash, no skills                         ag_skills   flash + skills (90 + gemm-flatten v1)
#     ag_dspredo  skills + DSP floor (90% fill)            ag_refneu   neutral refined skills, no DSP ask
#     ag_refnf    refined, DSP asked in words only         ag_refv2    refined + DSP floor
#     ag_win7080  refined + DSP window 70-80%
#   Plain mode (no tools; controlled layer), both models, r1-r3:
#     pl_zs  zero-shot on the fixed bench                  pl_zsk  zero-shot + the same skill block
# Agent jobs run in 6 Slurm singleton slots (one agent session per slot at a time; the agent shim has 6 workers) and
# start at the 3 h tier; plain jobs start at 1 h. Wall time escalation: study/jobs/ledger.py escalate.
set -euo pipefail
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
FB=$PWD/study/bench_fixed/n1024; OUT=$PWD/study/agentic_ablation/runs; mkdir -p "$OUT"
INC=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include
BLOCK=$PWD/study/agentic_ablation/skills_block_90gemm_v1.txt
COMMON="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,OUT_ROOT=$OUT,C2HLS_COMPILE_INCLUDE_PATHS=$INC,BENCH_ROOT=$FB"
SEED="SEED_CODE=$FB/autosa_mm/hls_baseline.cpp"
AG="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat study/agentic_ablation/shim_agent_url.txt),CC_LLM_TIMEOUT=50400"
CC="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat shim_url.txt)"
DS="MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$(cat controlled/ds_proxy_url.txt)"
L=study/jobs/ledger.py
rung() {
  case $1 in
    zs)       echo "FLAVOR=zero_shot";;
    zsk)      echo "FLAVOR=zero_shot,ZS_SKILLS_BLOCK=$BLOCK";;
    noskills) echo "FLAVOR=noskills,$SEED";;
    skills)   echo "FLAVOR=skills,$SEED";;
    dspredo)  echo "FLAVOR=skills,DSP_REDO=1,$SEED";;
    refneu)   echo "FLAVOR=skills,REFINED=1,REFINED_NEUTRAL=1,$SEED";;
    refnf)    echo "FLAVOR=skills,REFINED=1,NO_DSP_FLOOR=1,$SEED";;
    refv2)    echo "FLAVOR=skills,REFINED=1,$SEED";;
    win7080)  echo "FLAVOR=skills,REFINED=1,DSP_WINDOW=6317:7219,$SEED";;
  esac
}
i=0
for r in 1 2 3; do
  for v in zs zsk noskills skills dspredo refneu refnf refv2 win7080; do
    t=ag_${v}_r$r; slot=agabl_s$((i % 6)); i=$((i + 1))
    python3 $L submit $t --move-glob "$OUT/*_${t}_*" --tier 1 -- --job-name=$slot --dependency=singleton \
      --mem=128G --export=$COMMON,$(rung $v),$AG,TAG=$t sbatch_flash_cc.sh
  done
  for v in zs zsk; do
    for m in cc ds; do
      case $m in cc) M=$CC;; ds) M=$DS;; esac
      t=pl_${v}_${m}_r$r
      python3 $L submit $t --move-glob "$OUT/*_${t}_*" -- --job-name=$t --mem=64G --export=$COMMON,$(rung $v),$M,TAG=$t sbatch_flash_cc.sh
    done
  done
done
