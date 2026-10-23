#!/bin/bash
# Resubmit every controlled refv2/win7080 run whose Slurm job ended in TIMEOUT, with a 6 h limit
# (double the 3 h). The timed-out cell is moved to runs_controlled/_superseded_timeout_3h/.
set -uo pipefail
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
R=runs_controlled; S=$R/_superseded_timeout_3h; mkdir -p $S
SEED=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls/artifacts/pc2/autosa_mm_ijk_benches/n1024/autosa_mm/hls_baseline.cpp
INC=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include
COMMON="ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,OUT_ROOT=$PWD/$R,C2HLS_COMPILE_INCLUDE_PATHS=$INC"
PAT=${1:?usage: resubmit_timeouts.sh <tag regex>, e.g. _skills_}  # only tags not yet given their 6 h rerun
awk -v p="$PAT" '$1 ~ p' $R/jobs.txt | while read t j; do
  st=$(sacct -X -n -j $j -o State | awk '{print $1}' | head -1)
  [ "$st" = TIMEOUT ] || continue
  for d in $R/*_${t}_*; do [ -d "$d" ] && mv "$d" $S/; done
  echo "$t $j" >> $S/jobs.txt
  m=$(echo $t | cut -d_ -f2); v=$(echo $t | cut -d_ -f3)
  case $v in skills) V="FLAVOR=skills,SEED_CODE=$SEED";; refnf) V="FLAVOR=skills,REFINED=1,NO_DSP_FLOOR=1,SEED_CODE=$SEED";; refneu) V="FLAVOR=skills,REFINED=1,REFINED_NEUTRAL=1,SEED_CODE=$SEED";; refv2) V="FLAVOR=skills,REFINED=1,SEED_CODE=$SEED";; win7080) V="FLAVOR=skills,REFINED=1,DSP_WINDOW=6317:7219,SEED_CODE=$SEED";; esac
  if [ $m = cc ]; then MM="MODEL=claude-opus-5-5,CLAUDE_CODE_SHIM_URL=$(cat shim_url.txt)"; else MM="MODEL=deepseek-v4-flash,CLAUDE_CODE_SHIM_URL=$(cat controlled/ds_proxy_url.txt)"; fi
  nj=$(sbatch --parsable --job-name=$t --time=${RESUB_TIME:-06:00:00} --mem=128G --export=$COMMON,$V,$MM,TAG=$t sbatch_flash_cc.sh 2>/dev/null)
  sed -i "s/^$t $j\$/$t $nj/" $R/jobs.txt
  echo "resubmitted $t: $j (TIMEOUT) -> $nj (6h)"
done
