#!/bin/bash
# Strict n1024 csim of a controlled run's selected kernel: controlled/submit_csim.sh ctl_<m>_<v>_r<n>
set -euo pipefail
CC=/scratch/asa582/workspaces/c2hls_claude_code
T=$1; SRC=verify/ctl_cc_zs_r1
cell=$(ls -d $CC/runs_controlled/*_${T}_*/autosa_mm | tail -1)
D=$CC/verify/$T; mkdir -p $D
cp $cell/autosa_mm_selected.cpp $D/kernel.cpp
cp $CC/$SRC/{kernel.h,testbench.cpp,run_eval.py} $D/
sed -e "s#$SRC#verify/$T#g" -e 's/--cpus-per-task=8/--cpus-per-task=2/' -e 's/--mem=64G/--mem=16G/' \
    -e 's/--time=24:00:00/--time=01:00:00/' -e 's/C2HLS_CSIM_TIMEOUT=86400/C2HLS_CSIM_TIMEOUT=3000/' \
    $CC/$SRC/sbatch_csim.sh > $D/sbatch_csim.sh
echo "$T $(sbatch --parsable $D/sbatch_csim.sh) $cell"
