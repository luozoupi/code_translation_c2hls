#!/bin/bash
# mk.sh <variant dir> : set up strict csim (fixed bench) and csynth jobs for study/manual_df/<variant>/kernel.cpp
set -euo pipefail
H=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
D=$(cd "$1" && pwd); T=$H/verify/ctl_cc_zs_r1; BEN=$H/study/bench_fixed/n1024/autosa_mm
cp $BEN/kernel.h $BEN/testbench.cpp $D/; cp $H/study/jobs/run_eval_local.py $D/run_eval.py
sed -e "s#$T#$D#g" -e 's/--cpus-per-task=8/--cpus-per-task=2/' -e 's/--mem=64G/--mem=16G/' -e 's/--time=24:00:00/--time=01:00:00/' \
    -e 's/C2HLS_CSIM_TIMEOUT=86400/C2HLS_CSIM_TIMEOUT=3300/' -e 's/--job-name=verify_csim/--job-name=mdf_csim/' $T/sbatch_csim.sh > $D/sbatch_csim.sh
# csynth: 1 h first (escalate 3 h, then 6 h on TIMEOUT via --time override at resubmit)
sed -e "s#$T#$D#g" -e 's/--time=24:00:00/--time=01:00:00/' -e 's/--job-name=verify_csim/--job-name=mdf_csynth/' \
    -e 's#run_eval.py csim#run_eval.py csynth#' -e 's#csim-%j#csynth-%j#g' $T/sbatch_csim.sh > $D/sbatch_csynth.sh
echo "$(basename $D) csim $(sbatch --parsable $D/sbatch_csim.sh) csynth $(sbatch --parsable $D/sbatch_csynth.sh)" | tee -a $H/study/manual_df/jobs.txt
