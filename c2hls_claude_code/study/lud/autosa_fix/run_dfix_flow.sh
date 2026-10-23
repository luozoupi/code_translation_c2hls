#!/bin/bash
# "AutoSA + dangling-stream repair" flow over the LUD design space (ablation partner of plain AutoSA in dse/).
# For every dse/<design> whose csynth stopped with HLS 200-970, write dse_dfix/<design>/out/src with the named
# streams localized (fix_dangling_streams.py --only-from <csynth log>), then submit Vitis csim and csynth through the
# ledger (csim 1 h; csynth 3 h -> 12 h -> 24 h -> 2 days). Designs without HLS 200-970 are not copied: their repaired
# flow result is the plain AutoSA result, byte-identical source.
set -euo pipefail
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/lud
L=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/jobs/ledger.py
for d in dse/n*_st[0-9]; do
  b=$(basename $d); log=$(grep -l "HLS 200-970" $d/out/job_csynth-*.out 2>/dev/null | tail -1 || true)
  [ -n "$log" ] || continue
  o=dse_dfix/$b/out; mkdir -p $o/src; cp $d/out/src/* $o/src/
  python3 autosa_fix/fix_dangling_streams.py $d/out/src/kernel_kernel.cpp $o/src/kernel_kernel.cpp --only-from $log > $o/dfix_report.json
  python3 $L submit lud_dfix_${b}_csim -- --job-name=lud_dfix_csim --output=$PWD/$o/job_csim-%j.out $PWD/job_vitis.sh $PWD/$o csim
  python3 $L submit lud_dfix_${b}_csynth --tiers 03:00:00,12:00:00,24:00:00,2-00:00:00 -- --job-name=lud_dfix_csynth --output=$PWD/$o/job_csynth-%j.out $PWD/job_vitis.sh $PWD/$o csynth
done
