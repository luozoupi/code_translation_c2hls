#!/bin/bash
# AutoSA LU DSE on U280 / 3.33 ns: every N in the paper (12,16,20,24) plus the README default (32), every space_time 0..5.
# Generation in the AutoSA container (login node, seconds); csim + csynth on Slurm via study/jobs/ledger.py (1h->3h->6h).
set -uo pipefail
S=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/lud; L=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/jobs/ledger.py
for N in 12 16 20 24 32; do for ST in 0 1 2 3 4 5; do
  D=$S/dse/n${N}_st${ST}
  if [ ! -s $D/out/src/kernel_kernel.cpp ]; then
    timeout 180 $S/gen_lu.sh $N $ST $D > $D.gen.log 2>&1; rc=$?
    if [ $rc -ne 0 ] || [ ! -s $D/out/src/kernel_kernel.cpp ]; then
      echo "n$N st$ST: generation failed rc=$rc (124 = 180 s timeout)" | tee -a $S/dse/gen_failures.txt
      ps -u $USER -o pid,args | awk '/libs\/autosa/ && !/awk/ {print $1}' | xargs -r kill
      continue
    fi
  fi
  for M in csim csynth; do
    python3 $L submit lud_n${N}_st${ST}_$M -- --job-name=lud_${M} --output=$D/out/job_$M-%j.out $S/job_vitis.sh $D/out $M
  done
done; done
