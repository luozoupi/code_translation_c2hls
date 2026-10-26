#!/bin/bash
# submit_stp_all.sh [--dry-run] : LightningSim of the patched SODA 3D designs (heat3d, jacobi3d) with every pipeline
# forced to style = stp (no Vitis free-running pipelines, which LightningSim does not model). heat3d_u1 was the test
# (heat3d_u1_stp, cosim 3,016,143 = LightningSim); this does the other 13. Per design: copy to inv3d/<d>_stp, append
# "style = stp" to each "#pragma HLS pipeline II = N", csynth into R/<d>_stp/proj.tar.gz, then LightningSim (afterok).
set -eo pipefail
B=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/soda
L=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
R=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/inv3d
MODE=${1:-}; mkdir -p $R
for d in heat3d_u2 heat3d_u4 heat3d_u4_b128 heat3d_u4_b128_laafd heat3d_u8 heat3d_u16 \
         jacobi3d_u1 jacobi3d_u2 jacobi3d_u4 jacobi3d_u4_b128 jacobi3d_u4_b128_laafd jacobi3d_u8 jacobi3d_u16; do
  E=$B/inv3d/${d}_stp
  if [ ! -d $E ]; then
    mkdir -p $E; cp $B/dse/designs_patched/$d/{kernel.cpp,host.cpp,frt.h,config.json} $E/
    sed -i -E 's|^(\s*#pragma HLS pipeline II = [0-9]+)\s*$|\1 style = stp  // inv3d: was auto|' $E/kernel.cpp
  fi
  n=$(grep -c 'style = stp' $E/kernel.cpp); echo "$d: $n pipelines set to stp"
  case $d in *laafd|*_u8) MEM=480G;; *) MEM=192G;; esac   # these OOM'd at 192G in lsim_soda_pat 3820398
  if [ "$MODE" = --dry-run ]; then echo "  proj + lsim --mem=$MEM"; continue; fi
  J1=$(sbatch --parsable -J inv3d_stp_proj -t 12:00:00 --mem=96G -o $E/job_proj-%j.out $B/lsim/job_proj_otus.sh $E $R/${d}_stp)
  J2=$(sbatch --parsable -J inv3d_stp_lsim -d afterok:$J1 -t 24:00:00 --cpus-per-task=4 --mem=$MEM -o $R/lsim-%j.out \
       --export=ALL,LSIM_HERE=$L,LSIM_T=/scratch/hpc-prf-llmfpga/asa582/projects/tools/lightningsim,ROOT=$R,MODEL=v2023,PAR=1,ONLY=${d}_stp \
       $L/otus/sbatch_lsim_otus.sh)
  echo "${d}_stp proj $J1 lsim $J2 (mem $MEM)" | tee -a $B/inv3d/submitted.txt
done
