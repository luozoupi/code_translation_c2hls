#!/bin/bash
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/bench_wide/_doitgen_fix/job-%j.out
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis
set +e +u
H=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/bench_wide/_doitgen_fix
for v in v3_loopvars v4_localsum; do
  W=$SLURM_TMPDIR/$v; mkdir -p $W/home $W/tmp; cp $H/$v.cpp $W/kernel.cpp; cp $H/doitgen.h $W/
  cat > $W/run.tcl <<TCL
open_project p
set_top kernel_doitgen
add_files kernel.cpp
open_solution s -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33
csynth_design
exit
TCL
  (cd $W && HOME=$W/home TMPDIR=$W/tmp vitis-run --tcl --input_file run.tcl > vitis.log 2>&1)
  echo "== $v"; grep -E "ERROR|Finished C synthesis|Latency" $W/vitis.log | head -5; echo "-- tail"; tail -n 8 $W/vitis.log
done
