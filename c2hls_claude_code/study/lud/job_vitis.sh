#!/bin/bash
# job_vitis.sh <design out dir> <csim|csynth> : Vitis HLS 2023.2 (Fir container) on an AutoSA LU design, U280, 3.33 ns.
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -eo pipefail
D=$1; MODE=$2
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
# File-count quota on /scratch: run in node-local storage, copy back only reports/sources/json/logs (keep_copy.sh).
OUT=$D; LOCAL=${HLS_LOCAL_ROOT:-${SLURM_TMPDIR:-}}
if [ -n "$LOCAL" ]; then
  W=$LOCAL/lud_$(basename $(dirname $D))_$MODE; mkdir -p $W; cp -r $D/src $W/
  back() { /scratch/asa582/workspaces/c2hls_claude_code/study/jobs/keep_copy.sh $W $OUT || true; }
  trap back EXIT; trap 'back; exit 143' TERM
  D=$W
fi
cd $D
P=hls_prj_$MODE
cat > run_$MODE.tcl <<TCL
open_project $P
set_top kernel0
add_files src/kernel_kernel.h
add_files src/kernel_kernel.cpp
add_files -tb src/kernel_host.cpp
add_files -tb src/kernel.h
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 50
$( [ $MODE = csim ] && echo csim_design || echo csynth_design )
exit
TCL
vitis-run --tcl --input_file run_$MODE.tcl > vitis_$MODE.log 2>&1 || echo "vitis-run rc=$?" >> vitis_$MODE.log
grep -E "Passed|Failed|ERROR" vitis_$MODE.log | tail -5 || true
# file-count quota: if not node-local, keep syn/report (.rpt/.xml) and the log; drop intermediates and RTL (no xclbin)
[ -n "$LOCAL" ] || rm -rf $P/solution1/.autopilot $P/solution1/syn/verilog $P/solution1/syn/vhdl $P/solution1/syn/systemc $P/solution1/impl $P/solution1/csim/build
