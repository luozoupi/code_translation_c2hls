#!/bin/bash
# job_csynth.sh <design dir with src/kernel.cpp> <top> : Vitis HLS 2023.2 csynth of a sodac --xocl-kernel file, U280, 3.33 ns.
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -eo pipefail
D=$1; TOP=$2
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
OUT=$D; LOCAL=${HLS_LOCAL_ROOT:-${SLURM_TMPDIR:-}}
if [ -n "$LOCAL" ]; then
  W=$LOCAL/soda_$(basename $D); mkdir -p $W; cp -r $D/src $W/
  back() { /scratch/asa582/workspaces/c2hls_claude_code/study/jobs/keep_copy.sh $W $OUT || true; }
  trap back EXIT; trap 'back; exit 143' TERM
  D=$W
fi
cd $D
cat > run_csynth.tcl <<TCL
open_project hls_prj_csynth
set_top $TOP
add_files src/kernel.cpp -cflags "-std=c++14"
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 80
csynth_design
exit
TCL
vitis-run --tcl --input_file run_csynth.tcl > vitis_csynth.log 2>&1 || echo "vitis-run rc=$?" >> vitis_csynth.log
grep -E "ERROR|Estimated|Latency" vitis_csynth.log | tail -5 || true
[ -n "$LOCAL" ] || rm -rf hls_prj_csynth/solution1/.autopilot hls_prj_csynth/solution1/syn/verilog hls_prj_csynth/solution1/syn/vhdl hls_prj_csynth/solution1/syn/systemc hls_prj_csynth/solution1/impl
