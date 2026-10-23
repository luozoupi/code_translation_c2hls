#!/bin/bash
# job_soda.sh <design dir> <csynth|cosim> : Vitis HLS 2023.2 on one gen_dse.py design, U280, 3.33 ns, node-local disk.
#   csynth: resources (latency is "?" for SODA's data-driven modules).
#   cosim : csynth + cosim with SODA's own test main (host.cpp, -DSODA_TEST_MAIN -DSODA_CPP_BINDING) as testbench;
#           no waveform. Cycles from the cosim report; correctness = SODA's golden ("INFO: PASS").
# Keeps only reports + log tails in <design dir>/<mode>/.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -eo pipefail
D=$(readlink -f $1); MODE=$2
TOP=$(python3 -c "import json;print(json.load(open('$D/config.json'))['top'])")
ARGV=$(python3 -c "import json;print(json.load(open('$D/config.json'))['argv'])")
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis
W=$SLURM_TMPDIR/soda_$(basename $D)_$MODE; mkdir -p $W/home/.Xilinx $W/tmp
cp $D/kernel.cpp $D/host.cpp $D/frt.h $W/
O=$D/$MODE; mkdir -p $O
keep() {
  cp $W/hls_prj/sol1/syn/report/csynth.rpt $O/ 2>/dev/null || true
  cp $W/hls_prj/sol1/syn/report/csynth.xml $O/ 2>/dev/null || true
  cp $W/hls_prj/sol1/sim/report/*cosim.rpt $O/ 2>/dev/null || true
  find $W/hls_prj/sol1/sim -name "*.result.lat.rb" -exec cp {} $O/ \; 2>/dev/null || true
  grep -E "INFO: (PASS|FAIL)|ERROR|CRITICAL|Estimated|Latency|COSIM|co-simulation|!= .*@" $W/vitis.log 2>/dev/null | head -200 > $O/vitis_grep.log || true
  tail -n 150 $W/vitis.log > $O/vitis_tail.log 2>/dev/null || true
}
trap keep EXIT; trap 'keep; exit 143' TERM
cd $W
COSIM=""
[ "$MODE" = cosim ] && COSIM="cosim_design -trace_level none -argv \"$ARGV\""
cat > run.tcl <<TCL
open_project hls_prj
set_top $TOP
add_files kernel.cpp -cflags "-std=c++14"
add_files -tb host.cpp -cflags "-std=c++14 -DSODA_TEST_MAIN -DSODA_CPP_BINDING -I$W"
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 80
csynth_design
$COSIM
exit
TCL
HOME=$W/home TMPDIR=$W/tmp vitis-run --tcl --input_file run.tcl > vitis.log 2>&1 || echo "vitis-run rc=$?" >> vitis.log
grep -E "INFO: (PASS|FAIL)|ERROR|Total Execution|Verilog\|" vitis.log | tail -5 || true
