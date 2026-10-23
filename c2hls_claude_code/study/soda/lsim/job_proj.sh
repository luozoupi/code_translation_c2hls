#!/bin/bash
# job_proj.sh <design dir> <out dir> : csynth of one SODA design (Vitis HLS 2023.2, U280, 3.33 ns) on node-local disk,
# saved as <out>/proj.tar.gz in the layout LightningSim reads (../lsim harness: hls_proj/sol1 + host.cpp, frt.h).
# The testbench is SODA's host (-DSODA_TEST_MAIN -DSODA_CPP_BINDING); its argv (config.json) goes into hls.app's
# <Simulation argv=...>, which is where LightningSim reads it. Nothing else is changed.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -eo pipefail
D=$(readlink -f $1); O=$(readlink -f -m $2); mkdir -p $O
TOP=$(python3 -c "import json;print(json.load(open('$D/config.json'))['top'])")
ARGV=$(python3 -c "import json;print(json.load(open('$D/config.json'))['argv'])")
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis
W=$SLURM_TMPDIR/lp_$(basename $O); mkdir -p $W/home/.Xilinx $W/tmp
cp $D/kernel.cpp $D/host.cpp $D/frt.h $W/
cd $W
cat > run.tcl <<TCL
open_project hls_proj
set_top $TOP
add_files kernel.cpp -cflags "-std=c++14"
add_files -tb host.cpp -cflags "-DSODA_TEST_MAIN -DSODA_CPP_BINDING -I../.."
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 80
csynth_design
exit
TCL
HOME=$W/home TMPDIR=$W/tmp vitis-run --tcl --input_file run.tcl > vitis.log 2>&1 || echo "vitis-run rc=$?" >> vitis.log
tail -n 60 vitis.log > $O/vitis_tail.log
cp hls_proj/sol1/syn/report/csynth.rpt $O/ 2>/dev/null || { echo "csynth failed"; exit 1; }
python3 - hls_proj/hls.app "$ARGV" <<'PY'
import sys, re
p, argv = sys.argv[1], sys.argv[2]
s = open(p).read()
assert "<Simulation" not in s, "hls.app already has a Simulation element"
s = s.replace("</AutoPilot:project>", f'    <Simulation argv="{argv}"/>\n</AutoPilot:project>')
# Vitis stores the TB's -I../.. relative to the project dir; LightningSim compiles from sol1, where the root is ../..
s = s.replace("-I../../../../.", "-I../..")
open(p, "w").write(s)
PY
tar --exclude='hls_proj/sol1/sim' --exclude='hls_proj/sol1/impl' -czf $O/proj.tar.gz hls_proj kernel.cpp host.cpp frt.h
cp $D/config.json $O/
echo "proj ok $(du -h $O/proj.tar.gz | cut -f1)"
