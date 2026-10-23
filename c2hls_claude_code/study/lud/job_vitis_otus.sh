#!/bin/bash
# job_vitis.sh <design out dir> <csim|csynth> : Vitis HLS 2023.2 (Fir container) on an AutoSA LU design, U280, 3.33 ns.
# Otus copy of job_vitis.sh: native Vitis instead of the Fir container; node-local dir falls back when $SLURM_TMPDIR is
# unset; STUDY_DIR (for jobs/keep_copy.sh) comes from resubmit_lud_otus.sh.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -eo pipefail
D=$1; MODE=$2
source ${OTUS_VITIS_SETTINGS:-/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/otus/vitis_settings_otus.sh}
command -v vitis-run
KEEP=${STUDY_DIR:?}/jobs/keep_copy.sh
# File-count quota on /scratch: run in node-local storage, copy back only reports/sources/json/logs (keep_copy.sh).
OUT=$D; LOCAL=${HLS_LOCAL_ROOT:-${SLURM_TMPDIR:-${OTUS_LOCAL_TMP:-/tmp}/$USER/$SLURM_JOB_ID}}
if [ -n "$LOCAL" ]; then
  W=$LOCAL/lud_$(basename $(dirname $D))_$MODE; mkdir -p $W; cp -r $D/src $W/
  back() { $KEEP $W $OUT || true; [ -n "${SLURM_TMPDIR:-}${HLS_LOCAL_ROOT:-}" ] || rm -rf ${OTUS_LOCAL_TMP:-/tmp}/$USER/$SLURM_JOB_ID; }
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
