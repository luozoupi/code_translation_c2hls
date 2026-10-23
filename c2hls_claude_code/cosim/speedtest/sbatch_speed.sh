#!/bin/bash
# C/RTL cosim of one kernel on the 1024^3 bench, one node.  sbatch --export=ALL,NAME=x,KERNEL=/path/k.cpp sbatch_cosim.sh
#SBATCH --job-name=cospeed
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/speedtest/logs/speed-%x-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/speedtest/logs/speed-%x-%j.err
set -euo pipefail
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
# Storage: no core dumps (a killed 128 GB simulator can dump one), and all Vitis work on the node-local disk
# ($SLURM_TMPDIR), which Slurm deletes when the job ends, even when it is killed. Only small results reach /scratch.
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis
export C2HLS_TMP_ROOT="${SLURM_TMPDIR}/c2hls"
mkdir -p "$C2HLS_TMP_ROOT"
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
# hls_eval reads these at import: generous timeouts inside the 7-day wall, no waveform.
# -disable_dependency_check: no DEPENDENCE-pragma checker; its millions of warnings bloated logs (GBs) and the
# runner's in-memory log. Values are still checked by the testbench. xelab -mt off (avoids the xelab SIGSEGV on large designs); run_cosim_one.py runs sim.sh inside the Vitis container.
export C2HLS_SYNTH_TIMEOUT=86400 C2HLS_CSIM_TIMEOUT=86400 C2HLS_COSIM_TIMEOUT=580000 C2HLS_COSIM_TRACE_LEVEL=none C2HLS_COSIM_EXTRA_ARGS="-disable_dependency_check ${COSIM_O:+-O}" C2HLS_COSIM_XELAB_MT_OFF=1
"${PY}" /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/speedtest/run_speed.py "${NAME:?}" "${KERNEL:?}" ${SIZE:+--size "$SIZE"}
