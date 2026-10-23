#!/bin/bash
# One design at one size: csynth (+ optional Vitis cosim) via run_design.py, on node-local disk.
#   sbatch --job-name=ld_NAME --export=ALL,NAME=x,KERNEL=/path/kernel.cpp,SIZE=256,COSIM=1[,VCD=1] sbatch_design.sh
#SBATCH --job-name=lsim_design
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim/logs/design-%x-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim/logs/design-%x-%j.err
set -uo pipefail
HERE=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis >/dev/null
set +e
export C2HLS_TMP_ROOT="${SLURM_TMPDIR}/c2hls"
mkdir -p "$C2HLS_TMP_ROOT"
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
# Same cosim settings as ../sbatch_cosim.sh (no dependence checker, xelab -mt off, no waveform unless VCD=1).
export C2HLS_SYNTH_TIMEOUT=86400 C2HLS_CSIM_TIMEOUT=86400 C2HLS_COSIM_TIMEOUT=170000 C2HLS_COSIM_TRACE_LEVEL=none \
       C2HLS_COSIM_EXTRA_ARGS="-disable_dependency_check" C2HLS_COSIM_XELAB_MT_OFF=1
"$PY" "$HERE/run_design.py" "${NAME:?}" "${KERNEL:?}" "${OUT:-/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/designs}" --size "${SIZE:?}" \
  ${COSIM:+--cosim} ${VCD:+--vcd} ${TOP:+--top "$TOP"} ${TB:+--tb "$TB"} ${HDR:+--hdr "$HDR"}
