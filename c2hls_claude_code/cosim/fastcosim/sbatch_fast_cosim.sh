#!/bin/bash
# TAPA/PASTA fast cosim of one kernel on the 1024^3 bench, one node.  sbatch --export=ALL,NAME=x,KERNEL=/path/k.cpp sbatch_fast_cosim.sh
#SBATCH --job-name=fastcosim
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/fastcosim
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/fastcosim/logs/fastcosim-%x-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/fastcosim/logs/fastcosim-%x-%j.err
set -euo pipefail
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
# Storage: no core dumps, and all work on the node-local disk ($SLURM_TMPDIR), deleted by Slurm when the job ends.
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis
export C2HLS_TMP_ROOT="${SLURM_TMPDIR}/c2hls"
mkdir -p "$C2HLS_TMP_ROOT"
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_SYNTH_TIMEOUT=86400
"${PY}" /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/fastcosim/run_fast_cosim.py "${NAME:?}" "${KERNEL:?}" ${SIZE:+--size "$SIZE"}
