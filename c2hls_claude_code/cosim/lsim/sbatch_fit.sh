#!/bin/bash
# Fit the v2023 AXI constants on the microbenchmarks (fit_lsim.py inside the Vitis container), one node.
#   sbatch --export=ALL,MB=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/mb,VCD=/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/mb_vcd,OUT=.../fit.json sbatch_fit.sh
#SBATCH --job-name=lsim_fit
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim/logs/fit-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim/logs/fit-%j.err
set -uo pipefail
HERE=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/cosim/lsim
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
ulimit -c 0
export APPTAINER_BINDPATH="/scratch/${USER},${HOME}/.Xilinx,${SLURM_TMPDIR}"
dse_v3_source_vitis >/dev/null
set +e
ENV=/scratch/asa582/tools/lightningsim/env
export APPTAINERENV_TMPDIR="$SLURM_TMPDIR"
apptainer exec "$XILINX_SIF" bash -c "
  export XILINX_HLS=/opt/Xilinx/Vitis_HLS/2023.2 PATH=$ENV/bin:\$PATH
  unset LD_LIBRARY_PATH PYTHONPATH
  cd / && $ENV/bin/python $HERE/fit_lsim.py '${MB:?}' '${VCD:?}' '${OUT:?}' --procs ${PROCS:-14}"
