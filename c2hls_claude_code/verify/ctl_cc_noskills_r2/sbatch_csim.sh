#!/bin/bash
#SBATCH --job-name=verify_csim
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/verify/ctl_cc_noskills_r2
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/verify/ctl_cc_noskills_r2/csim-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/verify/ctl_cc_noskills_r2/csim-%j.err
set -euo pipefail
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/verify/ctl_cc_noskills_r2
# Only for PY, PYTHONPATH, temp dirs and the Vitis container; DSE switches are unset below.
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=3000 C2HLS_SYNTH_TIMEOUT=86400
"${PY}" /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/verify/ctl_cc_noskills_r2/run_eval.py csim
