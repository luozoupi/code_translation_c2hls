#!/bin/bash
#SBATCH --job-name=verify_csim
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/scratch/asa582/workspaces/c2hls_claude_code/verify_zeroC/control_orig/04_refined_v2_agent_r1
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/verify_zeroC/control_orig/04_refined_v2_agent_r1/csim-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/verify_zeroC/control_orig/04_refined_v2_agent_r1/csim-%j.err
set -euo pipefail
cd /scratch/asa582/workspaces/c2hls_claude_code/verify_zeroC/control_orig/04_refined_v2_agent_r1
# Only for PY, PYTHONPATH, temp dirs and the Vitis container; DSE switches are unset below.
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=86400 C2HLS_SYNTH_TIMEOUT=86400
"${PY}" /scratch/asa582/workspaces/c2hls_claude_code/verify_zeroC/control_orig/04_refined_v2_agent_r1/run_eval.py csim
