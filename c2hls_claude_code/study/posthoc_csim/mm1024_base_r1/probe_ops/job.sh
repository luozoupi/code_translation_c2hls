#!/bin/bash
# Post-hoc csim, 2026-10-22, of /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/agentfw/runs/mm1024_base_r1/exp/probe_ops/kernel.cpp: synthesized in its session, never csim'd there.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --job-name=posthoc_csim
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/mm1024_base_r1/probe_ops/csim-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/mm1024_base_r1/probe_ops/csim-%j.err
set -eo pipefail
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=10000 C2HLS_SYNTH_TIMEOUT=10000
"$PY" /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py csim /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/mm1024_base_r1/probe_ops /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/ws/task.json
