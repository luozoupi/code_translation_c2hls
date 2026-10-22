#!/bin/bash
# Post-hoc csynth, 2026-10-22, of /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/runs/mm1024_know_r1/exp/v6_single_loop/kernel.cpp: csim passed in its session, never synthesized there.
#SBATCH --account=def-zhenman
#SBATCH --job-name=posthoc_csynth
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/mm1024_know_r1/exp_v6_single_loop/csynth-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/mm1024_know_r1/exp_v6_single_loop/csynth-%j.err
set -eo pipefail
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=200000 C2HLS_SYNTH_TIMEOUT=200000
"$PY" /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py csynth /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/mm1024_know_r1/exp_v6_single_loop /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/ws/task.json
