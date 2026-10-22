#!/bin/bash
# Post-hoc csim, 2026-10-22, of /scratch/asa582/workspaces/c2hls_claude_code/logs/shim_agent_abl_20261015/call_00006/workspace/checks/4_csynth_work/kernel.cpp: synthesized in its session, never csim'd there.
#SBATCH --account=def-zhenman
#SBATCH --job-name=posthoc_csim
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/skills_ag_dspredo_r1_20261016T104045Z/4/csim-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/skills_ag_dspredo_r1_20261016T104045Z/4/csim-%j.err
set -eo pipefail
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=10000 C2HLS_SYNTH_TIMEOUT=10000
"$PY" /scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py csim /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/skills_ag_dspredo_r1_20261016T104045Z/4 /scratch/asa582/workspaces/c2hls_claude_code/study/posthoc_csim/ws/task.json
