#!/bin/bash
# Post-hoc csynth, 2026-10-22, of /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/logs/shim_agent_abl_20261015/call_00009/workspace/checks/8_csim_work/kernel.cpp: csim passed in its session, never synthesized there.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --job-name=posthoc_csynth
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/skills_ag_refnf_r1_20261016T161951Z/call_00009_8_csim/csynth-%j.out
#SBATCH --error=/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/skills_ag_refnf_r1_20261016T161951Z/call_00009_8_csim/csynth-%j.err
set -eo pipefail
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=200000 C2HLS_SYNTH_TIMEOUT=200000
"$PY" /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py csynth /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/csynth/skills_ag_refnf_r1_20261016T161951Z/call_00009_8_csim /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/posthoc_csim/ws/task.json
