#!/bin/bash
# Agent-mode Claude Code shim for the study (fixed bench: C = A*B^T, poisoned C). Run on login3.
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
exec python3 claude_code_shim.py \
  --port 36014 --workers 3 --log-dir /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/logs/shim_agent_study_20261013 \
  --agent-match '^(FLASH MODE:|HARD REJECT —|The current HLS code already synthesizes, but the controller flagged)' \
  --agent-bench-dir /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/bench_fixed/n1024/autosa_mm \
  --agent-max-checks 12 --agent-csim-time 01:00:00 --agent-csynth-time 01:00:00 \
  --claude-bin /pc2/users/h/haqc2/.local/bin/claude
