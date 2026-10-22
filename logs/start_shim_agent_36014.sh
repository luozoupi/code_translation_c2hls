#!/bin/bash
# Agent-mode Claude Code shim for the study (fixed bench: C = A*B^T, poisoned C). Run on login3.
cd /scratch/asa582/workspaces/c2hls_claude_code
exec /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/python-exec/python3.11/python3 claude_code_shim.py \
  --port 36014 --workers 3 --log-dir /scratch/asa582/workspaces/c2hls_claude_code/logs/shim_agent_study_20261013 \
  --agent-match '^(FLASH MODE:|HARD REJECT —|The current HLS code already synthesizes, but the controller flagged)' \
  --agent-bench-dir /scratch/asa582/workspaces/c2hls_claude_code/study/bench_fixed/n1024/autosa_mm \
  --agent-max-checks 12 --agent-csim-time 01:00:00 --agent-csynth-time 01:00:00 \
  --claude-bin /home/asa582/.local/bin/claude
