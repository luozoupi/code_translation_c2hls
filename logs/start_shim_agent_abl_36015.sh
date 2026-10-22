#!/bin/bash
# Agent-mode Claude Code shim for the agentic ablation (2026-10-15; fixed bench: C = A*B^T, poisoned C). Run on login3.
# Same settings as start_shim_agent_36014.sh, plus the zero-shot flash prompt in --agent-match so the agentic
# zero-shot runs also get the tool session.
cd /scratch/asa582/workspaces/c2hls_claude_code
exec /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/python-exec/python3.11/python3 claude_code_shim.py \
  --port 36015 --workers 6 --log-dir /scratch/asa582/workspaces/c2hls_claude_code/logs/shim_agent_abl_20261015 \
  --agent-match '^(FLASH MODE:|HARD REJECT —|The current HLS code already synthesizes, but the controller flagged|You are an expert HLS engineer\. Translate and optimize)' \
  --agent-bench-dir /scratch/asa582/workspaces/c2hls_claude_code/study/bench_fixed/n1024/autosa_mm \
  --agent-max-checks 12 --agent-csim-time 01:00:00 --agent-csynth-time 01:00:00 \
  --claude-bin /home/asa582/.local/bin/claude
