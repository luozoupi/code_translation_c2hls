#!/bin/bash
# Plain (no-tools) Claude Code shim used by the controlled runs. Run on login3.
cd /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code
exec python3 claude_code_shim.py \
  --port 38585 --workers 2 --log-dir /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/logs/shim_20261013T1130 \
  --claude-bin /pc2/users/h/haqc2/.local/bin/claude
