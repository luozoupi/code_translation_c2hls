#!/bin/bash
# Plain (no-tools) Claude Code shim used by the controlled runs. Run on login3.
cd /scratch/asa582/workspaces/c2hls_claude_code
exec /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/python-exec/python3.11/python3 claude_code_shim.py \
  --port 38585 --workers 2 --log-dir /scratch/asa582/workspaces/c2hls_claude_code/logs/shim_20261013T1130 \
  --claude-bin /home/asa582/.local/bin/claude
