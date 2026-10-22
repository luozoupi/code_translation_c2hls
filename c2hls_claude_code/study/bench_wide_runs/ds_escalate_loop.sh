#!/bin/bash
# Plain loop (no LLM): every 20 min, resubmit timed-out / OOM DeepSeek bench jobs (bw_*_ds only) at the next tier.
# Exits when no bw_*_ds job is queued or running after an escalation pass.
cd /scratch/asa582/workspaces/c2hls_claude_code
while true; do
  python3 study/jobs/ledger.py escalate --match '^bw_.*_ds$' >> study/bench_wide_runs/ds_escalate.log 2>&1
  n=$(squeue -u asa582 -h -o %j | grep -c '^bw_.*_ds$')
  [ "$n" -eq 0 ] && { echo "$(date) no bw_*_ds jobs left" >> study/bench_wide_runs/ds_escalate.log; exit 0; }
  sleep 1200
done
