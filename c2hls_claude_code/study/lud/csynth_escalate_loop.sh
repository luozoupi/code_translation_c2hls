#!/bin/bash
# Plain loop (no LLM): every 30 min, escalate the 5 long LUD csynth jobs (12 h -> 24 h -> 2 days, per-tag tiers in the
# ledger). Exits when none of them is queued or running after an escalation pass.
cd /scratch/asa582/workspaces/c2hls_claude_code/study
M='^lud_n(20_st5|24_st5|32_st3|32_st4|32_st5)_csynth$'
while true; do
  python3 jobs/ledger.py escalate --match "$M" >> lud/csynth_escalate.log 2>&1
  live=$(python3 - <<'PY'
import json,subprocess,re
M=re.compile(r'^lud_n(20_st5|24_st5|32_st3|32_st4|32_st5)_csynth$')
last={}
for l in open("jobs/ledger.jsonl"):
    r=json.loads(l); last[r["tag"]]=r
ids=[r["job"] for t,r in last.items() if M.search(t) and not r.get("failed")]
out=subprocess.run(["squeue","-h","-j",",".join(ids)],capture_output=True,text=True).stdout if ids else ""
print(len(out.splitlines()))
PY
)
  [ "$live" -eq 0 ] && { echo "$(date) LUD long csynth: nothing left" >> lud/csynth_escalate.log; exit 0; }
  sleep 1800
done
