#!/bin/bash
# escalate_loop.sh REGEX LOG [SECONDS]: plain loop (no LLM). Every SECONDS (default 1800) run
# `ledger.py escalate --match REGEX`; exit when no matching, not-failed ledger job is queued or running.
cd /scratch/asa582/workspaces/c2hls_claude_code/study
RX=$1; LOG=$2; SEC=${3:-1800}
while true; do
  python3 jobs/ledger.py escalate --match "$RX" >> "$LOG" 2>&1
  live=$(python3 - "$RX" <<'PY'
import json,re,subprocess,sys
rx=re.compile(sys.argv[1]); last={}
for l in open("jobs/ledger.jsonl"):
    r=json.loads(l); last[r["tag"]]=r
ids=[r["job"] for t,r in last.items() if rx.search(t) and not r.get("failed")]
print(len(subprocess.run(["squeue","-h","-j",",".join(ids)],capture_output=True,text=True).stdout.splitlines()) if ids else 0)
PY
)
  [ "$live" -eq 0 ] && { echo "$(date) $RX: nothing left" >> "$LOG"; exit 0; }
  sleep "$SEC"
done
