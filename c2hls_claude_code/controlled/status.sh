#!/bin/bash
# Read-only status of the controlled DeepSeek-vs-Claude campaign.
CC=/scratch/asa582/workspaces/c2hls_claude_code; R=$CC/runs_controlled
date
echo "== endpoints"; echo "shim: $(curl -s -m 10 $(cat $CC/shim_url.txt)/health)"; echo "ds_proxy: $(curl -s -m 10 $(cat $CC/controlled/ds_proxy_url.txt)/health)"
echo "== slurm"; sacct -X -n -P -j $(awk '{print $2}' $R/jobs.txt | paste -sd,) -o JobName,JobID,State,Elapsed | column -t -s'|'
echo "== runs (cell: flash ok | latency | dsp | ctrl calls ok/trunc/empty/txerr)"
for c in $R/*/autosa_mm; do [ -d "$c" ] || continue; n=$(basename $(dirname $c))
  res=$(python3 - "$c" <<'PY'
import json,sys,os
c=sys.argv[1]; out=[]
try:
    m=json.load(open(f"{c}/autosa_mm_multistep_results.json")); f=next((s for s in m.get("steps",[]) if s.get("step_name")=="flash"),{})
    s=json.load(open(f"{c}/autosa_mm_selected_report.json")); r=s.get("report") or s
    out.append(f"flash_ok={f.get('success')} lat={max(r.get('latency_cycles') or 0, r.get('latency_cycles_worst') or 0):,} dsp={r.get('dsp')}")
except Exception: out.append("running")
k={"ok":0,"truncated":0,"empty":0,"transport_error":0}
p=f"{c}/ctrl_llm_calls.jsonl"
if os.path.exists(p):
    for l in open(p):
        k[json.loads(l).get("outcome","?")]=k.get(json.loads(l).get("outcome","?"),0)+1
out.append("calls "+"/".join(str(k[x]) for x in ("ok","truncated","empty","transport_error")))
print(" | ".join(out))
PY
)
  echo "$n: $res"; done
echo "== errors in slurm logs (last 3 per job)"
for j in $(awk '{print $2}' $R/jobs.txt); do e=$CC/logs/flash_cc-$j.err; [ -f $e ] && { x=$(grep -hE "Traceback|Error|ERROR|reject on attempt|Synthesis failed|timed out" $e | tail -3 | cut -c1-160); [ -n "$x" ] && echo "[$j] $x"; }; done
echo "== proxy: count of HTTP 200 calls, then any non-200 lines"; grep -c "\"status\": 200" $CC/controlled/ds_proxy_calls.jsonl; grep -v '"status": 200' $CC/controlled/ds_proxy_calls.jsonl | tail -3 | cut -c1-200
echo "== quota"; diskusage_report 2>/dev/null | grep -i scratch
