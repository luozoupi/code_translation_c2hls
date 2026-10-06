#!/usr/bin/env bash
# Overnight health check for HLSFactory Sonnet/Haiku/Luna/Grok campaigns.
# Prints a compact status report; exits non-zero if actionable failures found.
set -euo pipefail
ROOT="/scratch/hpc-prf-llmfpga/asa582/projects/c2hls"
cd "${ROOT}"
PY="${C2HLS_PYTHON:-python3}"
REPORT="/tmp/c2hls_overnight_watch_$(date -u +%Y%m%d).log"
STAMP_NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  echo "===== ${STAMP_NOW} overnight watch ====="
  "${PY}" - <<'PY'
import json, subprocess, re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/scratch/hpc-prf-llmfpga/asa582/projects/c2hls")
PATTERNS = [
    "batch_parallel_hlsfactory_sonnet5_*",
    "batch_parallel_hlsfactory_haiku45_*",
    "batch_parallel_hlsfactory_luna_*",
    "batch_parallel_hlsfactory_grok45_*",
]
PREFIXES = ("bphfs", "bphfn", "bphfh", "bphfl", "bphfg")  # sonnet/haiku/luna/grok-ish

def squeue():
    out = subprocess.check_output(["squeue", "-u", "haqc2", "-h", "-o", "%i %j %T %M %R"], text=True)
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        jid, name, state, elapsed = parts[0], parts[1], parts[2], parts[3]
        reason = parts[4] if len(parts) > 4 else ""
        if any(name.startswith(p) for p in ("bphfs", "bphfn", "bphfh", "bphfl", "bphfg")):
            rows.append({"id": jid, "name": name, "state": state, "elapsed": elapsed, "reason": reason})
    return rows

def sacct_state(jid):
    try:
        out = subprocess.check_output(["sacct", "-j", jid, "-n", "-X", "-o", "State"], text=True).strip().split()
        return out[0] if out else "?"
    except Exception:
        return "?"

# Active campaigns: recent stamps with campaign.json still running or incomplete
camps = []
for pat in PATTERNS:
    for p in sorted(ROOT.joinpath("artifacts/pc2").glob(pat)):
        cj = p / "campaign.json"
        if not cj.is_file():
            continue
        try:
            d = json.loads(cj.read_text())
        except Exception:
            continue
        status = d.get("campaign_status")
        # Keep recent / non-complete, or any with active post watcher
        if status in {"complete", "completed", "done"} and not (p / "CAMPAIGN_COMPLETE").exists():
            pass
        if status in {"complete", "completed", "done"}:
            # skip old completed unless stamp is today-ish
            if "20260727" not in p.name and "20260728" not in p.name:
                continue
            # still include today's completed for summary
        if "20260727" not in p.name and "20260728" not in p.name:
            # only watch stamps from this dual-launch wave + haiku top5
            if not any(x in p.name for x in [
                "20260727_071728", "20260727_120855", "20260727_135536",
                "20260727_143552", "20260727_143553", "20260727_143555",
                "20260727_144000", "20260727_144001",
            ]):
                continue
        camps.append((p, d))

q = squeue()
by_pref = defaultdict(list)
for r in q:
    by_pref[r["name"].split("-")[0]].append(r)

print(f"queue rows={len(q)}")
for pref in sorted(by_pref):
    states = defaultdict(int)
    for r in by_pref[pref]:
        states[r["state"]] += 1
    print(f"  {pref}: " + ", ".join(f"{k}={v}" for k, v in sorted(states.items())))

# Proxies
proxies = {
    "sonnet_skills": ("18192", "claude"),
    "sonnet_noskills": ("18193", "claude"),
    "haiku_skills": ("18194", "claude"),
    "haiku_noskills": ("18195", "claude"),
    "luna_skills": ("18200", "openai"),
    "luna_noskills": ("18201", "openai"),
    "grok_skills": ("18210", "xai"),
    "grok_noskills": ("18211", "xai"),
}
proxy_fail = []
import urllib.request
for label, (port, kind) in proxies.items():
    url = f"http://login5:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            ok = resp.status == 200
    except Exception as exc:
        ok = False
        proxy_fail.append((label, port, str(exc)))
        print(f"PROXY FAIL {label} :{port} {exc}")
    else:
        print(f"PROXY OK   {label} :{port}")

issues = []
print("\n--- campaigns ---")
for p, d in sorted(camps, key=lambda x: x[0].name):
    name = p.name
    status = d.get("campaign_status")
    compute = d.get("compute_state")
    model = d.get("model")
    post = str(d.get("post_watcher_job_id") or "")
    watch = str(d.get("watch_job_id") or "")
    nbench = len((d.get("config") or {}).get("pilot", {}).get("benches") or [])
    jobs = d.get("compute_jobs") or []
    post_st = sacct_state(post) if post else "-"
    watch_st = sacct_state(watch) if watch else "-"
    # active synth for this stamp
    stamp = name.rsplit("_", 1)[-1] if "_" in name else ""
    # better stamp: last two underscore parts YYYYMMDD_HHMMSS
    m = re.search(r"(20\d{6}_\d{6})$", name)
    stamp = m.group(1) if m else ""
    synth_q = [r for r in q if stamp and stamp in r["name"] and "-synth-" in r["name"]]
    helper_q = [r for r in q if stamp and stamp in r["name"] and any(x in r["name"] for x in ("-watch", "-drain", "-coord", "-post"))]
    # helpers don't always include stamp in name for watch/post - match by job id
    for r in q:
        if r["id"] in {post, watch, str(d.get("drain_job_id") or ""), str(d.get("coord_job_id") or "")}:
            helper_q.append(r)

    print(f"{name}")
    print(f"  status={status} compute={compute} model={model} benches={nbench} synth_q={len(synth_q)} post={post}:{post_st}")

    if post and post_st not in {"RUNNING", "PENDING", "COMPLETING", "-"} and status not in {"complete", "completed", "done"}:
        issues.append({"kind": "post_dead", "campaign": str(p), "job": post, "state": post_st})
    if watch and watch_st not in {"RUNNING", "PENDING", "COMPLETING", "-"} and status not in {"complete", "completed", "done"}:
        issues.append({"kind": "watch_dead", "campaign": str(p), "job": watch, "state": watch_st})
    if compute in {"waiting_for_gpu", "submitted"} and status == "running" and not synth_q:
        # may be early; only flag if campaign older than ~10 min and still no synth
        issues.append({"kind": "no_synth", "campaign": str(p), "compute": compute})
    # failed synth recently
    for j in jobs:
        jid = str(j.get("slurm_job_id") or "")
        if not jid:
            continue
        st = sacct_state(jid)
        if st in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"}:
            # only if still running campaign and work pending
            issues.append({"kind": "synth_failed", "campaign": str(p), "job": jid, "state": st, "node": j.get("node_index")})

for label, port, err in proxy_fail:
    issues.append({"kind": "proxy_down", "label": label, "port": port, "error": err})

# Deduplicate no_synth noise: only keep if compute_jobs empty AND waiting
filtered = []
for iss in issues:
    if iss["kind"] == "no_synth":
        camp = Path(iss["campaign"])
        d = json.loads((camp / "campaign.json").read_text())
        if d.get("compute_jobs") and d.get("compute_state") == "submitted":
            # jobs registered but not in queue -> need resubmit
            iss["kind"] = "synth_missing"
            filtered.append(iss)
        elif d.get("compute_state") == "waiting_for_gpu" and d.get("external_llm"):
            # watch should submit; flag if watch running
            filtered.append(iss)
        else:
            filtered.append(iss)
    else:
        filtered.append(iss)

# Dedup synth_failed by campaign
seen = set()
final = []
for iss in filtered:
    key = (iss["kind"], iss.get("campaign"), iss.get("job"), iss.get("port"))
    if key in seen:
        continue
    seen.add(key)
    final.append(iss)

print("\n--- issues ---")
if not final:
    print("none")
else:
    for iss in final:
        print(json.dumps(iss))

out = ROOT / "artifacts/pc2/overnight_watch_last_issues.json"
out.write_text(json.dumps({"ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()), "issues": final}, indent=2) + "\n")
print(f"\nwrote {out} n_issues={len(final)}")
raise SystemExit(1 if final else 0)
PY
} 2>&1 | tee -a "${REPORT}"
exit ${PIPESTATUS[0]}
