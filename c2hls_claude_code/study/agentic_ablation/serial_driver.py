#!/usr/bin/env python3
"""Run the Claude jobs of the agentic ablation one at a time, and survive Claude usage limits.

Plain loop, no LLM. For each tag in QUEUE order (r1 of every rung, then r2, then r3):
  1. Probe the plain shim with a tiny request; while Claude answers "hit your ... limit", sleep until the reset time it
     names (+5 min), or 30 min when no time is given, and probe again.
  2. Submit the tag through study/jobs/ledger.py (same --export as submit.sh; no singleton slots) and wait for it to end.
  3. A shim call in that window that failed on a usage limit (calls.jsonl error text, or the result event of an agent
     transcript) invalidates the run: its run dir moves to runs/_invalid_usage_limit/, the driver waits for the reset,
     and the same tag is rerun.
     TIMEOUT: run dir to runs/_timeout/, rerun at the next wall-time tier (1 h -> 3 h -> 6 h), then count it as failed.
     OUT_OF_MEMORY: rerun once at --mem=192G. Anything else is final.
State: driver_state.json; log: driver.log. Restarting the driver resumes from the state file.
"""
import glob, json, re, shutil, subprocess, sys, time, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code")
HERE = CC / "study/agentic_ablation"
RUNS = HERE / "runs"
STATE = HERE / "driver_state.json"
LOG = HERE / "driver.log"
LEDGER = CC / "study/jobs/ledger.py"
TIERS = ["01:00:00", "03:00:00", "06:00:00"]
SHIM_LOGS = [CC / "logs/shim_agent_abl_20261015", CC / "logs/shim_20261013T1130"]
PLAIN_SHIM = "http://login5:38585/v1"  # login3 rebooted 2026-10-19
TZ = ZoneInfo("America/Vancouver")
LIMIT_RE = re.compile(r"hit your (session |usage |weekly )?limit|usage limit|rate limit reached|limit reached", re.I)
RESET_RE = re.compile(r"resets\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.I)
RUNGS = ["zs", "zsk", "noskills", "skills", "dspredo", "refneu", "refnf", "refv2", "win7080"]
QUEUE = [t for r in (1, 2, 3) for t in (f"pl_zs_cc_r{r}", f"pl_zsk_cc_r{r}", *[f"ag_{v}_r{r}" for v in RUNGS])]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def base_argv(tag):
    """The sbatch argv submit.sh used for this tag, minus slot name, singleton dependency and time."""
    recs = [json.loads(l) for l in (CC / "study/jobs/ledger.jsonl").read_text().splitlines() if l.strip()]
    rec = [r for r in recs if r["tag"] == tag][0]
    out = []
    for a in rec["argv"]:
        if a.startswith("--dependency") or a.startswith("--time"):
            continue
        out.append(f"--job-name={tag}" if a.startswith("--job-name") else a)
    return out


def reset_wait(text):
    """Seconds to sleep for a usage-limit message: until the named reset time + 5 min, else 30 min."""
    m = RESET_RE.search(text or "")
    if not m:
        return 1800
    h, mi, ap = int(m[1]) % 12, int(m[2] or 0), m[3].lower()
    h += 12 if ap == "pm" else 0
    now = datetime.now(TZ)
    t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return max(300, (t - now).total_seconds() + 300)


def probe():
    """None when Claude answers; else the limit message."""
    body = json.dumps({"model": "claude-opus-5-5", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "Reply with the single word OK."}]}).encode()
    req = urllib.request.Request(PLAIN_SHIM + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        return None
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        return msg if LIMIT_RE.search(msg) else None   # other errors are not a usage limit
    except Exception as e:
        log(f"probe: shim unreachable ({e}); waiting 10 min")
        time.sleep(600)
        return probe()


def wait_until_usable(first_msg=None):
    msg = first_msg or probe()
    while msg:
        s = reset_wait(msg)
        log(f"usage limit: {msg[:160]!r}; sleeping {s / 60:.0f} min")
        time.sleep(s)
        msg = probe()


def limit_hits(since):
    """Usage-limit messages from shim calls that started at or after `since` (epoch seconds)."""
    hits = []
    for d in SHIM_LOGS:
        cj = d / "calls.jsonl"
        if cj.exists():
            for l in cj.read_text().splitlines():
                try:
                    c = json.loads(l)
                except ValueError:
                    continue
                t = time.mktime(time.strptime(c["t"], "%Y-%m-%dT%H:%M:%S"))
                if t >= since and c.get("is_error") and LIMIT_RE.search(str(c.get("error"))):
                    hits.append(str(c.get("error")))
        for call in d.glob("call_*"):
            if call.stat().st_mtime < since:
                continue
            tr = call / "agent_transcript.jsonl"
            if tr.exists():
                for l in tr.read_text().splitlines()[-5:]:
                    if '"type":"result"' in l.replace(" ", "") and LIMIT_RE.search(l):
                        hits.append(l[:300])
            rp = call / "reply.txt"
            if rp.exists() and rp.stat().st_size < 2000 and LIMIT_RE.search(rp.read_text()):
                hits.append(rp.read_text()[:300])
    return hits


def move_runs(tag, sub):
    dst = RUNS / sub
    dst.mkdir(parents=True, exist_ok=True)
    for d in glob.glob(str(RUNS / f"*_{tag}_2026*")):
        shutil.move(d, dst / Path(d).name)


def sacct_state(job):
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job, "-o", "State"], capture_output=True, text=True).stdout
    return (out.split() or ["?"])[0]


def run_tag(tag, st):
    rec = st.setdefault(tag, {"tier": 0 if tag.startswith("pl_") else 1, "mem": None, "tries": 0})
    while True:
        wait_until_usable()
        argv = base_argv(tag)
        if rec["mem"]:
            argv = [a for a in argv if not a.startswith("--mem")] + [f"--mem={rec['mem']}"]
        since = time.time()
        out = subprocess.run([sys.executable, str(LEDGER), "submit", tag, "--move-glob", str(RUNS / f"*_{tag}_*"),
                              "--tier", str(rec["tier"]), "--"] + argv + ["sbatch_flash_cc.sh"],
                             cwd=CC, capture_output=True, text=True)
        job = (out.stdout.split() or ["", ""])[1]
        if not job.isdigit():
            log(f"{tag}: submit failed: {out.stdout} {out.stderr}; retry in 10 min")
            time.sleep(600)
            continue
        rec["tries"] += 1
        rec["job"] = job
        save_state(st)
        log(f"{tag}: submitted job {job} at {TIERS[rec['tier']]} (try {rec['tries']})")
        state = "PENDING"
        while state in ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "?"):
            time.sleep(120)
            state = sacct_state(job)
        hits = limit_hits(since)
        if hits:
            log(f"{tag}: job {job} {state}, but {len(hits)} Claude call(s) hit the usage limit; run invalid, will rerun")
            move_runs(tag, "_invalid_usage_limit")
            wait_until_usable(hits[-1])
            continue
        if state.startswith("TIMEOUT"):
            move_runs(tag, "_timeout")
            if rec["tier"] >= 2:
                rec["final"] = "failed: timeout at 6 h"
                log(f"{tag}: TIMEOUT at 6 h; counted as failed")
                return
            rec["tier"] += 1
            log(f"{tag}: TIMEOUT; rerun at {TIERS[rec['tier']]}")
            continue
        if state.startswith("OUT_OF_MEMORY") and not rec["mem"]:
            move_runs(tag, "_oom")
            rec["mem"] = "192G"
            log(f"{tag}: OUT_OF_MEMORY; rerun at 192G")
            continue
        rec["final"] = state
        log(f"{tag}: job {job} ended {state}")
        return


def main():
    st = load_state()
    log(f"driver start; {sum(1 for t in QUEUE if 'final' not in st.get(t, {}))} of {len(QUEUE)} tags to run")
    for tag in QUEUE:
        if "final" in st.get(tag, {}):
            continue
        run_tag(tag, st)
        save_state(st)
    log("driver done")


if __name__ == "__main__":
    main()
