#!/usr/bin/env python3
"""Job ledger for the study, with the wall-time escalation rule: 1 h -> 3 h -> 6 h -> failure.

  ledger.py submit TAG [--move-glob GLOB] [--tier N] -- sbatch-args... script [args]
        submits with --time=01:00:00 (tier 0, or tier N) and appends a record to ledger.jsonl
  ledger.py escalate [--match REGEX]
        for every latest record whose job ended in TIMEOUT: resubmit the same command at the next tier
        (03:00:00, then 06:00:00); before resubmitting, move dirs matching --move-glob into <dir>/../_timeout/.
        After a 6 h TIMEOUT the tag is marked failed (no more resubmits).
        OUT_OF_MEMORY: resubmit at the same wall-time tier with the next memory tier (192G, then 512G);
        an OOM at 512G is marked failed.
  ledger.py status
Records: {tag, job, tier, time, argv, cwd, move_glob, t}. Nothing is ever deleted.
"""
import glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
TIERS = ["01:00:00", "03:00:00", "06:00:00"]
MEM_TIERS = ["192G", "512G"]   # used after an OUT_OF_MEMORY; the first submit uses the script's own --mem


def records():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def append(rec):
    with open(LEDGER, "a") as fp:
        fp.write(json.dumps(rec) + "\n")


def sbatch(argv, tier, cwd, mem=None, tiers=None):
    args = [a for a in argv if not a.startswith("--time") and not (mem and a.startswith("--mem"))]
    extra = [f"--mem={mem}"] if mem else []
    out = subprocess.run(["sbatch", "--parsable", f"--time={(tiers or TIERS)[tier]}"] + extra + args, cwd=cwd,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"sbatch failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def states(jobs):
    if not jobs:
        return {}
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs), "-o", "JobID,State,Elapsed"],
                         capture_output=True, text=True).stdout
    return {l.split("|")[0]: (l.split("|")[1].split()[0], l.split("|")[2]) for l in out.splitlines() if "|" in l}


def latest():
    last = {}
    for r in records():
        last[r["tag"]] = r
    return last


def cmd_submit(av):
    tag = av[0]
    move = None
    rest = av[1:]
    if rest and rest[0] == "--move-glob":
        move, rest = rest[1], rest[2:]
    tier, tiers = 0, None
    if rest and rest[0] == "--tier":
        tier, rest = int(rest[1]), rest[2:]
    if rest and rest[0] == "--tiers":   # per-tag wall-time ladder, e.g. 12:00:00,24:00:00,2-00:00:00
        tiers, rest = rest[1].split(","), rest[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    cwd = os.getcwd()
    mem = None
    if rest and rest[0] == "--mem-override":
        mem, rest = rest[1], rest[2:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    job = sbatch(rest, tier, cwd, mem, tiers)
    rec = {"tag": tag, "job": job, "tier": tier, "time": (tiers or TIERS)[tier], "argv": rest, "cwd": cwd,
           "move_glob": move, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if tiers:
        rec["tiers"] = tiers
    if mem:
        rec["mem"] = mem
    append(rec)
    print(f"{tag} {job} {(tiers or TIERS)[tier]}")


def cmd_escalate(av=()):
    last = latest()
    if av and av[0] == "--match":   # only tags matching this regex (drivers that escalate their own tags stay untouched)
        rx = re.compile(av[1])
        last = {t: r for t, r in last.items() if rx.search(t)}
    st = states([r["job"] for r in last.values() if r.get("job")])
    for tag, r in last.items():
        if r.get("failed") or r.get("merged_into"):
            continue
        s, el = st.get(r["job"], ("?", ""))
        if s in ("FAILED", "NODE_FAIL") and el and el < "00:02:00" and r.get("startup_retries", 0) < 2:
            # died at start (e.g. output file could not be opened, node problem): same tier, at most twice
            job = sbatch(r["argv"], r["tier"], r["cwd"], r.get("mem"), r.get("tiers"))
            append({**r, "job": job, "prev_job": r["job"], "startup_retries": r.get("startup_retries", 0) + 1,
                    "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(f"{tag}: {r['job']} {s} after {el} -> {job} (startup retry)")
            continue
        if s == "OUT_OF_MEMORY":
            mt = r.get("mem_tier", -1) + 1
            if mt >= len(MEM_TIERS):
                append({**r, "failed": f"out of memory at {r.get('mem')}", "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
                print(f"{tag}: OUT_OF_MEMORY at {r.get('mem')} -> counted as failure")
                continue
            if r.get("move_glob"):
                for d in glob.glob(r["move_glob"]):
                    dest = Path(d).parent / "_oom"
                    dest.mkdir(exist_ok=True)
                    shutil.move(d, dest / Path(d).name)
            job = sbatch(r["argv"], r["tier"], r["cwd"], MEM_TIERS[mt], r.get("tiers"))
            append({**r, "job": job, "mem_tier": mt, "mem": MEM_TIERS[mt], "prev_job": r["job"],
                    "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(f"{tag}: {r['job']} OUT_OF_MEMORY -> {job} with --mem={MEM_TIERS[mt]}")
            continue
        if s != "TIMEOUT":
            continue
        tl = r.get("tiers") or TIERS
        if r["tier"] + 1 >= len(tl):
            append({**r, "failed": f"timeout at {tl[-1]}", "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(f"{tag}: TIMEOUT at {r['time']} -> counted as failure")
            continue
        if r.get("move_glob"):
            for d in glob.glob(r["move_glob"]):
                dest = Path(d).parent / "_timeout"
                dest.mkdir(exist_ok=True)
                shutil.move(d, dest / Path(d).name)
        tier = r["tier"] + 1
        job = sbatch(r["argv"], tier, r["cwd"], r.get("mem"), r.get("tiers"))
        append({**r, "job": job, "tier": tier, "time": tl[tier], "prev_job": r["job"],
                "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"{tag}: {r['job']} TIMEOUT at {r['time']} -> {job} at {tl[tier]}")


def cmd_status():
    last = latest()
    st = states([r["job"] for r in last.values()])
    for tag, r in sorted(last.items()):
        if r.get("merged_into"):
            continue
        s, el = st.get(r["job"], ("?", ""))
        print(f"{tag:34s} {r['job']:>9s} tier={r['time']} {(r.get('mem') or ''):5s} {s:10s} {el}" + (f"  FAILED: {r['failed']}" if r.get("failed") else ""))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"submit": lambda: cmd_submit(sys.argv[2:]), "escalate": lambda: cmd_escalate(sys.argv[2:]), "status": cmd_status}[cmd]()
