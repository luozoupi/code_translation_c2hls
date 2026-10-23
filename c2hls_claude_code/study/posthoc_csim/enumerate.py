#!/usr/bin/env python3
"""List designs that could change a result under the 80% LUT cap if they were csim'd: synthesized in-session, never csim'd
(no byte-identical csim-passed kernel in that run), under the LUT cap and the DSP total, faster than the run's counted best.
Also lists csim-verified in-session designs of agent-backend runs that beat the run's final (the symmetry check).
Writes candidates.json."""
import json, glob, os, re
from datetime import datetime, timezone
from pathlib import Path
CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code"); S = CC / "study"
LUT, DSP = 1303680, 9024; CAP = 0.8 * LUT
WIN = (6317, 7219)
out = []

def ts(p):
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)

# Claude Code on its own
for w in sorted(glob.glob(str(S / "agentfw/runs/mm1024_*"))):
    w = Path(w)
    if w.name.startswith("_") or not (w / "ended_at").exists():
        continue
    end = datetime.fromisoformat((w / "ended_at").read_text().strip().replace("Z", "+00:00"))
    window = "hlsm" in w.name
    ok = {}
    for c in w.glob("exp/*/csim_summary.txt"):
        if "correct=True" in c.read_text() and (c.parent / "kernel.cpp").exists():
            ok[(c.parent / "kernel.cpp").read_text()] = c.parent.name
    cands = []
    for s in w.glob("exp/*/csynth_summary.json"):
        try: d = json.loads(s.read_text())
        except Exception: continue
        if not d.get("success") or not (s.parent / "kernel.cpp").exists(): continue
        r = d["report"]; wc = r.get("latency_cycles_worst") or r.get("latency_cycles")
        fits = (r.get("lut") or 0) <= CAP and (r.get("dsp") or 0) <= DSP and (not window or WIN[0] <= (r.get("dsp") or 0) <= WIN[1])
        cands.append(dict(name=s.parent.name, worst=wc, dsp=r.get("dsp"), lut_pct=round(100 * (r.get("lut") or 0) / LUT, 1),
                          fits=fits, verified=(s.parent / "kernel.cpp").read_text() in ok, in_session=ts(s) <= end,
                          kernel=str(s.parent / "kernel.cpp")))
    best = min((c["worst"] for c in cands if c["fits"] and c["verified"] and c["in_session"]), default=None)
    for c in cands:
        if c["fits"] and not c["verified"] and c["in_session"] and (best is None or c["worst"] < best):
            out.append(dict(side="claude_code_on_its_own", run=w.name, counted_best=best, **c))

def worst_of(c):
    """Worst-case latency of a session check, from its Vitis report (the check's .txt gives the best case)."""
    x = next((c.parent / c.name.replace(".txt", "_work")).glob("hls_proj/*/syn/report/csynth.xml"), None)
    m = re.search(r"<Worst-caseLatency>(\d+)<", x.read_text()) if x else None
    return int(m.group(1)) if m else None


# HLSmith with an agent backend: the sessions' own checks
rows = json.loads((S / "exp_hlsmith/cost_table/rows.json").read_text())
for x in rows:
    if not x.get("agent_dirs") or not x.get("ctrl_path"): continue
    mr = next(Path(x["ctrl_path"]).parent.glob("*_multistep_results.json"), None)
    if not mr: continue
    fr = json.loads(mr.read_text()).get("final_report") or {}
    fw = fr.get("latency_cycles_worst") or fr.get("latency_cycles")
    counted = fw if (fr.get("lut") or 0) <= CAP else None
    for sd in x.get("shim_dirs") or []:
        for ad in x["agent_dirs"]:
            ck = CC / "logs" / sd / ad / "workspace/checks"
            if not ck.is_dir(): continue
            okk = {(c.parent / c.name.replace(".txt", "_work") / "kernel.cpp").read_text() for c in ck.glob("*_csim.txt")
                   if "PASSED" in c.read_text() and (c.parent / c.name.replace(".txt", "_work") / "kernel.cpp").exists()}
            for c in ck.glob("*_csynth.txt"):
                k = c.parent / c.name.replace(".txt", "_work") / "kernel.cpp"
                f = dict(re.findall(r"^\s+(latency_cycles|dsp|lut): (\d+)", c.read_text(), re.M))
                if not (k.exists() and f.get("lut")): continue
                wc, dsp, lut = worst_of(c) or int(f["latency_cycles"]), int(f["dsp"]), int(f["lut"])
                fits = lut <= CAP and dsp <= DSP
                ver = k.read_text() in okk
                if fits and (counted is None or wc < counted):
                    out.append(dict(side="agent_backend", run=x["tag"], call=ad, counted_best=counted, final=fw,
                                    name=c.name, worst=wc, dsp=dsp, lut_pct=round(100 * lut / LUT, 1), fits=True,
                                    verified=ver, in_session=True, kernel=str(k)))
Path(__file__).with_name("candidates.json").write_text(json.dumps(out, indent=1))
for o in out:
    print(o["side"][:5], o["run"][:44], o.get("call", ""), o["name"], o["worst"], o["dsp"], o["lut_pct"], "verified" if o["verified"] else "UNVERIFIED", "best", o["counted_best"])
