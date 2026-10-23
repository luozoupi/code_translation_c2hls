#!/usr/bin/env python3
"""The other half of the post-hoc rule: kernels a run csim'd (passed) in its session but never synthesized, on both sides.
A kernel counts as synthesized if any csynth of that run (session check, controller step/final, or post-hoc) has the
byte-identical text. Writes candidates_csynth.json."""
import json, glob
from pathlib import Path
CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code"); S = CC / "study"
out = []
# Claude Code on its own
for w in sorted(glob.glob(str(S / "agentfw/runs/mm1024_*"))):
    w = Path(w)
    if w.name.startswith("_") or not (w / "ended_at").exists():
        continue
    syn = set()
    for s in list(w.glob("exp/*/csynth_summary.json")) + list(w.glob("exp_post/*/csynth_summary.json")):
        if (s.parent / "kernel.cpp").exists():
            try:
                if json.loads(s.read_text()).get("success"): syn.add((s.parent / "kernel.cpp").read_text())
            except Exception: pass
    pend = {(p / "kernel.cpp").read_text() for p in w.glob("exp_post/*_csynth") if (p / "kernel.cpp").exists()}
    seen = set()
    for c in sorted(w.glob("exp/*/csim_summary.txt")):
        k = c.parent / "kernel.cpp"
        if "correct=True" in c.read_text() and k.exists():
            t = k.read_text()
            if t not in syn and t not in seen:
                seen.add(t)
                out.append(dict(side="claude_code_on_its_own", run=w.name, name=c.parent.name, kernel=str(k),
                                already_submitted=t in pend))
# HLSmith with an agent backend
rows = json.loads((S / "exp_hlsmith/cost_table/rows.json").read_text())
for x in rows:
    if not x.get("agent_dirs") or not x.get("ctrl_path"): continue
    rd = Path(x["ctrl_path"]).parent
    syn = {p.read_text() for p in rd.glob("*.cpp")}            # controller-synthesized kernels (final, flash_opt, selected, ...)
    cks = []
    for sd in x.get("shim_dirs") or []:
        for ad in x["agent_dirs"]:
            ck = CC / "logs" / sd / ad / "workspace/checks"
            if ck.is_dir(): cks.append((ad, ck))
    for ad, ck in cks:
        for c in ck.glob("*_csynth.txt"):
            k = c.parent / c.name.replace(".txt", "_work") / "kernel.cpp"
            if k.exists() and "csynth: OK" in c.read_text(): syn.add(k.read_text())
    seen = set()
    for ad, ck in cks:
        for c in sorted(ck.glob("*_csim.txt")):
            k = c.parent / c.name.replace(".txt", "_work") / "kernel.cpp"
            if "PASSED" in c.read_text() and k.exists():
                t = k.read_text()
                if t not in syn and t not in seen:
                    seen.add(t)
                    out.append(dict(side="agent_backend", run=x["tag"], call=ad, name=c.name.replace(".txt", ""), kernel=str(k),
                                    already_submitted=False))
Path(__file__).with_name("candidates_csynth.json").write_text(json.dumps(out, indent=1))
from collections import Counter
print(Counter((o["side"], o["already_submitted"]) for o in out))
for o in out: print(o["side"][:5], o["run"][:42], o.get("call", ""), o["name"], "pending" if o["already_submitted"] else "")
