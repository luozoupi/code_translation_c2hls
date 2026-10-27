#!/usr/bin/env python3
"""LightningSim FRP rule variants on one Vitis solution (trace resolved once, each variant is a re-execution).

  frp_eval.py SOLUTION_DIR OUT.json NPTR [COSIM]
Variants (fifo_raw_extra / fifo_depths given to the installed fork, which already has fifo_raw_extra):
  base     : no FRP rule, depths from the trace (pragma)
  v1       : +1 on every FIFO an FRP pipeline reads (the 2026-10-26 rule)
  v2       : +r on every FIFO an FRP pipeline reads at stage r, and every FIFO depth taken from the RTL
  v2_nodep : +r, pragma depths
  dep      : RTL depths only
Mapped = P*ceil((end-14)/P)+K, P = 6+NPTR, K = 27+12*NPTR. LSIM_FRP=0 is set so the installed rule stays out of the way.
"""
import asyncio, dataclasses, json, math, os, sys
from pathlib import Path

os.environ["LSIM_FRP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frp_rtl import analyse  # noqa: E402
from lightningsim.model import Solution  # noqa: E402
from lightningsim.runner import Runner  # noqa: E402

sol, out, nptr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cosim = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("", "None") else None
rtl = analyse(sol)
trace = asyncio.run(Runner(Solution(sol)).run())
p = trace.params


def match(lname, names):
    n = lname.replace(".", "_")
    if n in names:
        return n
    hits = [x for x in names if n.endswith(x) or x.endswith(n)]
    return max(hits, key=len) if hits else None


frp, dep, unmatched = {}, {}, []
for f in trace.fifos:
    m = match(f.name, rtl["fifos"])
    if m is None:
        unmatched.append(f.name)
        continue
    dep[f.id] = rtl["fifos"][m]["depth"]
    if m in rtl["frp_reads"]:
        frp[f.id] = rtl["frp_reads"][m]
P, K = 6 + nptr, 27 + 12 * nptr


def run(extra=None, depths=None):
    q = dataclasses.replace(p, fifo_raw_extra=extra or {}, fifo_depths={**p.fifo_depths, **(depths or {})})
    try:
        e = trace.compiled.execute(q).top_module.end
    except Exception as x:   # e.g. "deadlock detected" with one variant's depths
        return {"end": None, "mapped": None, "diff": None, "error": str(x)[:200]}
    mp = P * math.ceil((e - 14) / P) + K
    return {"end": e, "mapped": mp, "diff": mp - cosim if cosim else None}


res = {"cosim": cosim, "nptr": nptr, "fifos": len(trace.fifos), "unmatched": unmatched,
       "frp_fifos": {f.name: frp[f.id] for f in trace.fifos if f.id in frp},
       "depth_changed": {f.name: [p.fifo_depths.get(f.id), dep[f.id]] for f in trace.fifos
                         if f.id in dep and p.fifo_depths.get(f.id) != dep[f.id]}}
res["base"] = run()
res["v1"] = run({i: 1 for i in frp})
res["v2"] = run(frp, dep)
res["v2_nodep"] = run(frp)
res["dep"] = run(None, dep)
out.write_text(json.dumps(res, indent=1) + "\n")
print(out.name, {k: res[k].get("error") or (res[k]["diff"] if cosim else res[k]["end"]) for k in ("base", "v1", "v2", "v2_nodep", "dep")},
      "frp", len(frp), "depth changed", len(res["depth_changed"]), "unmatched", len(unmatched))
