#!/usr/bin/env python3
"""Why LightningSim reports a deadlock (a cycle in its event graph) on a design whose cosim passes.

  dl_diag.py SOLUTION_DIR OUT.json NPTR [COSIM]      (same argv as frp_eval.py)
1. per FIFO: writes, reads and depth in the trace (writes != reads leaves items behind or reads nothing);
2. all depths huge: if that still deadlocks the cycle is not a FIFO-capacity one;
3. else one FIFO at a time made huge, and the depth each such FIFO needs (doubling, then bisection).
"""
import asyncio, dataclasses, json, math, os, sys
from pathlib import Path

os.environ["LSIM_FRP"] = "0"
from lightningsim.model import Solution  # noqa: E402
from lightningsim.runner import Runner  # noqa: E402

BIG = 1000   # with width 1 where the real FIFO is a shift register, so the read-after-write delay (type) is kept
sol, out, nptr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cosim = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("", "None") else None
trace = asyncio.run(Runner(Solution(sol)).run())
p = trace.params
P, K = 6 + nptr, 27 + 12 * nptr


def is_srl(i):
    d, w = p.fifo_depths.get(i), p.fifo_widths.get(i)
    return d is None or w is None or d <= 2 or w * d <= 1024


def params(depths):
    """depths overridden; a FIFO that is a shift register at its real depth gets width 1 so it stays one."""
    widths = {i: 1 for i, d in depths.items() if d is not None and is_srl(i)}
    return dataclasses.replace(p, fifo_depths={**p.fifo_depths, **depths}, fifo_widths={**p.fifo_widths, **widths})


def run(depths):
    try:
        e = trace.compiled.execute(params(depths)).top_module.end
        return e, P * math.ceil((e - 14) / P) + K
    except Exception as x:
        return None, str(x)[:80]


res = {"cosim": cosim, "base": run({})}
io = {}
try:
    sim = trace.compiled.execute(params({f.id: BIG for f in trace.fifos}))
    io = {f.id: (len(v.writes), len(v.reads), v.get_observed_depth()) for f, v in sim.fifo_io.items()}
except Exception as x:
    res["all_big_error"] = str(x)[:80]
res["fifos"] = {f.name: {"depth": p.fifo_depths.get(f.id), "writes": io.get(f.id, (None,))[0],
                         "reads": io.get(f.id, (None, None))[1],
                         "observed_depth_unbounded": io.get(f.id, (None, None, None))[2]} for f in trace.fifos}
res["all_big"] = run({f.id: BIG for f in trace.fifos})
res["unbalanced"] = {n: v for n, v in res["fifos"].items() if v["writes"] != v["reads"]}
fix = {}
if res["base"][0] is None and res["all_big"][0] is not None:
    for f in trace.fifos:
        if run({f.id: BIG})[0] is not None:
            lo, hi = p.fifo_depths.get(f.id) or 1, 2 * (p.fifo_depths.get(f.id) or 1)
            while run({f.id: hi})[0] is None:
                lo, hi = hi, hi * 2
            while hi - lo > 1:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if run({f.id: mid})[0] is not None else (mid, hi)
            fix[f.name] = {"depth": p.fifo_depths.get(f.id), "needs": hi, "result": run({f.id: hi})}
res["single_fifo_fixes"] = fix
out.write_text(json.dumps(res, indent=1) + "\n")
print(out.name, "base", res["base"], "all_big", res["all_big"], "unbalanced", len(res["unbalanced"]),
      "single fixes", {k: v["needs"] for k, v in fix.items()})
