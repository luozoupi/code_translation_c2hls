#!/usr/bin/env python3
"""Smallest set of FIFOs LightningSim needs deeper than the RTL to get through a design that cosim runs, and how much.

  dl_min.py SOLUTION_DIR OUT.json NPTR [COSIM]      (same argv as frp_eval.py; the installed FRP rules stay on)
Start with every FIFO deep (1000, width 1 if it is a shift register at its real depth, so its type is kept); give each
FIFO its real depth back in turn and keep it deep only if the real depth deadlocks. For each FIFO left, the smallest
depth that runs with the others of the set deep, and the result with all of them at those depths.
"""
import asyncio, dataclasses, json, math, sys
from pathlib import Path

from lightningsim.model import Solution
from lightningsim.runner import Runner

BIG = 1000
sol, out, nptr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cosim = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("", "None") else None
trace = asyncio.run(Runner(Solution(sol)).run())
p = trace.params
P, K = 6 + nptr, 27 + 12 * nptr
real = dict(p.fifo_depths)


def srl(i):
    d, w = real.get(i), p.fifo_widths.get(i)
    return d is None or w is None or d <= 2 or w * d <= 1024


def run(depths):
    widths = {i: 1 for i, d in depths.items() if d > 1024 // max(1, p.fifo_widths.get(i) or 1) and srl(i)}
    q = dataclasses.replace(p, fifo_depths={**real, **depths}, fifo_widths={**p.fifo_widths, **widths})
    try:
        e = trace.compiled.execute(q).top_module.end
        return e, P * math.ceil((e - 14) / P) + K
    except Exception as x:
        return None, str(x)[:60]


names = {f.id: f.name for f in trace.fifos}
res = {"cosim": cosim, "base": run({})}
deep = {i: BIG for i in names}
res["all_deep"] = run(deep)
if res["base"][0] is None and res["all_deep"][0] is not None:
    for i in sorted(names, key=lambda i: names[i]):
        trial = {k: v for k, v in deep.items() if k != i}
        if run(trial)[0] is not None:
            deep = trial
    need = {}
    for i in deep:
        lo, hi = real.get(i) or 1, BIG
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if run({**deep, i: mid})[0] is not None else (mid, hi)
        need[i] = hi
    res["needed"] = {names[i]: {"rtl": real.get(i), "needs": need[i]} for i in deep}
    res["with_needed"] = run(need)
out.write_text(json.dumps(res, indent=1) + "\n")
print(out.name, "base", res["base"], "all_deep", res["all_deep"], "set", len(res.get("needed", {})),
      "with_needed", res.get("with_needed"), list(res.get("needed", {}).items())[:8])
