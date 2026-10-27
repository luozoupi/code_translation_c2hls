#!/usr/bin/env python3
"""The installed LightningSim (its own FRP handling on) against cosim, on one solution.

  lsim_check.py SOLUTION_DIR OUT.json NPTR [COSIM]      (same argv as frp_eval.py)
"""
import asyncio, json, math, sys
from pathlib import Path

from lightningsim.model import Solution
from lightningsim.runner import Runner

sol, out, nptr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cosim = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("", "None") else None
P, K = 6 + nptr, 27 + 12 * nptr
res = {"cosim": cosim}
try:
    trace = asyncio.run(Runner(Solution(sol)).run())
    e = trace.compiled.execute(trace.params).top_module.end
    mp = P * math.ceil((e - 14) / P) + K
    res.update(end=e, mapped=mp, diff=(mp - cosim) if cosim else None,
               err_pct=round(100 * (mp - cosim) / cosim, 4) if cosim else None)
except Exception as x:
    res["error"] = str(x)[:200]
out.write_text(json.dumps(res, indent=1) + "\n")
print(out.name, res)
