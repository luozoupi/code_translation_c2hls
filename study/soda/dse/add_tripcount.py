#!/usr/bin/env python3
"""csynth-only copy of every SODA design with `loop_tripcount` on its data-driven loops, so csynth prints a latency.

  add_tripcount.py  -> designs_tc/<cfg>/{kernel.cpp, host.cpp, frt.h, config.json}
loop_tripcount changes only the csynth report, not the RTL. Counts from SODA's own host formulas, single tile:
  stencil modules  `for (bool enable = true; enable;)` : ceil((N + stencil_distance) / unroll)  (host: tile_data_count)
  BurstRead/Write  `for (uint64_t i = 0; i < data_num...)`: coalesced_data_num = ceil((N + stencil_distance) / epc),
                   epc = burst / element bits (largest over ports, one template instance per element type is not split)
The latency csynth prints is an estimate, like the LLM side's csynth number; cycles for the paper come from cosim.
"""
import json, math, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None   # add_tripcount.py [cfg,cfg,...]
for d in sorted((HERE / "designs").iterdir()):
    if only and d.name not in only:
        continue
    c = json.loads((d / "config.json").read_text())
    host = (d / "host.cpp").read_text()
    n, dist, k = math.prod(c["sizes"]), c["stencil_distance"], c["unroll"]
    widths = [int(w) for w in re.findall(r"constexpr int kWidth_\w+ = (\d+);", host)]
    t_mod = math.ceil((n + dist) / k)
    t_dram = max(math.ceil((n + dist) / (c.get("burst", 512) // w)) for w in widths)
    ker = (d / "kernel.cpp").read_text()
    ker, a = re.subn(r"(for \(bool enable = true; enable;\)\s*\{\s*\n#pragma HLS pipeline II = \d+)",
                     rf"\1\n#pragma HLS loop_tripcount min = {t_mod} max = {t_mod}", ker)
    ker, b = re.subn(r"(for \(uint64_t i = 0; i < data_num;[^\n]*\n#pragma HLS pipeline II = \d+)",
                     rf"\1\n#pragma HLS loop_tripcount min = {t_dram} max = {t_dram}", ker)
    left = len(re.findall(r"for \(bool enable = true; enable;\)", ker)) - a
    assert a > 0 and b == 2 and left == 0, (d.name, a, b, left)
    o = HERE / "designs_tc" / d.name
    o.mkdir(parents=True, exist_ok=True)
    (o / "kernel.cpp").write_text(ker)
    for f in ("host.cpp", "frt.h"):
        shutil.copy(d / f, o / f)
    (o / "config.json").write_text(json.dumps(dict(c, tripcount_module=t_mod, tripcount_dram=t_dram), indent=1) + "\n")
    print(f"{d.name}: modules={a} t_mod={t_mod} t_dram={t_dram}")
