#!/usr/bin/env python3
"""Occupancy of every FIFO in a cosim VCD: writes, reads, max occupancy, cycle of the first write/read.

  vcd_occ.py ALL.vcd.gz [NAME_REGEX]
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fifo_cmp import vcd_fifo_events  # noqa: E402

ev = vcd_fifo_events(sys.argv[1])
rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
for name in sorted(ev):
    if not rx.search(name):
        continue
    w, r = ev[name]
    occ = mx = 0
    i = j = 0
    while i < len(w) or j < len(r):
        if j >= len(r) or (i < len(w) and w[i] <= r[j]):
            occ += 1; i += 1
        else:
            occ -= 1; j += 1
        mx = max(mx, occ)
    print(name, "w", len(w), "r", len(r), "maxocc", mx, "first_w", w[0] if w else None, "first_r", r[0] if r else None,
          "last_r", r[-1] if r else None)
