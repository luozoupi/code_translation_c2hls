#!/usr/bin/env python3
"""Per-event comparison of LightningSim m_axi times with the kernel-side handshakes in a cosim VCD.

  io_compare.py AXI_TIMING.json LSIM_IO.json

AXI_TIMING.json comes from parse_axi_vcd.py (ports.<bundle>.user.{readreqs,reads,writereqs,writes,writeresps},
cycles from ap_start); LSIM_IO.json from lsim_io.py. Events are matched by index per interface and kind; prints
the distribution of (lightningsim - cosim) and where it changes, which localises a timing-model error.
"""
import json, sys
from collections import Counter


def main():
    vcd = json.load(open(sys.argv[1]))
    ls = json.load(open(sys.argv[2]))
    print(f"end: lsim={ls['end']} dut_done={vcd['ap_done']} diff={ls['end'] - vcd['ap_done']:+d}")
    for name, io in ls["interfaces"].items():
        port = vcd["ports"].get(name)
        if not port or "user" not in port:
            print(f"{name}: not in VCD")
            continue
        for kind in ("readreqs", "reads", "writereqs", "writes", "writeresps"):
            a, b = io[kind], port["user"][kind]
            if not a and not b:
                continue
            n = min(len(a), len(b))
            d = [a[i] - b[i] for i in range(n)]
            if not d:
                print(f"  {name}.{kind}: lsim n={len(a)} cosim n={len(b)}")
                continue
            hist = Counter(d).most_common(4)
            steps = [(i, d[i - 1], d[i]) for i in range(1, n) if d[i] != d[i - 1]][:6]
            print(f"  {name}.{kind}: n lsim={len(a)} cosim={len(b)}  diff first={d[0]:+d} last={d[-1]:+d} "
                  f"min={min(d):+d} max={max(d):+d} common={hist}  changes@={steps}")


if __name__ == "__main__":
    main()
