#!/usr/bin/env python3
"""SODA target results: one row per config. csynth (tripcount copy, designs_tc/<cfg>/csynth) gives latency estimate +
resources; cosim (designs/<cfg>/cosim) gives cycles and SODA's golden verdict.  collect_soda.py [--csv OUT]"""
import csv, json, re, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def csynth(d):
    x = d / "csynth.xml"
    if not x.exists():
        return {}
    r = ET.parse(x).getroot()
    g = lambda p: (r.find(p).text if r.find(p) is not None else None)
    lat = g(".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")
    res = {k.lower(): g(f".//AreaEstimates/Resources/{k}") for k in ("DSP", "LUT", "FF", "BRAM_18K", "URAM")}
    return dict(cs_lat=lat, **res)


def cosim(d):
    out = {}
    for rpt in d.glob("*cosim.rpt"):
        m = re.search(r"\|\s*Verilog\|\s*(\w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", rpt.read_text())
        if m:
            out = dict(cosim_status=m[1], cosim_cycles=int(m[4]))
    g = d / "vitis_grep.log"
    if g.exists():
        t = g.read_text()
        out["golden"] = "PASS" if "INFO: PASS" in t else ("FAIL" if "INFO: FAIL" in t else "")
    return out


def main():
    rows = []
    for d in sorted((HERE / "designs").iterdir()):
        c = json.loads((d / "config.json").read_text())
        r = dict(cfg=d.name, kernel=c["kernel"], unroll=c["unroll"], cr=c["cr"])
        r.update(csynth(HERE / "designs_tc" / d.name / "csynth"))
        r.update(cosim(d / "cosim"))
        rows.append(r)
    cols = ["cfg", "kernel", "unroll", "cr", "cs_lat", "dsp", "lut", "ff", "bram_18k", "uram", "cosim_cycles", "cosim_status", "golden"]
    w = csv.DictWriter(open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout, cols)
    w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
