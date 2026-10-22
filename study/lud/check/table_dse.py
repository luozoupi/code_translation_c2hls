#!/usr/bin/env python3
"""LUD DSE table: csim (AutoSA host + true-LU check of printed matrices), csynth latency and resources, PE count."""
import glob, re, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..", "dse")
CAP = dict(BRAM_18K=4032, DSP=9024, FF=2607360, LUT=1303680, URAM=960)
rows = []
for d in sorted(glob.glob(f"{ROOT}/n*_st?"), key=lambda p: (int(re.search(r"n(\d+)", p).group(1)), p)):
    name = os.path.basename(d); o = f"{d}/out"
    log = open(f"{o}/vitis_csim.log", errors="replace").read() if os.path.exists(f"{o}/vitis_csim.log") else ""
    m = re.findall(r"(Passed!|Failed with \d+ errors!)", log); csim = m[-1] if m else "-"
    src = open(f"{o}/src/kernel_kernel.cpp").read()
    pes = len(set(re.findall(r"\bPE_wrapper<\s*([0-9, ]+)>", src))) or len(set(re.findall(r"\bPE<\s*([0-9, ]+)>\s*\(", src)))
    rpt = f"{o}/hls_prj_csynth/solution1/syn/report/kernel0_csynth.rpt"
    r = dict(design=name, csim=csim, PEs=pes)
    if os.path.exists(rpt):
        t = open(rpt).read()
        lat = re.search(r"\|\s*Latency \(cycles\)[^\n]*\n[^\n]*\n[^\n]*\n\s*\|\s*([\d?]+)\|\s*([\d?]+)\|", t)
        if lat: r["lat_min"], r["lat_max"] = lat.group(1), lat.group(2)
        tot = re.search(r"\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", t)
        if tot: r.update(dict(zip(["BRAM", "DSP", "FF", "LUT", "URAM"], map(int, tot.groups()))))
    rows.append(r)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "dse_table.json"), "w"), indent=1)
print("| design | PEs | AutoSA host csim | lat min | lat max | DSP | LUT | FF | BRAM | URAM |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['design']} | {r['PEs']} | {r['csim']} | {r.get('lat_min','-')} | {r.get('lat_max','-')} | {r.get('DSP','-')} | {r.get('LUT','-')} | {r.get('FF','-')} | {r.get('BRAM','-')} | {r.get('URAM','-')} |")
