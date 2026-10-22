#!/usr/bin/env python3
"""LightningSim (v2023 AXI model) vs Vitis 2023.2 cosim on the SODA designs.

  compare_soda.py [--csv OUT]
Cosim cycles: the Verilog row of <design>/cosim/*cosim.rpt (designs/, diag/ for the blur variants).
LightningSim: /scratch/asa582/tmp/c2hls/lsim/soda/<name>/lsim_v2023.json "end".
"mapped" applies ../../../cosim/lsim/compare.py's report mapping, which also counts the testbench's AXI-Lite
programming and status poll: P*ceil((E - 11 - 3)/P) + K, P = 6 + n, K = 27 + 12n, with n = the kernel's s_axilite
arguments (SODA kernels also pass one scalar, coalesced_data_num, so n = pointers + 1 here; the mapping was
calibrated on pointer-only kernels, so it is approximate for SODA).
"""
import argparse, csv, json, math, re, sys
from pathlib import Path

DSE = Path(__file__).resolve().parent.parent / "dse"
LS = Path("/scratch/asa582/tmp/c2hls/lsim/soda")
RX = re.compile(r"\|\s*Verilog\|\s*(\w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|")


def cosim(d):
    for f in sorted(d.glob("cosim/*cosim.rpt")):
        m = RX.search(f.read_text(errors="replace"))
        if m and m.group(1) == "Pass":
            return int(m.group(4))
    return None


def nargs(d):
    ker = (d / "kernel.cpp").read_text()
    return len(re.findall(r"#pragma HLS interface s_axilite port = (?!return)\w+", ker))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for o in sorted(LS.iterdir()):
        js = o / "lsim_v2023.json"
        if not js.exists():
            continue
        name = o.name
        if name.startswith("diag_"):
            d = DSE / "diag" / name[len("diag_"):]
        elif name.startswith("pat_"):      # SODA hand-patched for Vitis 2023.2 (designs_patched/, patch_soda.py)
            d = DSE / "designs_patched" / name[len("pat_"):]
        else:
            d = DSE / "designs" / name
        cy = cosim(d) if d.exists() else None
        r = json.loads(js.read_text())
        e = r.get("end")
        row = {"design": name, "cosim": cy, "lsim_end": e, "lsim_rc": r.get("rc"), "lsim_wall_s": r.get("wall_s")}
        if e and d.exists():
            n = nargs(d)
            P, K = 6 + n, 27 + 12 * n
            row["n_axilite"] = n
            row["lsim_mapped"] = P * math.ceil((e - 11 - 3) / P) + K
        for k in ("lsim_end", "lsim_mapped"):
            if cy and row.get(k):
                row[f"err_{k}_pct"] = round(100 * (row[k] - cy) / cy, 2)
        rows.append(row)
    cols = ["design", "cosim", "lsim_end", "lsim_mapped", "err_lsim_end_pct", "err_lsim_mapped_pct", "n_axilite",
            "lsim_rc", "lsim_wall_s"]
    w = csv.DictWriter(open(a.csv, "w") if a.csv else sys.stdout, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
