#!/usr/bin/env python3
"""Table of the 58-bench Refined + Res.-Req runs: one row per bench and model, from the newest finished run dir.

  collect.py [--model ds|cc] [--csv OUT]
gt = the bench's ground-truth HLS design (the flow's reference); final = the flow's best csim-passing design.
"""
import argparse, csv, glob, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BW = HERE.parent / "bench_wide"
U280 = {"lut": 1303680, "ff": 2607360, "dsp": 9024, "bram": 4032, "uram": 960}


def row(bench, model):
    runs = sorted(glob.glob(str(HERE / f"runs/skills_bw_{bench}_{model}_*/*/*_multistep_results.json")))
    if not runs:
        return {"bench": bench, "status": "no result"}
    d = json.load(open(runs[-1]))
    fr, gt = d.get("final_report") or {}, d.get("ground_truth_report") or d.get("baseline_report") or {}
    csim = (d.get("csim") or {}).get("passed")
    lat = lambda r: r.get("latency_cycles_worst") or r.get("latency_cycles")
    fits = all((fr.get(k) or 0) <= v for k, v in U280.items()) if fr else None
    r = {"bench": bench, "status": "ok" if d.get("success") and csim else "fail", "csim": csim,
         "gt_lat": lat(gt), "gt_dsp": gt.get("dsp"), "lat": lat(fr), "dsp": fr.get("dsp"), "lut": fr.get("lut"),
         "bram": fr.get("bram"), "fits": fits}
    if r["gt_lat"] and r["lat"]:
        r["speedup"] = round(r["gt_lat"] / r["lat"], 2)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds")
    ap.add_argument("--csv")
    a = ap.parse_args()
    benches = sorted(p.name for p in BW.iterdir() if (p / "metadata.json").exists())
    rows = [row(b, a.model) for b in benches]
    cols = ["bench", "status", "csim", "gt_lat", "gt_dsp", "lat", "dsp", "lut", "bram", "fits", "speedup"]
    w = csv.DictWriter(open(a.csv, "w") if a.csv else sys.stdout, cols)
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
