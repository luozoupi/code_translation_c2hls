#!/usr/bin/env python3
"""T4 table: 1024^3 designs, LightningSim predictions next to the Vitis 2023.2 cosim reference (when finished).

  t4_table.py [--designs DIR] [--md OUT.md] [--csv OUT.csv]

Columns: csynth latency (NOT a reference: an estimate assuming ideal memory); Verilator RTL cycles (../verilator/runs, NOT a reference: its memory answers at once);
LightningSim end -> predicted cosim report (compare.report_map, v2021 and v2023, with LightningSim wall time);
Vitis cosim cycles from ../NAME/cosim_summary.json (the reference) and the error of each prediction.
"""
import argparse, csv, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from compare import cosim_cycles, end_offset, report_map, saxi_args  # noqa: E402


def csynth_latency(d):
    f = d / "csynth.rpt"
    if not f.exists():
        return None
    m = re.search(r"^\s*\|\+ \w+\s*\|[^|]*\|[^|]*\|\s*(\d+)\|", f.read_text(), re.M)
    return int(m.group(1)) if m else None


def verilator_cycles(name):
    log = HERE.parent / "verilator" / "runs" / name / "simulate.log"
    summ = HERE.parent / "verilator" / "runs" / name / "summary.json"
    if not log.exists():
        return None, None
    m = re.findall(r"VCOSIM_END status=(\w+) cycles=(\d+)", log.read_text(errors="replace"))
    ok = json.load(open(summ)).get("correct") if summ.exists() else None
    return (int(m[-1][1]) if m else None), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="/scratch/asa582/tmp/c2hls/lsim/designs")
    ap.add_argument("--md", default=str(HERE / "T4_n1024.md"))
    ap.add_argument("--csv", default=str(HERE / "T4_n1024.csv"))
    a = ap.parse_args()
    rows = []
    for d in sorted(Path(a.designs).glob("*_n1024")):
        name = d.name[: -len("_n1024")]
        r = {"design": name, "csynth_not_ref": csynth_latency(d)}
        vs = HERE.parent / "verilator" / "runs" / name / "summary.json"
        if r["csynth_not_ref"] is None and vs.exists():  # dataflow tops: take the csynth latency hls_eval parsed
            r["csynth_not_ref"] = json.load(open(vs)).get("csynth", {}).get("latency_cycles")
        r["verilator_not_ref"], r["verilator_correct"] = verilator_cycles(name)
        r["cosim_ref"] = cosim_cycles(d)
        for t in ("v2021", "v2023"):
            f = d / f"lsim_{t}.json"
            if f.exists():
                j = json.load(open(f))
                if j.get("end") is not None:
                    r[f"{t}_pred"] = report_map(j["end"] - end_offset(t), saxi_args(d))
                    r[f"{t}_wall_s"] = j.get("wall_s")
                    if r["cosim_ref"]:
                        r[f"{t}_err%"] = round(100.0 * (r[f"{t}_pred"] - r["cosim_ref"]) / r["cosim_ref"], 2)
        rows.append(r)
    cols = ["design", "csynth_not_ref", "verilator_not_ref", "verilator_correct", "v2021_pred", "v2023_pred",
            "v2023_wall_s", "cosim_ref", "v2021_err%", "v2023_err%"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    out = "\n".join(lines) + "\n"
    Path(a.md).write_text(out)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
