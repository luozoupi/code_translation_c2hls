#!/usr/bin/env python3
"""Compare LightningSim cycle counts with Vitis 2023.2 cosim, per configuration.

  compare.py ROOT [--vcd VCD_ROOT] [--tags fork_v2021,v2023] [--match REGEX] [--md OUT.md] [--csv OUT.csv]

Reference: the Verilog latency in ROOT/NAME/*_cosim.rpt (what Vitis reports). When VCD_ROOT/NAME/axi_timing.json
exists, the DUT's own ap_done cycle (ap_start = 0) is shown too.

LightningSim reports the top module's end, i.e. the DUT ap_done cycle. The cosim report also counts the
testbench's AXI-Lite programming before ap_start and its s_axilite status poll after ap_done, so a LightningSim
end E is mapped to a predicted report with
  pred = P * ceil((E - 3) / P) + K,   P = 6 + n,   K = 27 + 12 n
where n is the number of s_axilite-mapped top arguments (pointers). Measured exactly on the DUT ap_done of all
2-pointer microbenchmarks (P = 8, K = 51) and both 3-pointer autosa_mm self-tests (P = 9, K = 63).

For the v2023 model E is first reduced by END_OFFSET: LightningSim's schedule keeps 7-14 cycles after the final
write response that the RTL does not have (B and ap_done land in the same cycle in cosim); 11 centres LightningSim's
end on the DUT ap_done of the 65 microbenchmarks (mean error < 1 cycle).
"""
import argparse, csv, glob, json, math, os, re
from pathlib import Path


def cosim_cycles(d):
    for f in glob.glob(f"{d}/*cosim.rpt"):
        m = re.search(r"Verilog\|\s*Pass\|\s*(\d+)", open(f).read())
        if m:
            return int(m.group(1))
    f = Path(d) / "summary.json"
    if f.exists():  # run_design.py (cosim -setup path: the .rpt can be stale; hls_eval parses *.result.lat.rb)
        c = json.load(open(f)).get("kernel_runtime_cycles")
        if c:
            return c
    m = re.match(r"(.+)_n1024$", Path(d).name)
    ref = Path(__file__).resolve().parent.parent / (m.group(1) if m else "") / "cosim_summary.json"
    if m and ref.exists():  # the full 1024^3 Vitis cosim jobs (../run_cosim_one.py), once finished
        j = json.load(open(ref))
        if j.get("correct") or j.get("vitis_success"):
            return j.get("kernel_runtime_cycles")
    return None


END_OFFSET = {"v2023": 11}


def end_offset(tag):
    return next((v for k, v in END_OFFSET.items() if tag.endswith(k)), 0)


def report_map(end, nargs=2):
    period = 6 + nargs
    return period * math.ceil((end - 3) / period) + 27 + 12 * nargs


def saxi_args(d):
    """Number of top arguments mapped to s_axi_control registers (from csynth.rpt), default 2."""
    f = Path(d) / "csynth.rpt"
    if not f.exists():
        return 2
    args = set(re.findall(r"^\|\s*(\w+)\s*\|\s*s_axi_control\s*\|\s*register\s*\|\s*offset", f.read_text(), re.M))
    return len(args) or 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--vcd")
    ap.add_argument("--tags", default="v2021,fork_v2021,v2023")
    ap.add_argument("--md")
    ap.add_argument("--csv")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--match", default="", help="only configurations whose name matches this regex")
    a = ap.parse_args()
    tags = a.tags.split(",")
    rows = []
    for d in sorted(Path(a.root).iterdir()):
        if not d.is_dir() or (a.match and not re.search(a.match, d.name)):
            continue
        r = {"name": d.name, "cosim": cosim_cycles(d), "nargs": saxi_args(d)}
        if a.vcd and (Path(a.vcd) / d.name / "axi_timing.json").exists():
            r["dut_done"] = json.load(open(Path(a.vcd) / d.name / "axi_timing.json"))["ap_done"]
        for t in tags:
            f = d / f"lsim_{t}.json"
            if f.exists():
                j = json.load(open(f))
                if j.get("end") is not None:
                    r[f"{t}_end"] = j["end"] - end_offset(t)
                    r[f"{t}_pred"] = report_map(r[f"{t}_end"], r["nargs"])
                    r[f"{t}_wall"] = j.get("wall_s")
        rows.append(r)
    lines = []
    hdr = ["name", "cosim", "dut_done"] + [f"{t}{s}" for t in tags for s in ("_end", "_pred", "_err%")]
    # *_end is LightningSim's end minus the model's END_OFFSET (an ap_done estimate); *_pred the predicted report.
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    stats = {t: [] for t in tags}
    for r in rows:
        cells = [r["name"], str(r.get("cosim", "")), str(r.get("dut_done", ""))]
        for t in tags:
            e, p = r.get(f"{t}_end"), r.get(f"{t}_pred")
            err = (100.0 * (p - r["cosim"]) / r["cosim"]) if (p is not None and r.get("cosim")) else None
            if err is not None:
                r[f"{t}_err%"] = round(err, 3)
                stats[t].append((abs(err), abs(p - r["cosim"]), r["name"]))
            cells += [str(e if e is not None else ""), str(p if p is not None else ""), f"{err:+.2f}" if err is not None else ""]
        lines.append("| " + " | ".join(cells) + " |")
    summ = []
    for t, s in stats.items():
        if s:
            s.sort()
            within = sum(1 for e, c, _ in s if e <= 1.0 or c <= 5)
            dd = [abs(r[f"{t}_end"] - r["dut_done"]) for r in rows if r.get("dut_done") is not None and r.get(f"{t}_end") is not None]
            ddtxt = (f"; vs DUT ap_done: mean |err| {sum(dd) / len(dd):.1f} cycles, max {max(dd)}, "
                     f"within ±5 cycles {sum(1 for x in dd if x <= 5)}/{len(dd)}") if dd else ""
            summ.append(f"- `{t}`: {len(s)} configs, mean |err| {sum(e for e, _, _ in s) / len(s):.2f}%, "
                        f"max |err| {s[-1][0]:.2f}% ({s[-1][2]}), within ±1% or ±5 cycles: {within}/{len(s)}{ddtxt}")
    out = "\n".join(lines + [""] + summ) + "\n"
    if not a.quiet:
        print(out)
    else:
        print("\n".join(summ))
    if a.md:
        Path(a.md).write_text(out)
    if a.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "name", k))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
