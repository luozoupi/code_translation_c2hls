"""
Direct Vitis HLS synthesis sweep over HLSFactory polybench__float__small
benchmarks, producing per-benchmark synth reports (resources + latency in
cycles + latency in ns + Fmax). No LLM — pure reference/ground-truth pass.

Usage:
    VITIS_SETTINGS=/tools/Xilinx/Vitis/2023.2/settings64.sh \
    python3 synth_hlsfactory.py [--root <polybench_dir>] [--out <json>] [--limit N]

Outputs:
    <out_json> with a list of {bench, top, success, error, report} entries
    Prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from hls_eval import run_hls_synthesis, run_csim


DEFAULT_ROOT = Path(
    "/mnt/e/courses/UMN/c2hls/HLSFactory/hlsfactory/hls_dataset_sources/"
    "polybench__reproducible/polybench__float__small"
)

# Per the HLSFactory dataset's hls_template.tcl
HLSFACTORY_PART = "xczu9eg-ffvb1156-2-i"
HLSFACTORY_CLOCK_NS = 10


def _load_bench(bench_dir: Path) -> dict | None:
    top_file = bench_dir / "top.txt"
    src_dir = bench_dir / "src"
    tb_dir = bench_dir / "tb"
    if not top_file.is_file() or not src_dir.is_dir():
        return None
    top = top_file.read_text().strip()
    cpp_files = sorted(src_dir.glob("*.cpp"))
    h_files = sorted(src_dir.glob("*.h"))
    if not cpp_files:
        return None
    tb_files = sorted(tb_dir.glob("*_tb.cpp")) if tb_dir.is_dir() else []
    return {
        "name": bench_dir.name,
        "top": top,
        "cpp_path": cpp_files[0],
        "h_path": h_files[0] if h_files else None,
        "tb_path": tb_files[0] if tb_files else None,
    }


def _synth_one(bench: dict, part: str, clock_ns: float, do_csim: bool) -> dict:
    hls_code = bench["cpp_path"].read_text()
    header_code = bench["h_path"].read_text() if bench["h_path"] else ""
    header_name = bench["h_path"].name if bench["h_path"] else "kernel.h"

    t0 = time.time()
    synth = run_hls_synthesis(
        hls_code, header_code,
        header_name=header_name,
        top_function=bench["top"],
        part=part, clock_ns=clock_ns,
    )
    synth_secs = round(time.time() - t0, 2)
    report = synth.get("report", {}) or {}

    csim_info = {"ran": False}
    if do_csim and bench.get("tb_path") and synth.get("success"):
        tb_code = bench["tb_path"].read_text()
        t1 = time.time()
        csim = run_csim(
            hls_code, tb_code, header_code,
            header_name=header_name,
            top_function=bench["top"],
            part=part, clock_ns=clock_ns,
        )
        csim_info = {
            "ran": True,
            "passed": bool(csim.get("passed", False)),
            "success": bool(csim.get("success", False)),
            "error": (csim.get("error") or "")[:300],
            "seconds": round(time.time() - t1, 2),
        }
    elif do_csim and not bench.get("tb_path"):
        csim_info = {"ran": False, "skipped_reason": "no testbench"}

    return {
        "bench": bench["name"],
        "top": bench["top"],
        "success": bool(synth.get("success", False)),
        "error": synth.get("error", "")[:400],
        "synth_seconds": synth_secs,
        "csim": csim_info,
        "report": {
            "latency_cycles": report.get("latency_cycles"),
            "latency_ns": report.get("latency_ns"),
            "interval": report.get("interval"),
            "fmax_mhz": report.get("fmax_mhz"),
            "estimated_clock_period_ns": report.get("estimated_clock_period_ns"),
            "slack_ns": report.get("slack_ns"),
            "bram": report.get("bram"),
            "dsp": report.get("dsp"),
            "ff": report.get("ff"),
            "lut": report.get("lut"),
            "uram": report.get("uram"),
        },
    }


def _print_table(rows: list[dict]):
    cols = ["bench", "ok", "csim", "latency_cyc", "latency_ns", "fmax_mhz", "bram", "dsp", "ff", "lut", "secs"]
    widths = {c: max(len(c), 8) for c in cols}
    formatted = []
    for r in rows:
        rep = r["report"]
        csim = r.get("csim", {}) or {}
        if not csim.get("ran"):
            csim_cell = "-"
        else:
            csim_cell = "PASS" if csim.get("passed") else ("OK" if csim.get("success") else "FAIL")
        row = {
            "bench": r["bench"],
            "ok": "Y" if r["success"] else "N",
            "csim": csim_cell,
            "latency_cyc": str(rep.get("latency_cycles") or "-"),
            "latency_ns": (f"{float(rep['latency_ns']):.0f}" if rep.get("latency_ns") is not None else "-"),
            "fmax_mhz": (f"{rep['fmax_mhz']:.1f}" if rep.get("fmax_mhz") else "-"),
            "bram": str(rep.get("bram") or "-"),
            "dsp": str(rep.get("dsp") or "-"),
            "ff": str(rep.get("ff") or "-"),
            "lut": str(rep.get("lut") or "-"),
            "secs": f"{r['synth_seconds']:.1f}",
        }
        formatted.append(row)
        for c in cols:
            widths[c] = max(widths[c], len(row[c]))
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in formatted:
        print(" | ".join(row[c].ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON path (default: hlsfactory_<tag>_synth.json)")
    parser.add_argument("--tag", type=str, default="zynq_10ns",
                        help="label embedded in default output filename and report header")
    parser.add_argument("--part", type=str, default=HLSFACTORY_PART)
    parser.add_argument("--clock-ns", type=float, default=HLSFACTORY_CLOCK_NS)
    parser.add_argument("--csim", action="store_true", help="also run csim on each benchmark")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--only", type=str, default="", help="comma-separated bench names")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path(f"hlsfactory_polybench_float_small_synth_{args.tag}.json")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.root.is_dir():
        print(f"ERROR: root dir not found: {args.root}", file=sys.stderr)
        sys.exit(2)

    only = {n.strip() for n in args.only.split(",") if n.strip()}
    benches = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        if only and d.name not in only:
            continue
        info = _load_bench(d)
        if info:
            benches.append(info)
    if args.limit > 0:
        benches = benches[: args.limit]

    print(f"Found {len(benches)} HLSFactory benchmarks under {args.root}")
    print(f"Tag={args.tag}  Part={args.part}  Clock={args.clock_ns}ns  csim={args.csim}")

    results = []
    sweep_meta = {
        "tag": args.tag,
        "part": args.part,
        "clock_ns": args.clock_ns,
        "csim_enabled": args.csim,
        "root": str(args.root),
        "vitis_version": "2023.2",
        "vitis_settings": "/tools/Xilinx/Vitis/2023.2/settings64.sh",
    }
    for i, b in enumerate(benches, 1):
        print(f"[{i}/{len(benches)}] {b['name']}: synthesizing top={b['top']} ...", flush=True)
        try:
            row = _synth_one(b, args.part, args.clock_ns, args.csim)
        except Exception as e:
            row = {"bench": b["name"], "top": b["top"], "success": False,
                   "error": f"exception: {e}", "synth_seconds": 0.0,
                   "csim": {"ran": False}, "report": {}}
        results.append(row)
        ok = "PASS" if row["success"] else "FAIL"
        cyc = row["report"].get("latency_cycles")
        ns = row["report"].get("latency_ns")
        csim_summary = ""
        if args.csim:
            c = row.get("csim", {}) or {}
            csim_summary = f"  csim={'PASS' if c.get('passed') else ('OK' if c.get('success') else 'FAIL')}"
        print(f"    {ok}  cycles={cyc}  ns={ns}{csim_summary}  ({row['synth_seconds']}s)")
        args.out.write_text(json.dumps({"meta": sweep_meta, "rows": results}, indent=2, default=str))

    print()
    print("=== SUMMARY ===")
    print(f"  tag={args.tag}  part={args.part}  clock={args.clock_ns}ns")
    _print_table(results)
    print()
    passed = sum(1 for r in results if r["success"])
    print(f"Result: {passed}/{len(results)} benchmarks synthesized")
    if args.csim:
        passed_csim = sum(1 for r in results if (r.get("csim") or {}).get("passed"))
        ran_csim = sum(1 for r in results if (r.get("csim") or {}).get("ran"))
        print(f"Csim:   {passed_csim}/{ran_csim} passed (of {len(results)} attempted)")
    print(f"Full report written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
