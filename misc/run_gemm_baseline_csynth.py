#!/usr/bin/env python3
"""Csynth team vs ours gemm functional baselines (misc/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from c2hls_paths import apply_runtime_defaults, configure_site

configure_site("pc2")
apply_runtime_defaults()

import hls_eval  # noqa: E402

BENCH = REPO / "benchmarks" / "hlsfactory_gemm"
HEADER = (BENCH / "gemm.h").read_text()
PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33

CASES = [
    ("team_functional_m_axi", REPO / "misc" / "teams" / "team_gemm_functional_baseline" / "kernel.cpp"),
    ("ours_hls_baseline_top_only", REPO / "misc" / "ours_gemm_hls_baseline" / "kernel.cpp"),
    ("benchmarks_hls_baseline", BENCH / "hls_baseline.cpp"),
]


def main() -> int:
    results = []
    for label, cpp_path in CASES:
        code = cpp_path.read_text()
        print(f"\n=== csynth: {label} ({cpp_path.name}) ===")
        out = hls_eval.run_hls_synthesis(
            code,
            HEADER,
            header_name="gemm.h",
            top_function="kernel_gemm",
            part=PART,
            clock_ns=CLOCK_NS,
        )
        report = out.get("report") or {}
        row = {
            "label": label,
            "source": str(cpp_path.relative_to(REPO)),
            "success": bool(out.get("success")),
            "error": (out.get("error") or "")[:500],
            "latency_cycles": report.get("latency_cycles"),
            "bram": report.get("bram"),
            "dsp": report.get("dsp"),
            "lut": report.get("lut"),
            "ff": report.get("ff"),
            "fmax_mhz": report.get("fmax_mhz"),
            "work_dir": report.get("work_dir"),
        }
        results.append(row)
        if row["success"]:
            print(f"  latency_cycles: {row['latency_cycles']:,}")
            print(f"  bram={row['bram']} dsp={row['dsp']} lut={row['lut']} ff={row['ff']}")
        else:
            print(f"  FAILED: {row['error']}")

    out_path = REPO / "misc" / "gemm_baseline_csynth_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
