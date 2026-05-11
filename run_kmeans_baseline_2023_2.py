#!/usr/bin/env python3
"""One-off: synthesize kmeans rodinia baseline on whatever Vitis is sourced.

Used to compare 2023.2 vs 2025.2 numbers for the kmeans_0_baseline variant
against the dev.llm4hls.com reference. Re-run after sourcing the desired
settings64.sh; output JSON includes the Vitis version.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c2hls import _load_benchmark_inputs, _ground_truth_candidates  # noqa: E402
import hls_eval  # noqa: E402


def main() -> int:
    bench_dir = Path(__file__).resolve().parent / "benchmarks" / "kmeans"
    inputs = _load_benchmark_inputs(str(bench_dir))
    candidates = _ground_truth_candidates(inputs)
    baseline = next((c for c in candidates if c.get("variant_name") == "kmeans_0_baseline"), None)
    if baseline is None:
        print("error: kmeans_0_baseline candidate not found", file=sys.stderr)
        return 2

    meta = inputs["meta"]
    top = meta.get("hls_top", "workload")
    header_name = baseline.get("header_name") or inputs.get("header_name") or "kernel.h"
    header_code = baseline.get("header_code", inputs.get("header_code", ""))

    print(f"Vitis from $XILINX_VITIS = {os.environ.get('XILINX_VITIS', 'not set')}")
    print(f"Part = {hls_eval.DEFAULT_PART}, clock = {hls_eval.DEFAULT_CLOCK_NS} ns")
    print(f"Synthesizing kmeans_0_baseline (top={top}) ...")

    t0 = time.time()
    result = hls_eval.run_hls_synthesis(
        baseline["code"], header_code,
        header_name=header_name,
        top_function=top,
        part=hls_eval.DEFAULT_PART,
        clock_ns=hls_eval.DEFAULT_CLOCK_NS,
        extra_files=inputs.get("extra_files", []),
    )
    elapsed = round(time.time() - t0, 1)

    report = result.get("report") or {}
    vitis_version = "unknown"
    for env in ("XILINX_VITIS", "XILINX_HLS"):
        for token in os.environ.get(env, "").split("/"):
            if token.count(".") == 1 and token and token[0].isdigit():
                vitis_version = token
                break
        if vitis_version != "unknown":
            break

    summary = {
        "benchmark": "kmeans",
        "variant": "kmeans_0_baseline",
        "vitis_version": vitis_version,
        "part": hls_eval.DEFAULT_PART,
        "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
        "success": result.get("success"),
        "elapsed_sec": elapsed,
        "error": (result.get("error") or "")[:300],
        "metrics": {
            "latency_ns": report.get("latency_ns"),
            "fmax_mhz": report.get("fmax_mhz"),
            "lut": report.get("lut"),
            "ff": report.get("ff"),
            "bram": report.get("bram"),
            "dsp": report.get("dsp"),
        },
    }
    out_path = Path("artifacts") / f"kmeans_baseline_{vitis_version}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
