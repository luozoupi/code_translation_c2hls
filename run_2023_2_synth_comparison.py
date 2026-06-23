#!/usr/bin/env python3
"""Synthesize nw / pathfinder / knn variants on Vitis 2023.2 + xcu280, then
compare against the reference JSONL at csynth_vitis_2023.2__device_xilinx_u280_*.jsonl.

Pre-req: set C2HLS_VITIS_SETTINGS in local.env (see local.env.example).
Override env: C2HLS_PART=xcu280-fsvh2892-2L-e  C2HLS_CLOCK_NS=3.33

Outputs:
  artifacts/run_2023_2_xcu280.jsonl      — our synthesis records (sw_run-style)
  artifacts/run_2023_2_comparison.md     — side-by-side delta table
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from c2hls_paths import apply_runtime_defaults, configure_site

configure_site()
apply_runtime_defaults()

REFERENCE_PATH = REPO / "csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"
OUR_OUT_PATH = REPO / "artifacts" / "run_2023_2_xcu280.jsonl"
COMPARISON_PATH = REPO / "artifacts" / "run_2023_2_comparison.md"

BENCHES = ["nw", "pathfinder", "knn"]


def _vitis_version() -> str:
    if os.environ.get("C2HLS_VITIS_VERSION"):
        return os.environ["C2HLS_VITIS_VERSION"]
    for env in ("XILINX_VITIS", "XILINX_HLS"):
        for tok in os.environ.get(env, "").split("/"):
            if tok.count(".") == 1 and tok and tok[0].isdigit():
                return tok
    return "unknown"


def _short_variant_name(full: str) -> str:
    # "nw_1_tiling" -> "tiling"; "kmeans_0_baseline" -> "baseline"
    parts = full.split("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[2].replace("unrolling", "unroll").replace("double_buffer", "doublebuffer")
    return full


def _parse_ref_latency_ns(lat_str):
    if not lat_str or lat_str == "undef":
        return None
    try:
        num, unit = lat_str.split()
        return int(float(num) * {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[unit])
    except Exception:
        return None


def load_reference():
    """Return {(bench, short_variant): {latency_ns, clk_est_ns, lut, ff, bram, dsp}}."""
    out = {}
    for line in REFERENCE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        bench = r["problem"]["group_path"][0]
        if bench not in BENCHES:
            continue
        v = r["implementation"]["variant"]["name"]
        h = r.get("hls_synth", {})
        timing = h.get("PerformanceEstimates", {}).get("SummaryOfTimingAnalysis", {})
        lat = h.get("PerformanceEstimates", {}).get("SummaryOfOverallLatency", {})
        used = h.get("AreaEstimates", {}).get("Resources", {})
        out[(bench, v)] = {
            "latency_ns": _parse_ref_latency_ns(
                lat.get("Average-caseRealTimeLatency") or lat.get("Worst-caseRealTimeLatency")
            ),
            "clk_est_ns": float(timing.get("EstimatedClockPeriod", "0") or 0),
            "lut": int(used.get("LUT", 0)),
            "ff": int(used.get("FF", 0)),
            "bram": int(used.get("BRAM_18K", 0)),
            "dsp": int(used.get("DSP", 0)),
        }
    return out


def synth_one(bench_dir, candidate, inputs, hls_eval):
    meta = inputs["meta"]
    top = meta.get("hls_top", "workload")
    header_name = candidate.get("header_name") or inputs.get("header_name") or "kernel.h"
    header_code = candidate.get("header_code", inputs.get("header_code", ""))
    t0 = time.time()
    result = hls_eval.run_hls_synthesis(
        candidate["code"], header_code,
        header_name=header_name,
        top_function=top,
        part=hls_eval.DEFAULT_PART,
        clock_ns=hls_eval.DEFAULT_CLOCK_NS,
        extra_files=inputs.get("extra_files", []),
    )
    return result, round(time.time() - t0, 1)


def main() -> int:
    from c2hls import _load_benchmark_inputs, _ground_truth_candidates  # noqa: WPS433
    import hls_eval  # noqa: WPS433
    from export_schema_jsonl import (  # noqa: WPS433
        SCHEMA_VERSION,
        TARGET_CSYNTH,
        _build_hls_synth_payload,
        _build_implementation,
        _build_run,
        validate_jsonl,
    )

    print(f"Vitis: {_vitis_version()}")
    print(f"Part: {hls_eval.DEFAULT_PART}, clock: {hls_eval.DEFAULT_CLOCK_NS} ns")
    if "u280" not in hls_eval.DEFAULT_PART:
        print("WARNING: part is not xcu280 — set C2HLS_PART=xcu280-fsvh2892-2L-e for fair compare")

    OUR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUR_OUT_PATH.write_text("")
    out_records = []
    comparison_rows = []

    for bench in BENCHES:
        bench_dir = REPO / "benchmarks" / bench
        inputs = _load_benchmark_inputs(str(bench_dir))
        candidates = _ground_truth_candidates(inputs)
        print(f"\n=== {bench} ({len(candidates)} variants) ===")
        for cand in candidates:
            short = _short_variant_name(cand.get("variant_name", ""))
            print(f"  -> {cand.get('variant_name')} (short={short})", flush=True)
            result, elapsed = synth_one(bench_dir, cand, inputs, hls_eval)
            report = (result.get("report") or {}) if result else {}
            ok = bool(result.get("success")) if result else False
            run_meta = {
                "vitis_version": _vitis_version(),
                "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
                "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
            }
            rec = {
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, hls_eval.DEFAULT_PART, elapsed, run_meta),
                "problem": {"suite": "rodinia_hls", "group_path": [bench]},
                "implementation": _build_implementation(
                    inputs["meta"],
                    variant_name=cand.get("variant_name", ""),
                    origin_override="rodinia_hls_benchmark",
                    origin_version="2023_port",
                    origin_meta={
                        "source_variant": cand.get("variant_name", ""),
                        "direct_script": "run_2023_2_synth_comparison.py",
                    },
                ),
                "hls_synth": _build_hls_synth_payload(
                    report,
                    hls_eval.DEFAULT_PART,
                    hls_eval.DEFAULT_CLOCK_NS,
                    status="pass" if ok else "fail",
                ),
            }
            if not ok:
                rec["hls_synth"]["error"] = (result.get("error") or "")[:300] if result else "synthesis failed"
            out_records.append(rec)
            comparison_rows.append({
                "bench": bench,
                "variant": short,
                "ours": {
                    "latency_ns": report.get("latency_ns"),
                    "lut": report.get("lut"),
                    "ff": report.get("ff"),
                    "bram": report.get("bram"),
                    "dsp": report.get("dsp"),
                },
            })
            with OUR_OUT_PATH.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"\nwrote {len(out_records)} records to {OUR_OUT_PATH}")
    validation = validate_jsonl(OUR_OUT_PATH)
    if validation["invalid"]:
        print(f"ERROR: {validation['invalid']} malformed JSONL records in {OUR_OUT_PATH}", file=sys.stderr)
        return 1

    ref = load_reference()
    lines = [
        f"# 2023.2 + xcu280 vs reference JSONL\n",
        f"Vitis {_vitis_version()} / part {hls_eval.DEFAULT_PART} / clock {hls_eval.DEFAULT_CLOCK_NS} ns\n",
        "Reference: same Vitis + xcu280, dev.llm4hls.com 2023_port\n",
        "\n| bench | variant | metric | reference | ours | delta% |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in comparison_rows:
        bench = row["bench"]
        short = row["variant"]
        ours = row["ours"]
        rkey = (bench, short)
        ref_row = ref.get(rkey)
        if not ref_row:
            lines.append(f"| {bench} | {short} | — | (no ref) | — | — |")
            continue
        for metric, our_key in [("latency_ns", "latency_ns"),
                                ("lut", "lut"), ("ff", "ff"),
                                ("bram", "bram"), ("dsp", "dsp")]:
            r = ref_row.get(metric)
            o = ours.get(our_key)
            try:
                o_int = int(o) if o is not None else None
            except (TypeError, ValueError):
                o_int = None
            if r is None and o_int is None:
                continue
            if r and o_int:
                delta = round((o_int - r) / r * 100, 1)
                delta_s = f"{delta:+.1f}%"
            else:
                delta_s = "—"
            lines.append(f"| {bench} | {short} | {metric} | {r if r is not None else '—'} | "
                         f"{o_int if o_int is not None else '—'} | {delta_s} |")
    COMPARISON_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
