#!/usr/bin/env python3
"""Direct vitis_hls csynth on rodinia-hls-nova benches (cfd_flux, cfd_step_factor,
lc_gicov, lc_mgvf, lc_dilate) — no LLM. Compare against the user's reference
JSONL to validate our local install reproduces upstream numbers.

Outputs:
  artifacts/nova_direct_csynth.jsonl   — paired records (one per variant)
  artifacts/nova_direct_csynth_vs_ref.md — delta table vs reference
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

os.environ.setdefault(
    "C2HLS_VITIS_SETTINGS",
    "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh",
)
os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")

from c2hls import _load_benchmark_inputs, _ground_truth_candidates  # noqa: E402
import hls_eval  # noqa: E402
from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_CSYNTH,
    _build_hls_synth_payload,
    _build_implementation,
    _build_problem,
    _build_run,
    validate_jsonl,
)

REFERENCE_PATH = REPO / "csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"
OUT_JSONL = REPO / "artifacts" / "nova_direct_csynth.jsonl"
DELTA_MD  = REPO / "artifacts" / "nova_direct_csynth_vs_ref.md"

# (corpus_bench_name, group_path_in_reference)
NOVA_BENCHES = [
    ("cfd_flux",        ["cfd", "cfd_flux"]),
    ("cfd_step_factor", ["cfd", "cfd_step_factor"]),
    ("lc_gicov",        ["leukocyte", "lc_gicov"]),
    ("lc_mgvf",         ["leukocyte", "lc_mgvf"]),
]


def _parse_ref_lat_ns(s):
    if not s or s == "undef":
        return None
    try:
        n, u = s.split()
        return int(float(n) * {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[u])
    except Exception:
        return None


def _short_variant(full: str) -> str:
    parts = full.split("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[2].replace("unrolling", "unroll").replace("double_buffer", "doublebuffer")
    return full


def load_reference():
    out = {}
    for line in REFERENCE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gp = " / ".join(r["problem"]["group_path"])
        v = r["implementation"]["variant"]["name"]
        h = r.get("hls_synth", {})
        timing = h.get("PerformanceEstimates", {}).get("SummaryOfTimingAnalysis", {})
        lat = h.get("PerformanceEstimates", {}).get("SummaryOfOverallLatency", {})
        used = h.get("AreaEstimates", {}).get("Resources", {})
        out[(gp, v)] = {
            "latency_ns": _parse_ref_lat_ns(
                lat.get("Average-caseRealTimeLatency") or lat.get("Worst-caseRealTimeLatency")
            ),
            "clk_est_ns": float(timing.get("EstimatedClockPeriod", "0") or 0),
            "lut": int(used.get("LUT", 0)) if used.get("LUT") else None,
            "ff": int(used.get("FF", 0)) if used.get("FF") else None,
            "bram": int(used.get("BRAM_18K", 0)) if used.get("BRAM_18K") else None,
            "dsp": int(used.get("DSP", 0)) if used.get("DSP") else None,
        }
    return out


def main() -> int:
    ref = load_reference()
    print(f"Reference: {len(ref)} (group, variant) keys loaded")
    print(f"Vitis: device={hls_eval.DEFAULT_PART} clock={hls_eval.DEFAULT_CLOCK_NS}ns "
          f"flow={hls_eval.DEFAULT_FLOW_TARGET}\n", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    rows = []
    for bench_name, ref_gp in NOVA_BENCHES:
        bench_dir = REPO / "benchmarks" / bench_name
        if not bench_dir.is_dir():
            print(f"  SKIP {bench_name}: not in corpus")
            continue
        inputs = _load_benchmark_inputs(str(bench_dir))
        candidates = _ground_truth_candidates(inputs)
        meta = inputs["meta"]
        top = meta.get("hls_top", "workload")
        print(f"=== {bench_name} ({len(candidates)} variants) ===", flush=True)
        for cand in candidates:
            short = _short_variant(cand.get("variant_name", ""))
            t0 = time.time()
            r = hls_eval.run_hls_synthesis(
                cand["code"], cand.get("header_code", inputs.get("header_code", "")),
                header_name=cand.get("header_name", "kernel.h"),
                top_function=top,
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
            )
            elapsed = round(time.time() - t0, 1)
            ok = r.get("success", False)
            report = (r.get("report") or {}) if ok else {}
            ref_key = (" / ".join(ref_gp), short)
            ref_row = ref.get(ref_key)
            row = {
                "bench": bench_name, "variant_short": short,
                "variant_full": cand.get("variant_name", ""),
                "ok": ok,
                "elapsed_sec": elapsed,
                "ours": {"latency_ns": report.get("latency_ns"),
                         "lut": report.get("lut"), "ff": report.get("ff"),
                         "bram": report.get("bram"), "dsp": report.get("dsp"),
                         "fmax_mhz": report.get("fmax_mhz")},
                "ref": ref_row,
                "error": (r.get("error") or "")[:300] if not ok else None,
            }
            rows.append(row)
            run_meta = {
                "vitis_version": "2023.2",
                "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
                "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
            }
            record = {
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, hls_eval.DEFAULT_PART, elapsed, run_meta),
                "problem": {"suite": "rodinia_hls", "group_path": ref_gp},
                "implementation": _build_implementation(
                    meta,
                    variant_name=cand.get("variant_name", ""),
                    origin_override="rodinia_hls_benchmark",
                    origin_version="2023_port",
                    origin_meta={
                        "source_variant": cand.get("variant_name", ""),
                        "direct_script": "run_nova_direct_csynth.py",
                    },
                ),
                "hls_synth": _build_hls_synth_payload(
                    report,
                    hls_eval.DEFAULT_PART,
                    hls_eval.DEFAULT_CLOCK_NS,
                    status="pass" if ok else "fail",
                ),
            }
            with OUT_JSONL.open("a") as f:
                f.write(json.dumps(record) + "\n")
            r_lat = ref_row["latency_ns"] if ref_row else None
            our_lat = report.get("latency_ns")
            ratio = (our_lat / r_lat) if (our_lat and r_lat) else None
            ratio_s = f"{ratio:.2f}x" if ratio else "—"
            print(f"  {short:<14} ok={str(ok):<5} ours_lat={our_lat} ref_lat={r_lat} ratio={ratio_s} ({elapsed}s)",
                  flush=True)

    validation = validate_jsonl(OUT_JSONL)
    if validation["invalid"]:
        print(f"ERROR: {validation['invalid']} malformed JSONL records in {OUT_JSONL}", file=sys.stderr)
        return 1

    # Delta markdown
    lines = [
        "# Nova benchmarks: direct csynth vs reference",
        "",
        f"Vitis 2023.2 / {hls_eval.DEFAULT_PART} / {hls_eval.DEFAULT_CLOCK_NS}ns / flow={hls_eval.DEFAULT_FLOW_TARGET}",
        "",
        "| bench | variant | metric | reference | ours | delta% |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        ref_row = r.get("ref")
        ours = r.get("ours", {})
        if not ref_row:
            lines.append(f"| {r['bench']} | {r['variant_short']} | — | (no ref) | — | — |")
            continue
        for metric in ("latency_ns", "lut", "ff", "bram", "dsp"):
            ref_v = ref_row.get(metric)
            our_v = ours.get(metric)
            try:
                our_int = int(our_v) if our_v is not None else None
            except (TypeError, ValueError):
                our_int = None
            if ref_v is None and our_int is None:
                continue
            if ref_v and our_int:
                delta_s = f"{(our_int - ref_v) / ref_v * 100:+.1f}%"
            else:
                delta_s = "—"
            lines.append(f"| {r['bench']} | {r['variant_short']} | {metric} | "
                         f"{ref_v if ref_v is not None else '—'} | "
                         f"{our_int if our_int is not None else '—'} | {delta_s} |")
    DELTA_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(rows)} rows to {OUT_JSONL}")
    print(f"wrote delta table to {DELTA_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
