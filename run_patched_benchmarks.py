#!/usr/bin/env python3
"""Re-run benchmarks affected by recently-applied pipeline patches:
  - support.h strip in _rewrite_source_includes_for_local_support: fft, sort_merge, viterbi
  - regex extern "C" linkage check: hotspot

Outputs to results_haiku/ (overwrites prior failed runs) and emits a delta
JSONL plus a small markdown summary.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

BENCHES = ["fft", "sort_merge", "viterbi", "hotspot"]
MODEL_LABEL = "haiku"
MODEL_ID = "claude-haiku-4-5-20251001"

OUT_JSONL = REPO / "artifacts" / "run_patched_benchmarks.jsonl"
SUMMARY_MD = REPO / "artifacts" / "run_patched_benchmarks_summary.md"


def main() -> int:
    from c2hls import run_benchmark
    import export_schema_jsonl as ex

    summaries = []
    for bench in BENCHES:
        bench_dir = REPO / "benchmarks" / bench
        out_dir = REPO / f"results_{MODEL_LABEL}" / bench
        t0 = time.time()
        print(f"\n=== {bench} ===", flush=True)
        try:
            rec = run_benchmark(
                str(bench_dir),
                output_dir=str(out_dir),
                gpt_model=MODEL_ID,
                turns_limitation=3,
            )
            phase = rec.get("phase", "?")
            gen_lat = (rec.get("synth_report") or {}).get("latency_ns")
            gt_lat = (rec.get("ground_truth_report") or {}).get("latency_ns")
            csim_passed = (rec.get("csim") or {}).get("passed")
            elapsed = round(time.time() - t0, 1)
            summaries.append({
                "bench": bench, "phase": phase,
                "gen_latency_ns": gen_lat, "gt_latency_ns": gt_lat,
                "ratio": (gen_lat / gt_lat) if (gen_lat and gt_lat) else None,
                "csim_passed": csim_passed,
                "elapsed_sec": elapsed,
            })
            print(f"   phase={phase} gen={gen_lat} gt={gt_lat} csim={csim_passed} ({elapsed}s)",
                  flush=True)
        except Exception as exc:
            summaries.append({"bench": bench, "phase": "ERROR", "error": str(exc)[:200],
                              "elapsed_sec": round(time.time() - t0, 1)})
            print(f"   ERROR: {exc}", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    total = 0
    for bench in BENCHES:
        bench_dir = REPO / "benchmarks" / bench
        results_json = REPO / f"results_{MODEL_LABEL}" / bench / f"{bench}_results.json"
        if not results_json.exists():
            continue
        recs = ex._records_from_results_json(
            bench_dir, results_json,
            default_part="xcu280-fsvh2892-2L-e",
            default_clock_ns=3.33,
        )
        with OUT_JSONL.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        total += len(recs)
    print(f"\nwrote {total} records to {OUT_JSONL}", flush=True)

    lines = [
        f"# Patched-bench rerun — {MODEL_ID}\n",
        "Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis\n",
        "Patches applied:",
        " - strip `#include \"support.h\"` from upstream-rewritten headers",
        " - regex-match `extern \"C\"` linkage detection (no-space variant)\n",
        "| bench | phase | gen_lat_ns | gt_lat_ns | ratio | csim | sec |",
        "|---|---|---:|---:|---:|:---:|---:|",
    ]
    for s in summaries:
        ratio = f"{s['ratio']:.2f}×" if s.get("ratio") else "—"
        cg = "✓" if s.get("csim_passed") else ("✗" if s.get("csim_passed") is False else "—")
        lines.append(
            f"| {s['bench']} | {s.get('phase','?')} | {s.get('gen_latency_ns') or '—'} | "
            f"{s.get('gt_latency_ns') or '—'} | {ratio} | {cg} | {s.get('elapsed_sec','?')} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
