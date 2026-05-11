#!/usr/bin/env python3
"""Run the 14 remaining benchmarks (those not in the prior pathfinder/knn/nw
sweep) end-to-end with claude-haiku-4-5, then emit a combined JSONL with the
agentic-vs-GT comparison rows."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

DONE = {"pathfinder", "knn", "nw"}
MODEL_LABEL = "haiku"
MODEL_ID = "claude-haiku-4-5-20251001"

OUT_JSONL = REPO / "artifacts" / "run_remaining_haiku.jsonl"
SUMMARY_MD = REPO / "artifacts" / "run_remaining_haiku_summary.md"


def benches() -> list[str]:
    return sorted(
        p.name for p in (REPO / "benchmarks").iterdir()
        if p.is_dir() and p.name not in DONE
    )


def main() -> int:
    from c2hls import run_benchmark
    import export_schema_jsonl as ex

    targets = benches()
    print(f"running {len(targets)} benchmarks with {MODEL_ID}", flush=True)
    print(f"targets: {targets}", flush=True)

    summaries = []
    for bench in targets:
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
            gt_csim = (rec.get("reference_validation") or {}).get("csim", {}).get("passed")
            elapsed = round(time.time() - t0, 1)
            summaries.append({
                "bench": bench,
                "phase": phase,
                "gen_latency_ns": gen_lat,
                "gt_latency_ns": gt_lat,
                "ratio_gen_over_gt": (gen_lat / gt_lat) if (gen_lat and gt_lat) else None,
                "csim_passed": csim_passed,
                "gt_csim_passed": gt_csim,
                "elapsed_sec": elapsed,
                "error": rec.get("error", "")[:200] if rec.get("error") else "",
            })
            print(f"   phase={phase} gen_lat={gen_lat} gt_lat={gt_lat} csim={csim_passed} ({elapsed}s)",
                  flush=True)
        except Exception as exc:
            summaries.append({
                "bench": bench, "phase": "ERROR", "error": str(exc)[:300],
                "elapsed_sec": round(time.time() - t0, 1),
            })
            print(f"   ERROR: {exc}", flush=True)

    # Combined JSONL
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    total_recs = 0
    for bench in targets:
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
        total_recs += len(recs)
    print(f"\nwrote {total_recs} records to {OUT_JSONL}", flush=True)

    # Markdown summary
    lines = [
        f"# Remaining-14 agentic-workflow run — {MODEL_ID}\n",
        f"Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis\n",
        "| bench | phase | gen_lat_ns | gt_lat_ns | gen/gt | csim gen | csim gt | sec |",
        "|---|---|---:|---:|---:|:---:|:---:|---:|",
    ]
    for s in summaries:
        ratio = f"{s['ratio_gen_over_gt']:.2f}×" if s.get("ratio_gen_over_gt") else "—"
        gen_lat = s.get("gen_latency_ns") or "—"
        gt_lat  = s.get("gt_latency_ns") or "—"
        cg = "✓" if s.get("csim_passed") else ("✗" if s.get("csim_passed") is False else "—")
        gt = "✓" if s.get("gt_csim_passed") else ("✗" if s.get("gt_csim_passed") is False else "—")
        lines.append(
            f"| {s['bench']} | {s.get('phase','?')} | {gen_lat} | {gt_lat} | {ratio} | "
            f"{cg} | {gt} | {s.get('elapsed_sec','?')} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
