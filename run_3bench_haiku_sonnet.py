#!/usr/bin/env python3
"""Run pathfinder / knn / nw end-to-end through the c2hls orchestrator with
both claude-haiku-4-5 and claude-sonnet-4-6, then emit JSONL to compare.

Outputs:
  results_haiku/<bench>/   — orchestrator artifacts for haiku
  results_sonnet/<bench>/  — orchestrator artifacts for sonnet
  artifacts/run_3bench_haiku_sonnet.jsonl — combined records for both models
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

BENCHES = ["pathfinder", "knn", "nw"]
MODELS = [
    ("haiku",  "claude-haiku-4-5-20251001"),
    ("sonnet", "claude-sonnet-4-6"),
]

OUT_JSONL = REPO / "artifacts" / "run_3bench_haiku_sonnet.jsonl"


def main() -> int:
    from c2hls import run_benchmark
    import export_schema_jsonl as ex

    results: list[dict] = []  # high-level per-run summary

    for label, model_id in MODELS:
        print(f"\n{'='*70}\nMODEL: {label} ({model_id})\n{'='*70}", flush=True)
        for bench in BENCHES:
            bench_dir = REPO / "benchmarks" / bench
            out_dir = REPO / f"results_{label}" / bench
            t0 = time.time()
            print(f"  -> {bench} ...", flush=True)
            try:
                rec = run_benchmark(
                    str(bench_dir),
                    output_dir=str(out_dir),
                    gpt_model=model_id,
                    turns_limitation=3,
                )
                phase = rec.get("phase", "?")
                gen_lat = (rec.get("synth_report") or {}).get("latency_ns")
                gen_csim = (rec.get("csim") or {}).get("passed")
                results.append({
                    "model": label, "model_id": model_id, "bench": bench,
                    "phase": phase,
                    "gen_latency_ns": gen_lat,
                    "gen_csim_passed": gen_csim,
                    "elapsed_sec": round(time.time() - t0, 1),
                })
                print(f"     phase={phase} latency_ns={gen_lat} csim={gen_csim} ({results[-1]['elapsed_sec']}s)",
                      flush=True)
            except Exception as exc:
                results.append({
                    "model": label, "model_id": model_id, "bench": bench,
                    "phase": "ERROR", "error": str(exc)[:200],
                    "elapsed_sec": round(time.time() - t0, 1),
                })
                print(f"     ERROR: {exc}", flush=True)

    # Summary table
    print("\nSUMMARY")
    print(f"{'model':<8}{'bench':<14}{'phase':<14}{'lat_ns':>14}{'csim':>6}{'elapsed':>10}")
    for r in results:
        lat = r.get("gen_latency_ns") or "—"
        csim = "✓" if r.get("gen_csim_passed") else ("✗" if r.get("gen_csim_passed") is False else "—")
        print(f"{r['model']:<8}{r['bench']:<14}{r['phase']:<14}{str(lat):>14}{csim:>6}{r['elapsed_sec']:>10}")

    # Emit JSONL — walk the per-model results dirs and collect records.
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")  # truncate
    total = 0
    for label, _ in MODELS:
        results_root = REPO / f"results_{label}"
        for bench in BENCHES:
            bench_dir = REPO / "benchmarks" / bench
            results_json = results_root / bench / f"{bench}_results.json"
            if not results_json.exists():
                print(f"  no results.json for {label}/{bench}, skipping", flush=True)
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
            print(f"  {label}/{bench}: {len(recs)} records", flush=True)

    print(f"\nwrote {total} records to {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
