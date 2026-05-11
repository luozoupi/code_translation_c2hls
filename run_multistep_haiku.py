#!/usr/bin/env python3
"""Run pathfinder / knn / nw / lavaMD in multistep mode with claude-haiku-4-5.

Each step (tiling → pipeline → unroll → doublebuffer → coalescing) translates
the next optimization. The orchestrator synthesises both the LLM output AND
the rodinia GT variant of the same step name, so per-step gen-vs-gt
comparison is meaningful (the JSONL exporter now emits both).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

BENCHES = ["pathfinder", "knn", "nw", "lavaMD"]
MODEL_ID = "claude-haiku-4-5-20251001"

OUT_JSONL  = REPO / "artifacts" / "run_multistep_haiku.jsonl"
SUMMARY_MD = REPO / "artifacts" / "run_multistep_haiku_summary.md"


def main() -> int:
    from c2hls import run_benchmark_multistep
    import export_schema_jsonl as ex

    summaries = []
    for bench in BENCHES:
        bench_dir = REPO / "benchmarks" / bench
        out_dir   = REPO / "results_multistep_haiku" / bench
        t0 = time.time()
        print(f"\n=== {bench} (multistep) ===", flush=True)
        try:
            rec = run_benchmark_multistep(
                str(bench_dir),
                output_dir=str(out_dir),
                gpt_model=MODEL_ID,
            )
            steps = rec.get("steps", []) or []
            success_steps = [s for s in steps if s.get("success")]
            elapsed = round(time.time() - t0, 1)
            summaries.append({
                "bench": bench, "phase": rec.get("phase", "?"),
                "n_steps_attempted": len(steps),
                "n_steps_success": len(success_steps),
                "step_results": [
                    {"step": s.get("step_name"),
                     "success": s.get("success"),
                     "gen_lat":  (s.get("report") or {}).get("latency_ns"),
                     "gt_lat":   (s.get("gt_report") or {}).get("latency_ns"),
                     "csim":     (s.get("csim") or {}).get("passed"),
                     } for s in steps
                ],
                "elapsed_sec": elapsed,
            })
            for s in summaries[-1]["step_results"]:
                ratio = (s["gen_lat"] / s["gt_lat"]) if (s["gen_lat"] and s["gt_lat"]) else None
                ratio_s = f"{ratio:.2f}x" if ratio else "—"
                print(f"   step={s['step']:<14} ok={str(s['success']):<5} "
                      f"gen_lat={s['gen_lat']} gt_lat={s['gt_lat']} ratio={ratio_s} "
                      f"csim={s['csim']}", flush=True)
            print(f"   [{bench}] elapsed={elapsed}s, {len(success_steps)}/{len(steps)} steps ok",
                  flush=True)
        except Exception as exc:
            summaries.append({"bench": bench, "phase": "ERROR", "error": str(exc)[:200],
                              "elapsed_sec": round(time.time() - t0, 1)})
            print(f"   ERROR: {exc}", flush=True)

    # Combined JSONL via the new multistep exporter (which emits per-step
    # AI + GT records).
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    total = 0
    for bench in BENCHES:
        bench_meta_dir = REPO / "benchmarks" / bench
        ms_json = REPO / "results_multistep_haiku" / bench / f"{bench}_multistep_results.json"
        if not ms_json.exists():
            print(f"  no multistep_results.json for {bench}", flush=True)
            continue
        recs = ex._records_from_multistep(
            bench_meta_dir, ms_json,
            default_part="xcu280-fsvh2892-2L-e",
            default_clock_ns=3.33,
        )
        with OUT_JSONL.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        total += len(recs)
        print(f"  {bench}: {len(recs)} records (paired AI+GT per step)", flush=True)
    print(f"\nwrote {total} records to {OUT_JSONL}", flush=True)

    # Markdown summary
    lines = [
        f"# Multistep agentic-workflow run — {MODEL_ID}\n",
        "Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis\n",
        "Per-step AI vs GT comparison at the same optimisation step.\n",
    ]
    for s in summaries:
        bench = s["bench"]
        lines.append(f"## {bench}")
        lines.append("")
        if s.get("phase") == "ERROR":
            lines.append(f"  ERROR: {s.get('error')}")
            lines.append("")
            continue
        lines.append(f"  Steps: {s.get('n_steps_success')}/{s.get('n_steps_attempted')} succeeded · "
                     f"elapsed: {s.get('elapsed_sec')}s")
        lines.append("")
        lines.append("| step | success | gen_lat_ns | gt_lat_ns | ratio | csim |")
        lines.append("|---|:---:|---:|---:|---:|:---:|")
        for ss in s.get("step_results", []):
            gen = ss.get("gen_lat") or "—"
            gt  = ss.get("gt_lat")  or "—"
            r = (ss["gen_lat"] / ss["gt_lat"]) if (ss.get("gen_lat") and ss.get("gt_lat")) else None
            r_s = f"{r:.2f}×" if r else "—"
            ok = "✓" if ss.get("success") else "✗"
            cm = "✓" if ss.get("csim") else ("✗" if ss.get("csim") is False else "—")
            lines.append(f"| {ss.get('step','?')} | {ok} | {gen} | {gt} | {r_s} | {cm} |")
        lines.append("")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
