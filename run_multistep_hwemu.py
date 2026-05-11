#!/usr/bin/env python3
"""4-bench multistep + hw_emu post-step run with claude-haiku-4-5.

Each step runs the LLM-translation + csim + csynth chain; after the final
step succeeds, hw_emu fires once on the accepted final cpp via nova
`make check TARGET=hw_emu`. The JSONL exporter emits paired AI+GT csynth
records per step, plus a final hw_emu rtl_sim record per bench.
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

OUT_JSONL  = REPO / "artifacts" / "run_multistep_hwemu_haiku.jsonl"
SUMMARY_MD = REPO / "artifacts" / "run_multistep_hwemu_haiku_summary.md"


def main() -> int:
    from c2hls import run_benchmark_multistep
    import export_schema_jsonl as ex

    summaries = []
    for bench in BENCHES:
        bench_dir = REPO / "benchmarks" / bench
        out_dir   = REPO / "results_multistep_hwemu_haiku" / bench
        t0 = time.time()
        print(f"\n=== {bench} (multistep + hw_emu) ===", flush=True)
        try:
            rec = run_benchmark_multistep(
                str(bench_dir),
                output_dir=str(out_dir),
                gpt_model=MODEL_ID,
            )
            steps = rec.get("steps", []) or []
            ok_steps = [s for s in steps if s.get("success")]
            hw = rec.get("hw_emu") or {}
            elapsed = round(time.time() - t0, 1)
            summaries.append({
                "bench": bench, "phase": rec.get("phase", "?"),
                "n_steps_attempted": len(steps),
                "n_steps_success": len(ok_steps),
                "step_results": [
                    {"step": s.get("step_name"),
                     "success": s.get("success"),
                     "gen_lat":  (s.get("report") or {}).get("latency_ns"),
                     "gt_lat":   (s.get("gt_report") or {}).get("latency_ns"),
                     "csim":     (s.get("csim") or {}).get("passed")}
                    for s in steps
                ],
                "hw_emu": {
                    "ran":      hw.get("ran"),
                    "passed":   hw.get("passed"),
                    "success":  hw.get("success"),
                    "kernel_runtime_us":     hw.get("kernel_runtime_us"),
                    "kernel_runtime_cycles": hw.get("kernel_runtime_cycles"),
                    "skip_reason": hw.get("skip_reason"),
                    "error":      (hw.get("error") or "")[:200],
                },
                "elapsed_sec": elapsed,
            })
            for s in summaries[-1]["step_results"]:
                ratio = (s["gen_lat"] / s["gt_lat"]) if (s["gen_lat"] and s["gt_lat"]) else None
                ratio_s = f"{ratio:.2f}x" if ratio else "—"
                print(f"   step={s['step']:<14} ok={str(s['success']):<5} "
                      f"gen_lat={s['gen_lat']} gt_lat={s['gt_lat']} ratio={ratio_s} "
                      f"csim={s['csim']}", flush=True)
            if hw.get("ran"):
                print(f"   hw_emu: kernel_runtime_us={hw.get('kernel_runtime_us')} "
                      f"cycles={hw.get('kernel_runtime_cycles')} passed={hw.get('passed')}",
                      flush=True)
            print(f"   [{bench}] elapsed={elapsed}s, {len(ok_steps)}/{len(steps)} steps ok",
                  flush=True)
        except Exception as exc:
            summaries.append({"bench": bench, "phase": "ERROR", "error": str(exc)[:200],
                              "elapsed_sec": round(time.time() - t0, 1)})
            print(f"   ERROR: {exc}", flush=True)

    # JSONL export
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    total = 0
    for bench in BENCHES:
        bench_meta_dir = REPO / "benchmarks" / bench
        ms_json = REPO / "results_multistep_hwemu_haiku" / bench / f"{bench}_multistep_results.json"
        if not ms_json.exists():
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
        print(f"  {bench}: {len(recs)} records", flush=True)
    print(f"\nwrote {total} records to {OUT_JSONL}", flush=True)

    # Markdown summary
    lines = [
        f"# 4-bench multistep + hw_emu — {MODEL_ID}\n",
        "Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis\n",
        "Per-step AI vs GT (csynth) + final-stage hw_emu kernel runtime.\n",
    ]
    for s in summaries:
        lines.append(f"## {s['bench']}\n")
        if s.get("phase") == "ERROR":
            lines.append(f"  ERROR: {s.get('error')}\n")
            continue
        lines.append(f"  steps: {s['n_steps_success']}/{s['n_steps_attempted']} · "
                     f"elapsed: {s['elapsed_sec']}s")
        hw = s.get("hw_emu") or {}
        if hw.get("ran"):
            lines.append(f"  hw_emu: kernel_runtime_us={hw.get('kernel_runtime_us')} "
                         f"cycles={hw.get('kernel_runtime_cycles')} passed={hw.get('passed')}\n")
        elif hw.get("skip_reason"):
            lines.append(f"  hw_emu skipped: {hw['skip_reason']}\n")
        else:
            lines.append("")
        lines.append("| step | ok | gen_lat | gt_lat | ratio | csim |")
        lines.append("|---|:---:|---:|---:|---:|:---:|")
        for ss in s.get("step_results", []):
            r = (ss["gen_lat"] / ss["gt_lat"]) if (ss.get("gen_lat") and ss.get("gt_lat")) else None
            r_s = f"{r:.2f}×" if r else "—"
            ok = "✓" if ss.get("success") else "✗"
            cm = "✓" if ss.get("csim") else ("✗" if ss.get("csim") is False else "—")
            lines.append(f"| {ss.get('step','?')} | {ok} | "
                         f"{ss.get('gen_lat') or '—'} | {ss.get('gt_lat') or '—'} | {r_s} | {cm} |")
        lines.append("")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
