#!/usr/bin/env python3
"""Run the requested six-benchmark Claude multistep + final hw_emu smoke.

Each benchmark goes through the c2hls multistep agent workflow and, with
C2HLS_HW_EMU_FINAL=1, stages the accepted final kernel against the exact
upstream variant for a final `vitis.hw_emu` `rtl_sim` record.
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
os.environ.setdefault("C2HLS_HW_EMU_FINAL", "1")
os.environ.setdefault("C2HLS_HW_EMU_TIMEOUT", "86400")

BENCHES = ["knn", "lud", "pathfinder", "cfd_step_factor", "lc_dilate", "nw"]
MODEL_ID = os.getenv("C2HLS_AGENT_MODEL", "claude-haiku-4-5-20251001")
STEPS = [s.strip() for s in os.getenv(
    "C2HLS_AGENT_STEPS",
    "tiling,pipeline,unroll,doublebuffer,coalescing",
).split(",") if s.strip()]
WORKFLOW = {
    "entrypoint": "c2hls.run_benchmark_multistep",
    "orchestrator": "C2HLSOrchestrator",
    "agents": [
        {"name": "TranslatorAgent", "role": "Phase A C/C++ validation and initial HLS translation"},
        {"name": "SynthesisAgent", "role": "Phase B synthesis, csim/cosim gating, and repair loop"},
        {"name": "QualityRepairAgent", "role": "score/comparison-guided candidate repair when invoked by orchestrator APIs"},
    ],
    "phase_order": [
        "run_phase_a",
        "run_phase_b",
        "run_phase_c",
        "run_optimization_step",
        "_maybe_run_hw_emu_final",
    ],
}
OUT_ROOT = Path(os.getenv(
    "C2HLS_AGENT_RESULTS_DIR",
    str(REPO / "results_requested_agentic_hwemu"),
))
OUT_JSONL = Path(os.getenv(
    "C2HLS_AGENT_JSONL",
    str(REPO / "artifacts" / "requested_agentic_hwemu.jsonl"),
))
SUMMARY_JSON = OUT_JSONL.with_suffix(".summary.json")
SUMMARY_MD = OUT_JSONL.with_suffix(".md")
FORCE_RERUN = os.getenv("C2HLS_FORCE_RERUN", "").lower() in {"1", "true", "yes"}


def _multistep_json_path(bench: str) -> Path:
    return OUT_ROOT / bench / f"{bench}_multistep_results.json"


def _export_records_for_bench(bench: str) -> list[dict]:
    import export_schema_jsonl as ex

    path = _multistep_json_path(bench)
    if not path.exists():
        return []
    return ex._records_from_multistep(
        REPO / "benchmarks" / bench,
        path,
        default_part=os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
        default_clock_ns=float(os.getenv("C2HLS_CLOCK_NS", "3.33")),
    )


def _rewrite_jsonl(all_benches: list[str]) -> int:
    from export_schema_jsonl import validate_jsonl

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT_JSONL.open("w") as handle:
        for bench in all_benches:
            for record in _export_records_for_bench(bench):
                handle.write(json.dumps(record) + "\n")
                count += 1
    validation = validate_jsonl(OUT_JSONL)
    if validation["invalid"]:
        raise RuntimeError(f"agentic JSONL invalid_records={validation['invalid']} path={OUT_JSONL}")
    return count


def _summarize_result(bench: str, result: dict, elapsed: float) -> dict:
    steps = result.get("steps") or []
    hw = result.get("hw_emu") or {}
    return {
        "bench": bench,
        "phase": result.get("phase"),
        "success": result.get("success"),
        "elapsed_sec": round(elapsed, 3),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "step_status": [
            {
                "step": step.get("step_name"),
                "success": step.get("success"),
                "gen_latency_ns": (step.get("report") or {}).get("latency_ns"),
                "gt_latency_ns": (step.get("gt_report") or {}).get("latency_ns"),
                "csim_status": (step.get("csim") or {}).get("status"),
            }
            for step in steps
        ],
        "hw_emu": {
            "ran": hw.get("ran"),
            "success": hw.get("success"),
            "passed": hw.get("passed"),
            "variant_index": hw.get("variant_index"),
            "variant_name": hw.get("variant_name"),
            "variant_step": hw.get("variant_step"),
            "kernel_runtime_us": hw.get("kernel_runtime_us"),
            "kernel_runtime_cycles": hw.get("kernel_runtime_cycles"),
            "kernel_clock_freq_mhz": hw.get("kernel_clock_freq_mhz"),
            "clock_source": hw.get("clock_source"),
            "clock_fallback": hw.get("clock_fallback"),
            "skip_reason": hw.get("skip_reason"),
            "error": (hw.get("error") or "")[:300],
        },
    }


def _write_summaries(summaries: list[dict], jsonl_count: int) -> None:
    SUMMARY_JSON.write_text(json.dumps({
        "model": MODEL_ID,
        "steps": STEPS,
        "workflow": WORKFLOW,
        "jsonl": str(OUT_JSONL),
        "jsonl_records": jsonl_count,
        "summaries": summaries,
    }, indent=2) + "\n")
    lines = [
        "# Requested Agentic Multistep + hw_emu Smoke",
        "",
        f"model: `{MODEL_ID}`",
        f"steps: `{','.join(STEPS)}`",
        f"workflow: `{WORKFLOW['entrypoint']}` via `{WORKFLOW['orchestrator']}`",
        f"jsonl records: `{jsonl_count}`",
        "",
        "| bench | steps | hw_emu | variant | cycles | clock | note |",
        "|---|---:|:---:|---|---:|---:|---|",
    ]
    for item in summaries:
        hw = item.get("hw_emu") or {}
        note = hw.get("skip_reason") or hw.get("error") or ""
        lines.append(
            f"| {item['bench']} | {item.get('steps_success', 0)}/{item.get('steps_attempted', 0)} | "
            f"{'pass' if hw.get('success') else ('skip' if not hw.get('ran') else 'fail')} | "
            f"{hw.get('variant_name') or '-'} | "
            f"{hw.get('kernel_runtime_cycles') if hw.get('kernel_runtime_cycles') is not None else '-'} | "
            f"{hw.get('kernel_clock_freq_mhz') if hw.get('kernel_clock_freq_mhz') is not None else '-'} | "
            f"{note[:120]} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    from c2hls import run_benchmark_multistep

    summaries: list[dict] = []
    completed = []
    for bench in BENCHES:
        result_path = _multistep_json_path(bench)
        if result_path.exists() and not FORCE_RERUN:
            print(f"SKIP existing agentic result {bench}: {result_path}", flush=True)
            try:
                result = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                result = {"phase": "invalid_existing_result", "success": False}
            summaries.append(_summarize_result(bench, result, 0.0))
            completed.append(bench)
            _write_summaries(summaries, _rewrite_jsonl(completed))
            continue

        print(f"START agentic {bench} model={MODEL_ID} steps={STEPS}", flush=True)
        t0 = time.time()
        try:
            result = run_benchmark_multistep(
                str(REPO / "benchmarks" / bench),
                output_dir=str(OUT_ROOT / bench),
                gpt_model=MODEL_ID,
                steps=STEPS,
            )
        except Exception as exc:
            result = {
                "phase": "exception",
                "success": False,
                "error": str(exc)[:500],
                "steps": [],
                "hw_emu": {
                    "ran": False,
                    "skip_reason": f"agentic exception: {exc}",
                    "profile_required": True,
                },
            }
            (OUT_ROOT / bench).mkdir(parents=True, exist_ok=True)
            _multistep_json_path(bench).write_text(json.dumps(result, indent=2) + "\n")
        elapsed = time.time() - t0
        summary = _summarize_result(bench, result, elapsed)
        summaries.append(summary)
        completed.append(bench)
        jsonl_count = _rewrite_jsonl(completed)
        _write_summaries(summaries, jsonl_count)
        hw = summary["hw_emu"]
        print(
            f"DONE agentic {bench} steps={summary['steps_success']}/{summary['steps_attempted']} "
            f"hw_ran={hw.get('ran')} hw_success={hw.get('success')} "
            f"cycles={hw.get('kernel_runtime_cycles')} elapsed={elapsed:.1f}s",
            flush=True,
        )

    jsonl_count = _rewrite_jsonl(BENCHES)
    _write_summaries(summaries, jsonl_count)
    print(f"validated agentic jsonl records={jsonl_count} path={OUT_JSONL}", flush=True)
    print(f"summary={SUMMARY_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
