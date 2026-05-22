#!/usr/bin/env python3
"""Run the current agentic workflow on knn and StreamCluster with Haiku/Sonnet.

This is a focused rerun after the Phase 12 hardening work:
  - multistep Phase B defaults to functional translation
  - dynamic bottleneck routing is enabled
  - static report harvest and GT prepopulation are enabled
  - coalescing gets a bounded second candidate by default
  - final accepted kernel is staged through Nova hw_emu

Outputs are timestamped under results_phase12/ and artifacts/ so previous
Phase 9/10 artifacts are left untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

STAMP = os.getenv("C2HLS_PHASE12_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
BENCHES = ["knn", "StreamCluster"]
MODELS = [
    ("haiku", "claude-haiku-4-5-20251001"),
    ("sonnet", "claude-sonnet-4-6"),
]

OUT_ROOT = REPO / "results_phase12" / f"knn_sc_models_{STAMP}"
OUT_JSONL = REPO / "artifacts" / f"phase12_knn_sc_models_{STAMP}.jsonl"
SUMMARY_JSON = REPO / "artifacts" / f"phase12_knn_sc_models_{STAMP}.summary.json"
SUMMARY_MD = REPO / "artifacts" / f"phase12_knn_sc_models_{STAMP}.md"

PREVIOUS_RESULTS = {
    ("knn", "haiku"): REPO / "results_phase2" / "knn_haiku_phase9_u280_v2023" / "knn_multistep_results.json",
    ("knn", "sonnet"): REPO / "results_phase2" / "knn_sonnet_phase9_u280_v2023" / "knn_multistep_results.json",
    ("StreamCluster", "haiku"): REPO / "results_phase2" / "streamcluster_haiku_phase9_u280_v2023" / "StreamCluster_multistep_results.json",
    ("StreamCluster", "sonnet"): REPO / "results_phase2" / "streamcluster_sonnet_phase9_u280_v2023" / "StreamCluster_multistep_results.json",
}


def _set_default_env() -> None:
    os.environ.setdefault("C2HLS_VITIS_SETTINGS", "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh")
    os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_FLOW_TARGET", "vitis")
    os.environ.setdefault("C2HLS_EMU_ENV_SCRIPT", str(REPO / "scripts" / "setup_emu_env.sh"))
    os.environ.setdefault("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
    os.environ.setdefault("C2HLS_CLAUDE_KEY_FILE", "/home/luo00466/claude-api-key.txt")
    os.environ.setdefault("C2HLS_STRATEGY", "dynamic")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "1")
    os.environ.setdefault("C2HLS_PHASE8_FMAX_FLOOR", "0.80")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "1")
    os.environ.setdefault("C2HLS_PHASE7A", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_CANDIDATES_PER_STEP", '{"coalescing":2,"default":1}')
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "1")
    os.environ.setdefault("C2HLS_HW_EMU_TIMEOUT", "86400")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")


def _selected_benches() -> list[str]:
    raw = os.getenv("C2HLS_PHASE12_BENCHES", "").strip()
    if not raw:
        return list(BENCHES)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    aliases = {bench.lower(): bench for bench in BENCHES}
    selected: list[str] = []
    unknown: list[str] = []
    for item in requested:
        bench = aliases.get(item.lower())
        if bench:
            selected.append(bench)
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(f"unknown C2HLS_PHASE12_BENCHES entries: {unknown}")
    return selected


def _selected_models() -> list[tuple[str, str]]:
    raw = os.getenv("C2HLS_PHASE12_MODELS", "").strip()
    if not raw:
        return list(MODELS)
    requested = {item.strip().lower() for item in raw.split(",") if item.strip()}
    selected = [
        (label, model_id)
        for label, model_id in MODELS
        if label.lower() in requested or model_id.lower() in requested
    ]
    known = {label.lower() for label, _ in MODELS} | {model_id.lower() for _, model_id in MODELS}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown C2HLS_PHASE12_MODELS entries: {unknown}")
    return selected


def _result_path(bench: str, label: str) -> Path:
    return OUT_ROOT / f"{bench}_{label}" / f"{bench}_multistep_results.json"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cycles(report: dict[str, Any] | None) -> int | None:
    if not report:
        return None
    value = report.get("latency_cycles")
    if value is None:
        value = report.get("latency_cycle")
    f = _safe_float(value)
    return int(round(f)) if f is not None else None


def _lat_ns(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    return _safe_float(report.get("latency_ns"))


def _best_step(data: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    baseline = data.get("baseline_report") or {}
    if baseline:
        candidates.append({"step": "baseline", "report": baseline, "csim": data.get("baseline_csim") or {}})
    for step in data.get("steps") or []:
        if step.get("success") and step.get("report"):
            candidates.append({"step": step.get("step_name") or "step", "report": step["report"], "csim": step.get("csim") or {}})
    if not candidates:
        return {"step": None, "cycles": None, "latency_ns": None, "csim": None}
    best = min(candidates, key=lambda item: (_lat_ns(item["report"]) is None, _lat_ns(item["report"]) or float("inf")))
    return {
        "step": best["step"],
        "cycles": _cycles(best["report"]),
        "latency_ns": _lat_ns(best["report"]),
        "csim": (best.get("csim") or {}).get("passed"),
    }


def _summarize(data: dict[str, Any]) -> dict[str, Any]:
    steps = data.get("steps") or []
    hw_emu = data.get("hw_emu") or {}
    return {
        "phase": data.get("phase"),
        "success": data.get("success"),
        "phase_b_mode": data.get("phase_b_mode"),
        "baseline_cycles": _cycles(data.get("baseline_report") or {}),
        "baseline_latency_ns": _lat_ns(data.get("baseline_report") or {}),
        "best": _best_step(data),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "step_cycles": [
            {
                "step": step.get("step_name"),
                "success": step.get("success"),
                "cycles": _cycles(step.get("report") or {}),
                "latency_ns": _lat_ns(step.get("report") or {}),
                "csim": (step.get("csim") or {}).get("passed"),
                "candidate_attempts": len(step.get("candidate_attempts") or []),
                "skill_id": step.get("skill_id") or (step.get("routing_decision") or {}).get("skill_id"),
            }
            for step in steps
        ],
        "phase_b_fast_candidate": data.get("phase_b_fast_candidate"),
        "hw_emu": {
            "ran": hw_emu.get("ran"),
            "success": hw_emu.get("success"),
            "passed": hw_emu.get("passed"),
            "variant_index": hw_emu.get("variant_index"),
            "variant_name": hw_emu.get("variant_name"),
            "variant_step": hw_emu.get("variant_step"),
            "kernel_runtime_cycles": hw_emu.get("kernel_runtime_cycles"),
            "kernel_runtime_us": hw_emu.get("kernel_runtime_us"),
            "kernel_clock_freq_mhz": hw_emu.get("kernel_clock_freq_mhz"),
            "clock_source": hw_emu.get("clock_source"),
            "clock_fallback": hw_emu.get("clock_fallback"),
            "skip_reason": hw_emu.get("skip_reason"),
            "error": (hw_emu.get("error") or "")[:300],
        },
        "json": "",
    }


def _load_previous(bench: str, label: str) -> dict[str, Any] | None:
    path = PREVIOUS_RESULTS[(bench, label)]
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    summary = _summarize(data)
    summary["json"] = str(path)
    return summary


def _export_jsonl(completed: list[tuple[str, str]]) -> int:
    import export_schema_jsonl as ex

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT_JSONL.open("w") as handle:
        for bench, label in completed:
            path = _result_path(bench, label)
            if not path.exists():
                continue
            records = ex._records_from_multistep(
                REPO / "benchmarks" / bench,
                path,
                default_part=os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
                default_clock_ns=float(os.getenv("C2HLS_CLOCK_NS", "3.33")),
            )
            for record in records:
                handle.write(json.dumps(record) + "\n")
                count += 1
    validation = ex.validate_jsonl(OUT_JSONL)
    if validation.get("invalid"):
        raise RuntimeError(f"invalid JSONL records={validation['invalid']} path={OUT_JSONL}")
    return count


def _ratio(new_value: int | float | None, old_value: int | float | None) -> str:
    if new_value is None or old_value in (None, 0):
        return "-"
    return f"{float(new_value) / float(old_value):.3f}x"


def _write_reports(rows: list[dict[str, Any]], jsonl_count: int) -> None:
    SUMMARY_JSON.write_text(json.dumps({
        "stamp": STAMP,
        "out_root": str(OUT_ROOT),
        "jsonl": str(OUT_JSONL),
        "jsonl_records": jsonl_count,
        "env": {
            key: os.getenv(key)
            for key in [
                "C2HLS_VITIS_SETTINGS",
                "C2HLS_VITIS_VERSION",
                "C2HLS_PART",
                "C2HLS_CLOCK_NS",
                "C2HLS_EMU_ENV_SCRIPT",
                "C2HLS_DEVICE_PLATFORM",
                "C2HLS_STRATEGY",
                "C2HLS_PHASEB_MODE",
                "C2HLS_CANDIDATES_PER_STEP",
                "C2HLS_PHASE8_BASELINE_ALIGN",
                "C2HLS_PHASE5_GT_PREPOP",
                "C2HLS_PHASE7A",
                "C2HLS_HW_EMU_FINAL",
                "C2HLS_HW_EMU_TIMEOUT",
                "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS",
                "C2HLS_LLM_TIMEOUT",
                "C2HLS_PHASE12_BENCHES",
                "C2HLS_PHASE12_MODELS",
            ]
        },
        "rows": rows,
    }, indent=2) + "\n")

    lines = [
        "# Phase 12 knn / StreamCluster Agentic Rerun",
        "",
        f"stamp: `{STAMP}`",
        f"results root: `{OUT_ROOT}`",
        f"jsonl: `{OUT_JSONL}`",
        f"jsonl records: `{jsonl_count}`",
        "",
        "| bench | model | status | steps | Phase B | best step | best cycles | prev best cycles | new/prev | hw_emu | hw cycles | hw variant | result |",
        "|---|---|---|---:|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        cur = row.get("current") or {}
        prev = row.get("previous") or {}
        best = cur.get("best") or {}
        prev_best = prev.get("best") or {}
        hw = cur.get("hw_emu") or {}
        if hw.get("success"):
            hw_status = "pass"
        elif hw.get("ran"):
            hw_status = "fail"
        else:
            hw_status = "skip"
        lines.append(
            f"| {row['bench']} | {row['model']} | {'pass' if cur.get('success') else 'fail'} | "
            f"{cur.get('steps_success', 0)}/{cur.get('steps_attempted', 0)} | "
            f"{cur.get('phase_b_mode') or '-'} | {best.get('step') or '-'} | "
            f"{best.get('cycles') if best.get('cycles') is not None else '-'} | "
            f"{prev_best.get('cycles') if prev_best.get('cycles') is not None else '-'} | "
            f"{_ratio(best.get('cycles'), prev_best.get('cycles'))} | "
            f"{hw_status} | "
            f"{hw.get('kernel_runtime_cycles') if hw.get('kernel_runtime_cycles') is not None else '-'} | "
            f"{hw.get('variant_name') or hw.get('skip_reason') or '-'} | "
            f"`{cur.get('json') or ''}` |"
        )
    lines.extend([
        "",
        "## Step Details",
        "",
    ])
    for row in rows:
        cur = row.get("current") or {}
        lines.append(f"### {row['bench']} / {row['model']}")
        lines.append("")
        lines.append("| step | success | cycles | csim | candidates | skill |")
        lines.append("|---|:---:|---:|:---:|---:|---|")
        for step in cur.get("step_cycles") or []:
            lines.append(
                f"| {step.get('step') or '-'} | {'yes' if step.get('success') else 'no'} | "
                f"{step.get('cycles') if step.get('cycles') is not None else '-'} | "
                f"{'yes' if step.get('csim') else ('no' if step.get('csim') is False else '-')} | "
                f"{step.get('candidate_attempts', 0)} | {step.get('skill_id') or '-'} |"
            )
        fast = cur.get("phase_b_fast_candidate")
        if fast:
            lines.append("")
            lines.append(f"Phase B fast-candidate metadata: `{json.dumps(fast, sort_keys=True)}`")
        hw = cur.get("hw_emu") or {}
        lines.append("")
        lines.append(
            "hw_emu: "
            f"ran={hw.get('ran')} success={hw.get('success')} passed={hw.get('passed')} "
            f"variant={hw.get('variant_name') or '-'} cycles={hw.get('kernel_runtime_cycles') or '-'} "
            f"clock={hw.get('kernel_clock_freq_mhz') or '-'} "
            f"note={hw.get('skip_reason') or hw.get('error') or '-'}"
        )
        lines.append("")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    _set_default_env()
    from c2hls import run_benchmark_multistep
    import hls_eval

    print(f"VITIS_SETTINGS={hls_eval.VITIS_SETTINGS}", flush=True)
    if "2023.2" not in hls_eval.VITIS_SETTINGS:
        raise RuntimeError(f"expected Vitis 2023.2, got {hls_eval.VITIS_SETTINGS}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    completed: list[tuple[str, str]] = []
    models = _selected_models()
    benches = _selected_benches()
    print(f"SELECTED benches={','.join(benches)} models={','.join(label for label, _ in models)}", flush=True)

    for label, model_id in models:
        for bench in benches:
            out_dir = OUT_ROOT / f"{bench}_{label}"
            result_json = _result_path(bench, label)
            print(f"START bench={bench} model={label} id={model_id} out={out_dir}", flush=True)
            t0 = time.time()
            try:
                result = run_benchmark_multistep(
                    str(REPO / "benchmarks" / bench),
                    output_dir=str(out_dir),
                    gpt_model=model_id,
                    turns_limitation=int(os.getenv("C2HLS_TURNS", "4")),
                    steps=None,
                )
            except Exception as exc:
                out_dir.mkdir(parents=True, exist_ok=True)
                result = {
                    "benchmark": bench,
                    "success": False,
                    "phase": "exception",
                    "error": str(exc),
                    "steps": [],
                    "hw_emu": {
                        "ran": False,
                        "skip_reason": f"agentic exception: {exc}",
                        "profile_required": True,
                    },
                    "run": {
                        "model": model_id,
                        "vitis_version": os.getenv("C2HLS_VITIS_VERSION"),
                        "part": os.getenv("C2HLS_PART"),
                        "clock_ns": os.getenv("C2HLS_CLOCK_NS"),
                        "flow_target": os.getenv("C2HLS_FLOW_TARGET"),
                    },
                }
                result_json.write_text(json.dumps(result, indent=2) + "\n")
                print(f"ERROR bench={bench} model={label}: {exc}", flush=True)

            elapsed = round(time.time() - t0, 3)
            current = _summarize(result)
            current["elapsed_sec"] = elapsed
            current["json"] = str(result_json)
            previous = _load_previous(bench, label)
            rows.append({
                "bench": bench,
                "model": label,
                "model_id": model_id,
                "current": current,
                "previous": previous,
            })
            completed.append((bench, label))
            jsonl_count = _export_jsonl(completed)
            _write_reports(rows, jsonl_count)
            best = current.get("best") or {}
            print(
                f"DONE bench={bench} model={label} success={current.get('success')} "
                f"steps={current.get('steps_success')}/{current.get('steps_attempted')} "
                f"best={best.get('step')} cycles={best.get('cycles')} elapsed={elapsed}s",
                flush=True,
            )
    print(f"SUMMARY {SUMMARY_MD}", flush=True)
    print(f"JSONL {OUT_JSONL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
