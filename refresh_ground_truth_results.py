#!/usr/bin/env python3
"""
Refresh saved result JSONs against the current ground-truth selection rules.

By default this validates only the selected preferred GT variant for speed.
Use --full-workflow to validate every GT variant stage and persist the full
ground-truth workflow history into the saved result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hls_eval
from c2hls import (
    REPO_ROOT,
    _finalize_singleshot_results,
    _ground_truth_candidates,
    _load_benchmark_inputs,
    _preferred_reference_file,
    _summarize_test_result,
    _validate_ground_truth_candidate,
    validate_gold_reference,
)
from hls_eval import compare_reports


def _preserved_test_summary(previous_reference_validation: dict | None, selected_file: str,
                            key: str, supported: bool) -> dict:
    empty = _summarize_test_result(None, supported)
    if not isinstance(previous_reference_validation, dict):
        return empty

    workflow = previous_reference_validation.get("workflow") or []
    for stage in workflow:
        if not isinstance(stage, dict):
            continue
        if stage.get("file") != selected_file:
            continue
        summary = stage.get(key)
        if isinstance(summary, dict) and summary.get("ran"):
            return summary

    if previous_reference_validation.get("selected_variant_file") == selected_file:
        summary = previous_reference_validation.get(key)
        if isinstance(summary, dict) and summary.get("ran"):
            return summary

    return empty


def _selected_reference_validation(inputs: dict, previous_reference_validation: dict | None = None) -> dict:
    meta = inputs["meta"]
    supports_csim = bool(meta.get("supports_csim") and inputs.get("testbench_code"))
    supports_cosim = bool(meta.get("supports_cosim") and inputs.get("testbench_code"))
    candidates = _ground_truth_candidates(inputs)
    if not candidates:
        return {
            "benchmark_ready": False,
            "invalid_reason": "Missing gold HLS workflow code",
            "top_function": meta.get("hls_top", "workload"),
            "synthesis": {"status": "failed", "passed": False, "ran": False, "error": "Missing gold HLS workflow code", "report": {}},
            "csim": _summarize_test_result(None, supports_csim),
            "cosim": _summarize_test_result(None, supports_cosim),
            "report": {},
            "workflow": [],
            "selected_variant_name": "",
            "selected_variant_file": "",
            "selected_variant_step": "",
            "selection_reason": "",
        }

    preferred_file = _preferred_reference_file(
        meta,
        [{"file": c.get("file", ""), "step_name": c.get("step_name", "")} for c in candidates],
    )
    selected = None
    if preferred_file:
        for candidate in candidates:
            if candidate.get("file") == preferred_file:
                selected = candidate
                break
    if selected is None:
        selected = candidates[-1]

    validated = _validate_ground_truth_candidate(
        selected,
        inputs,
        supports_csim,
        supports_cosim,
        run_csim_check=False,
        run_cosim_check=False,
    )
    validated["csim"] = _preserved_test_summary(
        previous_reference_validation,
        validated.get("file", ""),
        "csim",
        supports_csim,
    )
    validated["cosim"] = _preserved_test_summary(
        previous_reference_validation,
        validated.get("file", ""),
        "cosim",
        supports_cosim,
    )
    validated["selected"] = True

    if not validated.get("benchmark_ready"):
        return {
            "benchmark_ready": False,
            "invalid_reason": validated.get("invalid_reason", "Invalid gold reference"),
            "top_function": meta.get("hls_top", "workload"),
            "synthesis": validated.get("synthesis", {}),
            "csim": validated.get("csim", {}),
            "cosim": validated.get("cosim", {}),
            "report": validated.get("report", {}),
            "workflow": [validated],
            "selected_variant_name": validated.get("variant_name", ""),
            "selected_variant_file": validated.get("file", ""),
            "selected_variant_step": validated.get("step_name", ""),
            "selection_reason": f"selected preferred validated variant `{validated.get('file', '')}`",
        }

    return {
        "benchmark_ready": True,
        "invalid_reason": "",
        "top_function": meta.get("hls_top", "workload"),
        "synthesis": validated.get("synthesis", {}),
        "csim": validated.get("csim", {}),
        "cosim": validated.get("cosim", {}),
        "report": validated.get("report", {}),
        "workflow": [validated],
        "selected_variant_name": validated.get("variant_name", ""),
        "selected_variant_file": validated.get("file", ""),
        "selected_variant_step": validated.get("step_name", ""),
        "selection_reason": f"selected preferred validated variant `{validated.get('file', '')}`",
    }


def _result_path(results_dir: Path, bench_name: str) -> Path:
    return results_dir / bench_name / f"{bench_name}_results.json"


def _benchmarks_from_args(args: argparse.Namespace) -> list[str]:
    if args.bench:
        return sorted(set(args.bench))

    index = json.loads((REPO_ROOT / "benchmarks" / "index.json").read_text())
    benches = [entry["benchmark"] for entry in index]
    if args.rodinia_only:
        kept = []
        for bench in benches:
            meta = json.loads((REPO_ROOT / "benchmarks" / bench / "metadata.json").read_text())
            if meta.get("source_repo") == "rodinia-hls":
                kept.append(bench)
        benches = kept
    return benches


def refresh_result(bench_name: str, results_dir: Path, full_workflow: bool) -> dict:
    bench_dir = REPO_ROOT / "benchmarks" / bench_name
    inputs = _load_benchmark_inputs(str(bench_dir))
    result_path = _result_path(results_dir, bench_name)

    if result_path.exists():
        saved = json.loads(result_path.read_text())
    else:
        saved = {"phase": "reference"}

    reference_validation = (
        validate_gold_reference(inputs)
        if full_workflow
        else _selected_reference_validation(inputs, saved.get("reference_validation"))
    )

    synth_report = saved.get("synth_report") or ((saved.get("comparison") or {}).get("generated_report"))
    if synth_report and reference_validation.get("benchmark_ready"):
        saved["comparison"] = {
            "success": True,
            "valid_reference": True,
            "invalid_reference": False,
            "generated_report": synth_report,
            "ground_truth_report": reference_validation.get("report", {}),
            "comparison": compare_reports(synth_report, reference_validation.get("report", {})),
        }
    elif not reference_validation.get("benchmark_ready"):
        saved["comparison"] = {
            "success": False,
            "valid_reference": False,
            "invalid_reference": True,
            "error": reference_validation.get("invalid_reason", "Invalid gold reference"),
        }

    success = bool(saved.get("phase") == "complete" and synth_report and reference_validation.get("benchmark_ready"))
    updated = _finalize_singleshot_results(bench_name, inputs["meta"], success, saved, reference_validation)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(updated, indent=2))

    report = reference_validation.get("report", {})
    return {
        "benchmark": bench_name,
        "gt_file": reference_validation.get("selected_variant_file", ""),
        "gt_step": reference_validation.get("selected_variant_step", ""),
        "latency_cycles": report.get("latency_cycles"),
        "latency_ns": report.get("latency_ns"),
        "fmax_mhz": report.get("fmax_mhz"),
        "status": "passed" if reference_validation.get("benchmark_ready") else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh saved result JSONs against current GT selection")
    parser.add_argument("--bench", action="append", help="Benchmark name to refresh (repeatable)")
    parser.add_argument("--rodinia-only", action="store_true", help="Refresh only Rodinia-HLS benchmarks")
    parser.add_argument("--full-workflow", action="store_true", help="Validate every GT stage instead of only the selected GT variant")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"), help="Saved single-shot results directory")
    parser.add_argument("--synth-timeout", type=int, default=None, help="Override Vitis synthesis timeout in seconds for this refresh run")
    args = parser.parse_args()

    if args.synth_timeout is not None:
        hls_eval.SYNTH_TIMEOUT = int(args.synth_timeout)

    results_dir = Path(args.results_dir)
    summaries = []
    for bench in _benchmarks_from_args(args):
        summaries.append(refresh_result(bench, results_dir, args.full_workflow))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
