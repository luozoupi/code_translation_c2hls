#!/usr/bin/env python3
"""Build traceable Qwen/Gemma one-shot, agentic, and SFT comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALTEST_SPLITS = {
    "hlsfactory_gramschmidt": "validation",
    "hlsfactory_durbin": "test",
    "hlsfactory_floyd_warshall": "test",
    "hlsfactory_gemm": "test",
    "hlsfactory_trmm": "test",
}
MODE_ORDER = {
    "none": 0,
    "skillless": 1,
    "matched": 2,
    "selective_positive": 3,
    "smart_best_fit": 4,
    "smart_exhaustive": 5,
    "all_positive": 6,
}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def normalize_benchmark(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    if not name:
        return ""
    return name if name.startswith("hlsfactory_") else f"hlsfactory_{name}"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def geomean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0 and math.isfinite(value)]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def pass_status(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in {"pass", "passed", "success", "succeeded"}
    record = as_dict(value)
    return (
        record.get("passed") is True
        or record.get("success") is True
        or str(record.get("status") or "").lower() in {"pass", "passed"}
    )


def canonical_status(value: Any, *, default: str = "unknown") -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    text = str(value or "").strip().lower()
    if text in {"pass", "passed", "success", "succeeded"}:
        return "pass"
    if "timeout" in text or "timed_out" in text:
        return "timeout"
    if text in {"skip", "skipped"}:
        return "skipped"
    if text in {"not_run", "not run", "none"}:
        return "not_run"
    if text in {"fail", "failed", "error", "exception"}:
        return "fail"
    return text or default


def load_references(path: Path) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            implementation = as_dict(record.get("implementation"))
            synth = as_dict(record.get("hls_synth"))
            if (
                record.get("report_type") != "hls_synth"
                or implementation.get("origin") != "hlsfactory_benchmark"
                or synth.get("status") != "pass"
            ):
                continue
            group_path = as_list(as_dict(record.get("problem")).get("group_path"))
            benchmark = normalize_benchmark(group_path[-1] if group_path else "")
            latency = as_dict(
                as_dict(synth.get("PerformanceEstimates")).get(
                    "SummaryOfOverallLatency"
                )
            )
            cycles = positive_number(latency.get("Worst-caseLatency"))
            if not benchmark or cycles is None:
                continue
            if benchmark in references:
                raise ValueError(
                    f"duplicate reference for {benchmark} at {path}:{line_number}"
                )
            run = as_dict(record.get("run"))
            assignments = as_dict(synth.get("UserAssignments"))
            references[benchmark] = {
                "cycles": cycles,
                "source_path": str(path.resolve()),
                "source_line": line_number,
                "vitis_version": run.get("vitis_version"),
                "device": run.get("device"),
                "clock_ns": positive_number(assignments.get("TargetClockPeriod")),
            }
    return references


def source_fields(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "model_family": source["model_family"],
        "model_label": source["model_label"],
        "checkpoint_id": source.get("checkpoint_id"),
        "training_state": source["training_state"],
        "training_recipe": source.get("training_recipe"),
        "agent_assignment": source.get("agent_assignment"),
        "workflow": source["workflow"],
        "strategy": source["strategy"],
        "skill_catalog_version": source.get("skill_catalog_version"),
        "cohort": source["cohort"],
        "one_shot_comparator_id": source.get("one_shot_comparator_id"),
        "base_one_shot_comparator_id": source.get("base_one_shot_comparator_id"),
        "notes": source.get("notes"),
    }


def toolchain_fields(
    run: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    fallback = fallback or {}
    return {
        "vitis_version": run.get("vitis_version") or fallback.get("version"),
        "flow_target": run.get("flow_target") or fallback.get("flow_target"),
        "part": run.get("part") or fallback.get("part"),
        "clock_ns": positive_number(run.get("clock_ns") or fallback.get("clock_ns")),
    }


def toolchain_matches(
    fields: dict[str, Any], required: dict[str, Any]
) -> bool | None:
    if not all(
        fields.get(key) is not None
        for key in ("vitis_version", "flow_target", "part", "clock_ns")
    ):
        return None
    return (
        str(fields["vitis_version"]) == str(required["version"])
        and str(fields["flow_target"]) == str(required["flow_target"])
        and str(fields["part"]) == str(required["part"])
        and math.isclose(
            float(fields["clock_ns"]),
            float(required["clock_ns"]),
            rel_tol=0,
            abs_tol=1e-6,
        )
    )


def selected_step(current: dict[str, Any]) -> dict[str, Any]:
    best = as_dict(current.get("best"))
    name = best.get("step")
    cycles = positive_number(best.get("cycles"))
    steps = [step for step in as_list(current.get("step_cycles")) if isinstance(step, dict)]
    for step in steps:
        if name is not None and step.get("step") == name:
            return step
    for step in steps:
        if cycles is not None and positive_number(step.get("cycles")) == cycles:
            return step
    return {}


def normalize_direct(
    source: dict[str, Any],
    source_path: Path,
    source_hash: str,
    required_toolchain: dict[str, Any],
) -> list[dict[str, Any]]:
    data = load_json(source_path)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(as_list(data.get("rows"))):
        if not isinstance(raw, dict):
            continue
        csim = as_dict(raw.get("csim"))
        synth = as_dict(raw.get("synth"))
        run = toolchain_fields(as_dict(raw.get("vitis")), as_dict(data.get("vitis")))
        csim_status = canonical_status(csim.get("status"))
        csynth_status = canonical_status(synth.get("status"))
        cycles = positive_number(synth.get("latency_cycles"))
        valid = csim_status == "pass" and csynth_status == "pass" and cycles is not None
        setup_id = source["source_id"]
        generated_path = str(raw.get("generated_code_path") or "")
        row = {
            "schema_version": "c2hls.model-comparison-record.v1",
            **source_fields(source),
            "setup_id": setup_id,
            "setup_label": source["label"],
            "skill_mode": "none",
            "benchmark": normalize_benchmark(raw.get("benchmark")),
            "split": VALTEST_SPLITS.get(normalize_benchmark(raw.get("benchmark")), "other"),
            "overall_status": "pass" if valid else csynth_status,
            "generation_status": "pass" if generated_path else "unknown",
            "csim_status": csim_status,
            "csynth_status": csynth_status,
            "valid_csim_csynth": valid,
            "latency_cycles": cycles,
            "latency_cycles_worst": positive_number(synth.get("latency_cycles_worst")),
            "latency_ns": positive_number(synth.get("latency_ns")),
            "fmax_mhz": positive_number(synth.get("fmax_mhz")),
            "bram": positive_number(synth.get("bram")),
            "dsp": positive_number(synth.get("dsp")),
            "ff": positive_number(synth.get("ff")),
            "lut": positive_number(synth.get("lut")),
            "uram": positive_number(synth.get("uram")),
            "phase_b_initial_cycles": None,
            "selected_step": "one_shot",
            "llm_calls": 1,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "selected_rendered_skill_count": 0,
            "selected_rendered_skill_ids": [],
            "candidate_max_rendered_skill_count": 0,
            **run,
            "toolchain_match": toolchain_matches(run, required_toolchain),
            "source_summary_path": str(source_path),
            "source_summary_sha256": source_hash,
            "source_row_index": index,
            "source_result_path": generated_path or None,
            "run_fingerprint_sha256": None,
            "error": synth.get("error") or csim.get("error") or None,
        }
        row["primary_comparable"] = bool(valid and row["toolchain_match"] is True)
        rows.append(row)
    return rows


def result_statuses(
    result: dict[str, Any], current: dict[str, Any], report: dict[str, Any]
) -> tuple[str, str]:
    evaluation = as_dict(result.get("evaluation_status"))
    if evaluation:
        return (
            canonical_status(evaluation.get("correctness_status")),
            canonical_status(evaluation.get("synthesis_status")),
        )
    csim_status = as_dict(result.get("csim_status")).get("generated")
    if csim_status is None:
        csim_status = result.get("correctness_status")
    if csim_status is None:
        step = selected_step(current)
        if step:
            csim_status = step.get("csim")
        elif current.get("baseline_csim") is not None:
            csim_status = current.get("baseline_csim")
    synth_status: Any = "pass" if positive_number(report.get("latency_cycles")) else None
    if synth_status is None and current.get("success") is False:
        synth_status = "fail"
    return canonical_status(csim_status), canonical_status(synth_status)


def normalize_agentic(
    source: dict[str, Any],
    source_path: Path,
    source_hash: str,
    required_toolchain: dict[str, Any],
) -> list[dict[str, Any]]:
    data = load_json(source_path)
    mode_map = as_dict(source.get("skill_mode_map"))
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(as_list(data.get("rows"))):
        if not isinstance(raw, dict):
            continue
        current = as_dict(raw.get("current"))
        raw_mode = str(raw.get("skill_mode") or "unknown")
        mode = str(source.get("skill_mode_override") or mode_map.get(raw_mode) or raw_mode)
        result_path_text = str(current.get("json") or "")
        result_path = Path(result_path_text) if result_path_text else None
        result: dict[str, Any] = {}
        if result_path is not None and result_path.is_file():
            try:
                result = load_json(result_path)
            except (OSError, ValueError, json.JSONDecodeError):
                result = {}

        report = as_dict(result.get("final_report"))
        cycles = positive_number(report.get("latency_cycles"))
        cycle_source = "selected_final_report"
        if cycles is None:
            cycles = positive_number(as_dict(current.get("best")).get("cycles"))
            cycle_source = "summary_best_fallback"
        csim_status, csynth_status = result_statuses(result, current, report)
        valid = (
            current.get("success") is True
            and csim_status == "pass"
            and csynth_status == "pass"
            and cycles is not None
        )
        selected = selected_step(current)
        selected_name = str(as_dict(current.get("best")).get("step") or "unknown")
        prompt = as_dict(selected.get("skill_prompt"))
        selected_ids = [
            value
            for value in as_list(prompt.get("injected_skill_ids"))
            if isinstance(value, str)
        ]
        selected_count = prompt.get("rendered_skill_count")
        if not isinstance(selected_count, int):
            selected_count = prompt.get("injected_skill_count")
        if not isinstance(selected_count, int):
            selected_count = len(selected_ids)
        if selected_name == "baseline":
            selected_count = 0
            selected_ids = []
        candidate_counts: list[int] = []
        for step in as_list(current.get("step_cycles")):
            step_prompt = as_dict(as_dict(step).get("skill_prompt"))
            count = step_prompt.get("rendered_skill_count")
            if not isinstance(count, int):
                count = step_prompt.get("injected_skill_count")
            if not isinstance(count, int):
                count = len(as_list(step_prompt.get("injected_skill_ids")))
            candidate_counts.append(count)

        run_record = as_dict(result.get("run"))
        run = toolchain_fields(run_record)
        usage = as_dict(current.get("llm_usage"))
        setup_id = f"{source['source_id']}__{mode}"
        benchmark = normalize_benchmark(raw.get("bench"))
        error = current.get("error")
        if not error:
            error = as_dict(result.get("tool_failure_status")).get("error")
        overall = "pass" if valid else (
            "provider_error"
            if current.get("phase") == "exception"
            else csynth_status
        )
        row = {
            "schema_version": "c2hls.model-comparison-record.v1",
            **source_fields(source),
            "setup_id": setup_id,
            "setup_label": f"{source['label']} / {mode}",
            "skill_mode": mode,
            "benchmark": benchmark,
            "split": VALTEST_SPLITS.get(benchmark, "other"),
            "overall_status": overall,
            "generation_status": (
                "pass" if current.get("success") is True else "fail"
            ),
            "csim_status": csim_status,
            "csynth_status": csynth_status,
            "valid_csim_csynth": valid,
            "latency_cycles": cycles,
            "latency_cycles_worst": positive_number(
                report.get("latency_cycles_worst")
            ),
            "latency_ns": positive_number(report.get("latency_ns")),
            "fmax_mhz": positive_number(report.get("fmax_mhz")),
            "bram": positive_number(report.get("bram")),
            "dsp": positive_number(report.get("dsp")),
            "ff": positive_number(report.get("ff")),
            "lut": positive_number(report.get("lut")),
            "uram": positive_number(report.get("uram")),
            "phase_b_initial_cycles": positive_number(
                as_dict(result.get("baseline_report")).get("latency_cycles")
                or current.get("baseline_cycles")
            ),
            "selected_step": selected_name,
            "cycle_source": cycle_source,
            "llm_calls": usage.get("calls"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "selected_rendered_skill_count": selected_count,
            "selected_rendered_skill_ids": selected_ids,
            "candidate_max_rendered_skill_count": max(candidate_counts, default=0),
            **run,
            "toolchain_match": toolchain_matches(run, required_toolchain),
            "source_summary_path": str(source_path),
            "source_summary_sha256": source_hash,
            "source_row_index": index,
            "source_result_path": str(result_path) if result_path else None,
            "run_fingerprint_sha256": raw.get("run_fingerprint_sha256"),
            "error": error,
        }
        row["primary_comparable"] = bool(valid and row["toolchain_match"] is True)
        rows.append(row)
    return rows


def attach_comparisons(
    rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]
) -> None:
    valid_lookup = {
        (row["setup_id"], row["benchmark"]): row
        for row in rows
        if row.get("primary_comparable")
    }
    for row in rows:
        reference = references.get(row["benchmark"])
        reference_cycles = reference.get("cycles") if reference else None
        row["reference_cycles_worst"] = reference_cycles
        row["reference_source_path"] = (
            reference.get("source_path") if reference else None
        )
        row["reference_source_line"] = (
            reference.get("source_line") if reference else None
        )
        cycles = positive_number(row.get("latency_cycles"))
        row["speedup_vs_reference_worst"] = (
            float(reference_cycles) / cycles
            if row.get("primary_comparable")
            and cycles is not None
            and positive_number(reference_cycles) is not None
            else None
        )
        for prefix, comparator_field in (
            ("corresponding_one_shot", "one_shot_comparator_id"),
            ("base_one_shot", "base_one_shot_comparator_id"),
        ):
            comparator_id = row.get(comparator_field)
            comparator = valid_lookup.get((comparator_id, row["benchmark"]))
            comparator_cycles = (
                positive_number(comparator.get("latency_cycles")) if comparator else None
            )
            row[f"{prefix}_cycles"] = comparator_cycles
            row[f"speedup_vs_{prefix}"] = (
                comparator_cycles / cycles
                if row.get("primary_comparable")
                and cycles is not None
                and comparator_cycles is not None
                else None
            )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def setup_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["model_family"],
        0 if row["training_state"] == "base" else 1,
        0 if row["workflow"] == "one_shot" else 1,
        0 if row["strategy"] == "direct" else 1 if row["strategy"] == "flash" else 2,
        MODE_ORDER.get(str(row["skill_mode"]), 99),
        row["setup_id"],
    )


def summarize_setups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["setup_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for setup_id, setup_rows in groups.items():
        first = setup_rows[0]
        valid = [row for row in setup_rows if row.get("primary_comparable")]
        corresponding = [
            float(row["speedup_vs_corresponding_one_shot"])
            for row in valid
            if positive_number(row.get("speedup_vs_corresponding_one_shot")) is not None
        ]
        base = [
            float(row["speedup_vs_base_one_shot"])
            for row in valid
            if positive_number(row.get("speedup_vs_base_one_shot")) is not None
        ]
        reference = [
            float(row["speedup_vs_reference_worst"])
            for row in valid
            if positive_number(row.get("speedup_vs_reference_worst")) is not None
        ]
        valtest = [row for row in setup_rows if row["split"] in {"validation", "test"}]
        summaries.append(
            {
                "setup_id": setup_id,
                "setup_label": first["setup_label"],
                "model_family": first["model_family"],
                "training_state": first["training_state"],
                "training_recipe": first.get("training_recipe"),
                "agent_assignment": first.get("agent_assignment"),
                "workflow": first["workflow"],
                "strategy": first["strategy"],
                "skill_mode": first["skill_mode"],
                "skill_catalog_version": first.get("skill_catalog_version"),
                "cohort": first["cohort"],
                "records": len(setup_rows),
                "benchmarks": len({row["benchmark"] for row in setup_rows}),
                "valid_csim_csynth": len(valid),
                "csim_pass": sum(row["csim_status"] == "pass" for row in setup_rows),
                "csynth_pass": sum(row["csynth_status"] == "pass" for row in setup_rows),
                "csynth_timeout": sum(
                    row["csynth_status"] == "timeout" for row in setup_rows
                ),
                "failed_or_not_run": sum(
                    row["overall_status"] != "pass" for row in setup_rows
                ),
                "valtest5_coverage": len(valtest),
                "valtest5_valid": sum(
                    row.get("primary_comparable") is True for row in valtest
                ),
                "paired_corresponding_one_shot": len(corresponding),
                "geomean_speedup_vs_corresponding_one_shot": geomean(corresponding),
                "median_speedup_vs_corresponding_one_shot": (
                    statistics.median(corresponding) if corresponding else None
                ),
                "wins_vs_corresponding_one_shot": sum(
                    value > 1.01 for value in corresponding
                ),
                "losses_vs_corresponding_one_shot": sum(
                    value < 1 / 1.01 for value in corresponding
                ),
                "paired_base_one_shot": len(base),
                "geomean_speedup_vs_base_one_shot": geomean(base),
                "paired_reference": len(reference),
                "geomean_speedup_vs_reference_worst": geomean(reference),
                "source_summary_path": first["source_summary_path"],
                "source_summary_sha256": first["source_summary_sha256"],
            }
        )
    return sorted(summaries, key=setup_sort_key)


def build_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("primary_comparable"):
            groups[(row["model_family"], row["benchmark"])].append(row)
    best_rows: list[dict[str, Any]] = []
    for (model_family, benchmark), values in sorted(groups.items()):
        best = min(values, key=lambda row: float(row["latency_cycles"]))
        base = next(
            (
                row
                for row in values
                if row["setup_id"] == row.get("base_one_shot_comparator_id")
            ),
            None,
        )
        best_rows.append(
            {
                "model_family": model_family,
                "benchmark": benchmark,
                "split": best["split"],
                "best_setup_id": best["setup_id"],
                "best_setup_label": best["setup_label"],
                "best_training_state": best["training_state"],
                "best_workflow": best["workflow"],
                "best_strategy": best["strategy"],
                "best_skill_mode": best["skill_mode"],
                "best_cycles": best["latency_cycles"],
                "base_one_shot_cycles": (
                    base.get("latency_cycles") if base else best.get("base_one_shot_cycles")
                ),
                "speedup_vs_base_one_shot": best.get("speedup_vs_base_one_shot"),
                "reference_cycles_worst": best.get("reference_cycles_worst"),
                "speedup_vs_reference_worst": best.get("speedup_vs_reference_worst"),
                "valid_setup_candidates": len(values),
            }
        )
    return best_rows


def write_wide(
    path: Path,
    rows: list[dict[str, Any]],
    setup_summaries: list[dict[str, Any]],
    value_field: str,
    benchmark_filter: set[str] | None = None,
) -> None:
    setup_ids = [summary["setup_id"] for summary in setup_summaries]
    lookup = {(row["benchmark"], row["setup_id"]): row for row in rows}
    benchmarks = sorted(
        {
            row["benchmark"]
            for row in rows
            if benchmark_filter is None or row["benchmark"] in benchmark_filter
        }
    )
    wide_rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        representative = next(row for row in rows if row["benchmark"] == benchmark)
        output: dict[str, Any] = {
            "benchmark": benchmark,
            "split": representative["split"],
            "reference_cycles_worst": representative.get("reference_cycles_worst"),
        }
        for setup_id in setup_ids:
            row = lookup.get((benchmark, setup_id))
            if row is None:
                output[setup_id] = None
            elif value_field == "latency_cycles":
                output[setup_id] = (
                    row.get(value_field) if row.get("primary_comparable") else None
                )
            else:
                output[setup_id] = row.get(value_field)
        wide_rows.append(output)
    write_csv(path, wide_rows)


def fmt_ratio(value: Any) -> str:
    number = positive_number(value)
    return f"{number:.3f}x" if number is not None else "-"


def write_report(
    path: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    setup_summaries: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    results_dir: Path,
    artifact_root: Path,
) -> None:
    qwen = [row for row in rows if row["model_family"] == "qwen3_6_27b"]
    gemma = [row for row in rows if row["model_family"] == "gemma4_31b"]
    lines = [
        "# Qwen/Gemma One-Shot, Agentic, and SFT Comparison",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "- Validity gate: CSim pass, Vitis HLS CSynth pass, positive latency, and exact Vitis 2023.2/U280/3.33 ns toolchain match.",
        "- Primary generated latency is the parser's average-case CSynth latency, with Vitis worst-case fallback when average is unavailable.",
        "- Failed, timed-out, and missing runs remain in the long/status tables; their cycle cells are blank.",
        "- No COSIM result is required for this comparison.",
        "",
        "## Coverage",
        "",
        "| model | records | valid | setups | one-shot records | agentic records | trained-checkpoint records |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, model_rows in (("Qwen3.6-27B", qwen), ("Gemma-4-31B", gemma)):
        lines.append(
            f"| {label} | {len(model_rows)} | "
            f"{sum(row['primary_comparable'] for row in model_rows)} | "
            f"{len({row['setup_id'] for row in model_rows})} | "
            f"{sum(row['workflow'] == 'one_shot' for row in model_rows)} | "
            f"{sum(row['workflow'] == 'agentic' for row in model_rows)} | "
            f"{sum(row['training_state'] == 'sft' for row in model_rows)} |"
        )

    lines.extend(
        [
            "",
            "## Setup Summary",
            "",
            "| setup | valid/records | val+test valid/coverage | paired one-shot | geomean vs corresponding one-shot | geomean vs base one-shot | geomean vs reference |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in setup_summaries:
        lines.append(
            f"| `{summary['setup_id']}` | "
            f"{summary['valid_csim_csynth']}/{summary['records']} | "
            f"{summary['valtest5_valid']}/{summary['valtest5_coverage']} | "
            f"{summary['paired_corresponding_one_shot']} | "
            f"{fmt_ratio(summary['geomean_speedup_vs_corresponding_one_shot'])} | "
            f"{fmt_ratio(summary['geomean_speedup_vs_base_one_shot'])} | "
            f"{fmt_ratio(summary['geomean_speedup_vs_reference_worst'])} |"
        )

    top = sorted(
        [
            row
            for row in best_rows
            if positive_number(row.get("speedup_vs_base_one_shot")) is not None
        ],
        key=lambda row: float(row["speedup_vs_base_one_shot"]),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Largest Best-of-Available Gains",
            "",
            "| model | benchmark | best setup | cycles | base one-shot | speedup |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in top[:20]:
        lines.append(
            f"| {row['model_family']} | {row['benchmark']} | "
            f"`{row['best_setup_id']}` | {float(row['best_cycles']):,.0f} | "
            f"{float(row['base_one_shot_cycles']):,.0f} | "
            f"{fmt_ratio(row['speedup_vs_base_one_shot'])} |"
        )

    availability = as_dict(manifest.get("sft_availability"))
    lines.extend(
        [
            "",
            "## SFT Provenance",
            "",
            f"- Qwen: {availability.get('qwen3_6_27b', 'not documented')}",
            f"- Gemma: {availability.get('gemma4_31b', 'not documented')}",
            "",
            "## Data Files",
            "",
            f"- Full long table: `{(results_dir / 'comparison_records.csv').resolve()}`",
            f"- Full cycle matrix: `{(results_dir / 'comparison_cycles_wide.csv').resolve()}`",
            f"- Full status matrix: `{(results_dir / 'comparison_status_wide.csv').resolve()}`",
            f"- Setup summary: `{(results_dir / 'setup_summary.csv').resolve()}`",
            f"- Per-benchmark best: `{(results_dir / 'benchmark_best.csv').resolve()}`",
            f"- Qwen normalized folder: `{(artifact_root / 'qwen3_6_27b').resolve()}`",
            f"- Gemma normalized folder: `{(artifact_root / 'gemma4_31b').resolve()}`",
            "",
            "## Interpretation Limits",
            "",
            "- Setup rows come from distinct campaigns and model samples. Compare only paired benchmark rows; do not average raw cycles across different benchmark sets.",
            "- Legacy skill-v1, skill-v2, role-selective SFT, and all-role SFT are separate setup fingerprints.",
            "- The reference column is reporting-only and uses website-schema worst-case cycles, while generated latency uses average-case with worst-case fallback.",
            "- Gemma SFT rows are absent because no Gemma adapter checkpoint completed; the July 25 load gates failed before training.",
            "- Best-of-available rows are an oracle over recorded setups, not a deployable router result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_readme(
    path: Path,
    model_family: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        f"# {model_family} Normalized Comparison Data",
        "",
        f"- Records: **{len(rows)}**",
        f"- Valid CSim+CSynth+toolchain rows: **{sum(row['primary_comparable'] for row in rows)}**",
        f"- Setup fingerprints: **{len(summaries)}**",
        "- `one_shot/` contains compact normalized copies of each one-shot source.",
        "- `records.jsonl` and `records.csv` include one-shot, agentic, base, and available SFT records.",
        "- `cycles_wide.csv` leaves invalid or unexecuted measurements blank; `status_wide.csv` preserves their states.",
        "",
        "## Setups",
        "",
        "| setup | training | workflow | strategy | skill mode | valid/records |",
        "|---|---|---|---|---|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['setup_id']}` | {summary['training_state']} | "
            f"{summary['workflow']} | {summary['strategy']} | "
            f"{summary['skill_mode']} | "
            f"{summary['valid_csim_csynth']}/{summary['records']} |"
        )
    availability = as_dict(manifest.get("sft_availability")).get(model_family)
    if availability:
        lines.extend(["", "## SFT Status", "", f"{availability}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    required = {
        "schema_version",
        "setup_id",
        "model_family",
        "training_state",
        "workflow",
        "strategy",
        "skill_mode",
        "benchmark",
        "csim_status",
        "csynth_status",
        "valid_csim_csynth",
        "primary_comparable",
        "source_summary_path",
        "source_summary_sha256",
    }
    keys: set[tuple[str, str]] = set()
    errors: list[str] = []
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"row {index}: missing {missing}")
        key = (str(row.get("setup_id")), str(row.get("benchmark")))
        if key in keys:
            errors.append(f"row {index}: duplicate setup/benchmark {key}")
        keys.add(key)
        if row.get("primary_comparable") and not (
            row.get("valid_csim_csynth")
            and row.get("toolchain_match") is True
            and positive_number(row.get("latency_cycles")) is not None
        ):
            errors.append(f"row {index}: invalid primary-comparable gate")
    if errors:
        raise ValueError("\n".join(errors[:20]))


def build_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "c2hls.model-comparison-artifact-manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-schema", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    manifest_dir = manifest_path.parent
    required_toolchain = as_dict(manifest.get("required_toolchain"))
    reference_path = (
        args.reference_schema.resolve()
        if args.reference_schema
        else resolve_path(str(manifest["reference_schema"]), manifest_dir)
    )
    references = load_references(reference_path)

    rows: list[dict[str, Any]] = []
    for source in as_list(manifest.get("sources")):
        if not isinstance(source, dict):
            continue
        source_path = resolve_path(str(source["path"]), manifest_dir)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source: {source_path}")
        source_hash = sha256_file(source_path)
        if source["kind"] == "one_shot":
            rows.extend(
                normalize_direct(
                    source, source_path, source_hash, required_toolchain
                )
            )
        elif source["kind"] == "agentic":
            rows.extend(
                normalize_agentic(
                    source, source_path, source_hash, required_toolchain
                )
            )
        else:
            raise ValueError(f"unknown source kind: {source['kind']}")

    attach_comparisons(rows, references)
    rows.sort(key=lambda row: (*setup_sort_key(row), row["benchmark"]))
    validate_rows(rows)
    setup_summaries = summarize_setups(rows)
    best_rows = build_best_rows(rows)

    artifact_root = args.artifact_root.resolve()
    results_dir = args.results_dir.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_snapshot = {
        **manifest,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "reference_schema": str(reference_path),
        "reference_schema_sha256": sha256_file(reference_path),
    }
    (results_dir / "manifest_snapshot.json").write_text(
        json.dumps(manifest_snapshot, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(results_dir / "comparison_records.jsonl", rows)
    write_csv(results_dir / "comparison_records.csv", rows)
    write_csv(results_dir / "setup_summary.csv", setup_summaries)
    write_csv(results_dir / "benchmark_best.csv", best_rows)
    write_wide(
        results_dir / "comparison_cycles_wide.csv",
        rows,
        setup_summaries,
        "latency_cycles",
    )
    write_wide(
        results_dir / "comparison_status_wide.csv",
        rows,
        setup_summaries,
        "overall_status",
    )
    write_wide(
        results_dir / "valtest5_cycles_wide.csv",
        rows,
        setup_summaries,
        "latency_cycles",
        set(VALTEST_SPLITS),
    )

    for model_family in ("qwen3_6_27b", "gemma4_31b"):
        model_rows = [row for row in rows if row["model_family"] == model_family]
        model_summaries = [
            summary
            for summary in setup_summaries
            if summary["model_family"] == model_family
        ]
        model_dir = artifact_root / model_family
        model_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(model_dir / "records.jsonl", model_rows)
        write_csv(model_dir / "records.csv", model_rows)
        write_csv(model_dir / "setup_summary.csv", model_summaries)
        write_wide(
            model_dir / "cycles_wide.csv",
            model_rows,
            model_summaries,
            "latency_cycles",
        )
        write_wide(
            model_dir / "status_wide.csv",
            model_rows,
            model_summaries,
            "overall_status",
        )
        for source_id in sorted(
            {
                row["source_id"]
                for row in model_rows
                if row["workflow"] == "one_shot"
            }
        ):
            direct_rows = [row for row in model_rows if row["source_id"] == source_id]
            write_jsonl(model_dir / "one_shot" / f"{source_id}.jsonl", direct_rows)
        write_model_readme(
            model_dir / "README.md",
            model_family,
            model_rows,
            model_summaries,
            manifest,
        )

    write_report(
        results_dir / "report.md",
        manifest,
        rows,
        setup_summaries,
        best_rows,
        results_dir,
        artifact_root,
    )
    verification = {
        "schema_version": "c2hls.model-comparison-verification.v1",
        "record_count": len(rows),
        "setup_count": len(setup_summaries),
        "benchmark_count": len({row["benchmark"] for row in rows}),
        "primary_comparable_count": sum(
            row["primary_comparable"] for row in rows
        ),
        "duplicate_setup_benchmark_keys": 0,
        "toolchain_match_true": sum(
            row.get("toolchain_match") is True for row in rows
        ),
        "toolchain_match_false": sum(
            row.get("toolchain_match") is False for row in rows
        ),
        "toolchain_match_unknown": sum(
            row.get("toolchain_match") is None for row in rows
        ),
        "model_counts": {
            model_family: {
                "records": sum(
                    row["model_family"] == model_family for row in rows
                ),
                "valid": sum(
                    row["model_family"] == model_family
                    and row["primary_comparable"]
                    for row in rows
                ),
                "setups": len(
                    {
                        row["setup_id"]
                        for row in rows
                        if row["model_family"] == model_family
                    }
                ),
                "one_shot_records": sum(
                    row["model_family"] == model_family
                    and row["workflow"] == "one_shot"
                    for row in rows
                ),
                "agentic_records": sum(
                    row["model_family"] == model_family
                    and row["workflow"] == "agentic"
                    for row in rows
                ),
                "sft_records": sum(
                    row["model_family"] == model_family
                    and row["training_state"] == "sft"
                    for row in rows
                ),
            }
            for model_family in ("qwen3_6_27b", "gemma4_31b")
        },
        "source_counts": {
            source_id: sum(row["source_id"] == source_id for row in rows)
            for source_id in sorted({row["source_id"] for row in rows})
        },
        "required_toolchain": required_toolchain,
        "manifest_sha256": sha256_file(manifest_path),
        "reference_schema_sha256": sha256_file(reference_path),
    }
    (results_dir / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    artifact_manifest = build_artifact_manifest(results_dir)
    (results_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"records={len(rows)}")
    print(f"setups={len(setup_summaries)}")
    print(f"valid={sum(row['primary_comparable'] for row in rows)}")
    print(f"results={results_dir}")
    print(f"model_data={artifact_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
