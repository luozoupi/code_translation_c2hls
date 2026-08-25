#!/usr/bin/env python3
"""Export the completed Qwen/Gemma skill-v3 sweep as traceable tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_MODES = (
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
)
STRATEGIES = ("flash", "multistep")
MODE_ORDER = {name: index for index, name in enumerate(SKILL_MODES)}
STRATEGY_ORDER = {name: index for index, name in enumerate(STRATEGIES)}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def positive_number(value: Any) -> float | None:
    parsed = number(value)
    return parsed if parsed is not None and parsed > 0 else None


def canonical_benchmark(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    if not name:
        return ""
    return name if name.startswith("hlsfactory_") else f"hlsfactory_{name}"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geomean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0 and math.isfinite(value)]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def status_passed(value: Any) -> bool:
    if value is True:
        return True
    return str(value or "").strip().lower() in {
        "pass",
        "passed",
        "success",
        "succeeded",
    }


def load_schema_references(path: Path) -> dict[str, dict[str, Any]]:
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
            benchmark = canonical_benchmark(group_path[-1] if group_path else "")
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
                    f"duplicate schema reference for {benchmark} at {path}:{line_number}"
                )
            assignments = as_dict(synth.get("UserAssignments"))
            run = as_dict(record.get("run"))
            references[benchmark] = {
                "cycles": cycles,
                "source_kind": "website_schema",
                "source_path": str(path.resolve()),
                "source_line": line_number,
                "vitis_version": run.get("vitis_version"),
                "device": run.get("device"),
                "clock_ns": positive_number(assignments.get("TargetClockPeriod")),
            }
    return references


def extend_references_from_cache(
    references: dict[str, dict[str, Any]], cache_dir: Path
) -> None:
    candidates: dict[str, list[tuple[str, Path, float, dict[str, Any]]]] = defaultdict(
        list
    )
    for path in sorted(cache_dir.glob("hlsfactory_*.json")):
        try:
            data = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        configuration = as_dict(data.get("configuration"))
        validation = as_dict(data.get("reference_validation"))
        report = as_dict(validation.get("report"))
        csim = as_dict(validation.get("csim"))
        benchmark = canonical_benchmark(configuration.get("benchmark"))
        cycles = positive_number(
            report.get("latency_cycles_worst") or report.get("latency_cycles")
        )
        if (
            not benchmark
            or cycles is None
            or validation.get("benchmark_ready") is not True
            or csim.get("passed") is not True
        ):
            continue
        candidates[benchmark].append(
            (str(data.get("created_at") or ""), path, cycles, configuration)
        )

    for benchmark, values in candidates.items():
        if benchmark in references:
            continue
        _, path, cycles, configuration = sorted(values)[-1]
        references[benchmark] = {
            "cycles": cycles,
            "source_kind": "reference_validation_cache_not_schema",
            "source_path": str(path.resolve()),
            "source_line": None,
            "vitis_version": configuration.get("vitis_version"),
            "device": configuration.get("part"),
            "clock_ns": positive_number(configuration.get("clock_ns")),
        }


def selected_step(current: dict[str, Any]) -> dict[str, Any]:
    best = as_dict(current.get("best"))
    best_name = best.get("step")
    best_cycles = positive_number(best.get("cycles"))
    steps = [step for step in as_list(current.get("step_cycles")) if isinstance(step, dict)]
    for step in steps:
        if best_name is not None and step.get("step") == best_name:
            return step
    for step in steps:
        if best_cycles is not None and positive_number(step.get("cycles")) == best_cycles:
            return step
    return {}


def string_ids(value: Any) -> list[str]:
    return [item for item in as_list(value) if isinstance(item, str)]


def prompt_count(prompt: dict[str, Any], field: str, ids_field: str) -> int:
    value = prompt.get(field)
    return value if isinstance(value, int) and value >= 0 else len(string_ids(prompt.get(ids_field)))


def normalize_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    summary_path = Path(source["summary_path"])
    state_path = Path(source["state_path"])
    summary = load_json(summary_path)
    state = load_json(state_path)
    rows: list[dict[str, Any]] = []

    for index, raw in enumerate(as_list(summary.get("rows"))):
        if not isinstance(raw, dict):
            continue
        current = as_dict(raw.get("current"))
        result_path_text = str(current.get("json") or "")
        result_path = Path(result_path_text) if result_path_text else None
        result = load_json(result_path) if result_path and result_path.is_file() else {}
        final_report = as_dict(result.get("final_report"))
        baseline_report = as_dict(result.get("baseline_report"))
        evaluation = as_dict(result.get("evaluation_status"))
        feasibility = as_dict(result.get("candidate_feasibility"))
        audit = as_dict(result.get("reference_isolation_audit"))
        run = as_dict(result.get("run"))
        toolchain = as_dict(run.get("toolchain"))

        controller_cycles = positive_number(
            final_report.get("latency_cycles")
            or as_dict(current.get("best")).get("cycles")
        )
        worst_cycles = positive_number(
            final_report.get("latency_cycles_worst") or controller_cycles
        )
        baseline_controller_cycles = positive_number(
            baseline_report.get("latency_cycles") or current.get("baseline_cycles")
        )
        baseline_worst_cycles = positive_number(
            baseline_report.get("latency_cycles_worst")
            or baseline_controller_cycles
        )
        step = selected_step(current)
        best_name = str(as_dict(current.get("best")).get("step") or "unknown")
        prompt = as_dict(step.get("skill_prompt")) if best_name != "baseline" else {}
        rendered_ids = string_ids(prompt.get("rendered_skill_ids"))
        declared_ids = string_ids(prompt.get("declared_applied_skill_ids"))
        verified_ids = string_ids(prompt.get("verified_applied_skill_ids"))
        synthesized_ids = string_ids(prompt.get("synthesized_candidate_skill_ids"))
        candidate_prompts = [
            as_dict(as_dict(candidate).get("skill_prompt"))
            for candidate in as_list(current.get("step_cycles"))
            if isinstance(candidate, dict)
        ]
        candidate_rendered_counts = [
            prompt_count(item, "rendered_skill_count", "rendered_skill_ids")
            for item in candidate_prompts
        ]
        correctness_ok = (
            evaluation.get("correctness_status") == "passed"
            if evaluation
            else status_passed(step.get("csim") or current.get("baseline_csim"))
        )
        synthesis_ok = (
            evaluation.get("synthesis_status") == "passed"
            if evaluation
            else worst_cycles is not None
        )
        timing_met = feasibility.get("timing_met")
        if timing_met is None:
            slack = number(final_report.get("slack_ns"))
            timing_met = slack is not None and slack >= 0
        resource_fit = feasibility.get("resource_fit")
        if resource_fit is None:
            resource_fit = True if final_report else None
        valid = bool(
            current.get("success") is True
            and correctness_ok
            and synthesis_ok
            and worst_cycles is not None
        )
        comparison_valid = bool(valid and timing_met is True and resource_fit is True)
        usage = as_dict(current.get("llm_usage"))
        benchmark = canonical_benchmark(raw.get("bench"))
        mode = str(raw.get("skill_mode") or "unknown")
        setup_id = f"{source['model_key']}__{source['strategy']}__{mode}"
        rows.append(
            {
                "schema_version": "c2hls.skill-v3-sweep-table.v1",
                "model": source["model"],
                "model_key": source["model_key"],
                "model_revision": source["model_revision"],
                "strategy": source["strategy"],
                "skill_mode": mode,
                "setup_id": setup_id,
                "benchmark": benchmark,
                "controller_success": current.get("success") is True,
                "csim_status": "pass" if correctness_ok else "fail",
                "csynth_status": "pass" if synthesis_ok else "fail",
                "cosim_status": evaluation.get("cosim_execution_status", "not_run"),
                "valid_csim_csynth": valid,
                "timing_met": timing_met,
                "resource_fit": resource_fit,
                "comparison_valid": comparison_valid,
                "selected_step": best_name,
                "selected_baseline": best_name == "baseline",
                "controller_latency_cycles": controller_cycles,
                "worst_case_latency_cycles": worst_cycles,
                "latency_ns": positive_number(final_report.get("latency_ns")),
                "worst_case_latency_ns": positive_number(
                    final_report.get("latency_ns_worst")
                ),
                "interval_cycles": positive_number(final_report.get("interval")),
                "phase_b_controller_cycles": baseline_controller_cycles,
                "phase_b_worst_case_cycles": baseline_worst_cycles,
                "speedup_vs_phase_b_worst": (
                    baseline_worst_cycles / worst_cycles
                    if comparison_valid
                    and baseline_worst_cycles is not None
                    and worst_cycles is not None
                    else None
                ),
                "requested_clock_ns": positive_number(
                    final_report.get("requested_clock_period_ns")
                    or toolchain.get("clock_ns")
                ),
                "estimated_clock_ns": positive_number(
                    final_report.get("estimated_clock_period_ns")
                ),
                "slack_ns": number(final_report.get("slack_ns")),
                "fmax_mhz": positive_number(final_report.get("fmax_mhz")),
                "bram": number(final_report.get("bram")),
                "dsp": number(final_report.get("dsp")),
                "ff": number(final_report.get("ff")),
                "lut": number(final_report.get("lut")),
                "uram": number(final_report.get("uram")),
                "steps_attempted": current.get("steps_attempted"),
                "steps_success": current.get("steps_success"),
                "llm_calls": usage.get("calls"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "selected_rendered_skill_count": (
                    0
                    if best_name == "baseline"
                    else prompt_count(
                        prompt, "rendered_skill_count", "rendered_skill_ids"
                    )
                ),
                "selected_rendered_skill_ids": rendered_ids,
                "selected_declared_applied_skill_count": (
                    0
                    if best_name == "baseline"
                    else prompt_count(
                        prompt,
                        "declared_applied_skill_count",
                        "declared_applied_skill_ids",
                    )
                ),
                "selected_declared_applied_skill_ids": declared_ids,
                "selected_verified_applied_skill_count": (
                    0
                    if best_name == "baseline"
                    else prompt_count(
                        prompt,
                        "verified_applied_skill_count",
                        "verified_applied_skill_ids",
                    )
                ),
                "selected_verified_applied_skill_ids": verified_ids,
                "selected_synthesized_skill_count": (
                    0
                    if best_name == "baseline"
                    else prompt_count(
                        prompt,
                        "synthesized_candidate_skill_count",
                        "synthesized_candidate_skill_ids",
                    )
                ),
                "selected_synthesized_skill_ids": synthesized_ids,
                "candidate_max_rendered_skill_count": max(
                    candidate_rendered_counts, default=0
                ),
                "reference_isolation_audit_passed": audit.get("passed"),
                "reference_isolation_finding_count": audit.get("finding_count"),
                "run_fingerprint_sha256": raw.get("run_fingerprint_sha256"),
                "summary_row_index": index,
                "source_summary_path": str(summary_path.resolve()),
                "source_result_path": str(result_path.resolve()) if result_path else None,
                "wrapper_state_status": state.get("status"),
                "wrapper_state_exit_code": state.get("exit_code"),
                "wrapper_state_path": str(state_path.resolve()),
            }
        )
    return rows


def attach_references(
    rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        reference = references.get(row["benchmark"])
        reference_cycles = positive_number(reference.get("cycles")) if reference else None
        cycles = positive_number(row.get("worst_case_latency_cycles"))
        row["reference_worst_case_cycles"] = reference_cycles
        row["reference_source_kind"] = reference.get("source_kind") if reference else None
        row["reference_source_path"] = reference.get("source_path") if reference else None
        row["reference_source_line"] = reference.get("source_line") if reference else None
        row["speedup_vs_reference_worst"] = (
            reference_cycles / cycles
            if row.get("comparison_valid")
            and reference_cycles is not None
            and cycles is not None
            else None
        )


def attach_skillless_effects(rows: list[dict[str, Any]]) -> None:
    controls = {
        (row["model_key"], row["strategy"], row["benchmark"]): row
        for row in rows
        if row["skill_mode"] == "skillless" and row.get("comparison_valid")
    }
    for row in rows:
        control = controls.get((row["model_key"], row["strategy"], row["benchmark"]))
        control_cycles = (
            positive_number(control.get("worst_case_latency_cycles")) if control else None
        )
        cycles = positive_number(row.get("worst_case_latency_cycles"))
        row["paired_skillless_worst_case_cycles"] = control_cycles
        row["speedup_vs_paired_skillless_worst"] = (
            control_cycles / cycles
            if row.get("comparison_valid")
            and control_cycles is not None
            and cycles is not None
            else None
        )


def build_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("comparison_valid"):
            groups[(row["model_key"], row["benchmark"])].append(row)
    winners: list[dict[str, Any]] = []
    for (model_key, benchmark), candidates in sorted(groups.items()):
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row["worst_case_latency_cycles"]),
                STRATEGY_ORDER.get(row["strategy"], 99),
                MODE_ORDER.get(row["skill_mode"], 99),
            ),
        )
        winner = ranked[0]
        best_cycles = float(winner["worst_case_latency_cycles"])
        tie_count = sum(
            math.isclose(
                float(candidate["worst_case_latency_cycles"]),
                best_cycles,
                rel_tol=0,
                abs_tol=0,
            )
            for candidate in ranked
        )
        winners.append(
            {
                "model": winner["model"],
                "model_key": model_key,
                "benchmark": benchmark,
                "winning_setup_id": winner["setup_id"],
                "winning_strategy": winner["strategy"],
                "winning_skill_mode": winner["skill_mode"],
                "winning_selected_step": winner["selected_step"],
                "winning_worst_case_cycles": best_cycles,
                "winning_controller_cycles": winner["controller_latency_cycles"],
                "reference_worst_case_cycles": winner.get(
                    "reference_worst_case_cycles"
                ),
                "speedup_vs_reference_worst": winner.get(
                    "speedup_vs_reference_worst"
                ),
                "selected_rendered_skill_count": winner.get(
                    "selected_rendered_skill_count"
                ),
                "selected_verified_applied_skill_count": winner.get(
                    "selected_verified_applied_skill_count"
                ),
                "tie_count": tie_count,
                "candidate_count": len(ranked),
                "source_result_path": winner.get("source_result_path"),
            }
        )
    return winners


def build_setup_summaries(
    rows: list[dict[str, Any]], winners: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    winner_counts = Counter(winner["winning_setup_id"] for winner in winners)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["setup_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for setup_id, setup_rows in groups.items():
        first = setup_rows[0]
        valid = [row for row in setup_rows if row.get("comparison_valid")]
        reference_speedups = [
            float(row["speedup_vs_reference_worst"])
            for row in valid
            if positive_number(row.get("speedup_vs_reference_worst")) is not None
        ]
        skillless_speedups = [
            float(row["speedup_vs_paired_skillless_worst"])
            for row in valid
            if positive_number(row.get("speedup_vs_paired_skillless_worst"))
            is not None
        ]
        rendered = [
            int(row["selected_rendered_skill_count"])
            for row in valid
            if isinstance(row.get("selected_rendered_skill_count"), int)
        ]
        summaries.append(
            {
                "setup_id": setup_id,
                "model": first["model"],
                "model_key": first["model_key"],
                "strategy": first["strategy"],
                "skill_mode": first["skill_mode"],
                "completed": len(setup_rows),
                "valid_csim_csynth_timing_resource": len(valid),
                "baseline_selected": sum(row.get("selected_baseline") for row in setup_rows),
                "best_of_10_wins": winner_counts[setup_id],
                "geomean_speedup_vs_reference_worst": geomean(reference_speedups),
                "geomean_speedup_vs_paired_skillless_worst": geomean(
                    skillless_speedups
                ),
                "wins_vs_paired_skillless_over_1pct": sum(
                    value > 1.01 for value in skillless_speedups
                ),
                "ties_vs_paired_skillless_within_1pct": sum(
                    1 / 1.01 <= value <= 1.01 for value in skillless_speedups
                ),
                "losses_vs_paired_skillless_over_1pct": sum(
                    value < 1 / 1.01 for value in skillless_speedups
                ),
                "median_selected_rendered_skill_count": (
                    statistics.median(rendered) if rendered else None
                ),
                "reference_isolation_audit_failures": sum(
                    row.get("reference_isolation_audit_passed") is False
                    for row in setup_rows
                ),
                "wrapper_state_status": first.get("wrapper_state_status"),
                "wrapper_state_exit_code": first.get("wrapper_state_exit_code"),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["model_key"],
            STRATEGY_ORDER.get(row["strategy"], 99),
            MODE_ORDER.get(row["skill_mode"], 99),
        ),
    )


def build_wide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    benchmarks = sorted({row["benchmark"] for row in rows})
    lookup = {(row["benchmark"], row["setup_id"]): row for row in rows}
    setups = sorted(
        {row["setup_id"] for row in rows},
        key=lambda setup_id: (
            setup_id.split("__")[0],
            STRATEGY_ORDER.get(setup_id.split("__")[1], 99),
            MODE_ORDER.get(setup_id.split("__")[2], 99),
        ),
    )
    wide: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        sample = next(row for row in rows if row["benchmark"] == benchmark)
        record: dict[str, Any] = {
            "benchmark": benchmark,
            "reference_worst_case_cycles": sample.get("reference_worst_case_cycles"),
            "reference_source_kind": sample.get("reference_source_kind"),
        }
        for setup_id in setups:
            row = lookup.get((benchmark, setup_id))
            record[setup_id] = (
                row.get("worst_case_latency_cycles")
                if row and row.get("comparison_valid")
                else None
            )
        wide.append(record)
    return wide


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def fmt_float(value: Any) -> str:
    parsed = positive_number(value)
    return f"{parsed:.3f}x" if parsed is not None else "-"


def write_readme(
    path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Skill-v3 Qwen/Gemma Sweep Tables",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"- Complete setup cells: **{len(rows)}/540**.",
        f"- Deterministic per-model benchmark winners: **{len(winners)}/54**.",
        f"- Reference coverage: **{len(references)}/27** kernels.",
        "- Toolchain: Vitis 2023.2, `xcu280-fsvh2892-2L-e`, 3.33 ns.",
        "- Validation: CSim and CSynth; COSIM was intentionally not run.",
        "- Primary comparison metric: Vitis worst-case latency cycles. The controller latency field is retained separately.",
        "",
        "## Setup Summary",
        "",
        "| model | strategy | skill mode | valid | best-of-10 wins | geomean vs reference | geomean vs skillless | W/T/L vs skillless | baseline selected | median rendered |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['strategy']} | {summary['skill_mode']} | "
            f"{summary['valid_csim_csynth_timing_resource']}/{summary['completed']} | "
            f"{summary['best_of_10_wins']} | "
            f"{fmt_float(summary['geomean_speedup_vs_reference_worst'])} | "
            f"{fmt_float(summary['geomean_speedup_vs_paired_skillless_worst'])} | "
            f"{summary['wins_vs_paired_skillless_over_1pct']}/"
            f"{summary['ties_vs_paired_skillless_within_1pct']}/"
            f"{summary['losses_vs_paired_skillless_over_1pct']} | "
            f"{summary['baseline_selected']} | "
            f"{summary['median_selected_rendered_skill_count'] if summary['median_selected_rendered_skill_count'] is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `all_rows.csv` and `all_rows.jsonl`: one record per model/strategy/skill-mode/benchmark cell.",
            "- `worst_cycles_wide.csv`: 27 benchmarks by 20 setup columns.",
            "- `setup_summary.csv`: aggregate coverage, speedups, and paired skill effects.",
            "- `winners.csv`: lowest feasible worst-case setup for each model and benchmark.",
            "- `manifest.json`: source paths, hashes, wrapper states, and table invariants.",
            "",
            "## Publication Caveats",
            "",
            "- Qwen produced all 135 multistep rows and valid JSONL, but its outer launcher recorded exit code 2 after report generation. The row data are retained with the wrapper status visible.",
            "- The advisory reference-isolation auditor flagged all rows for `unlabeled_reference_metric`. Reference metrics were configured off for generation, but these findings need classification before publication-grade causal claims.",
            "- Skill-mode rows are one deterministic model sample each. They support setup comparison and router training, while repeated samples are still required for variance estimates.",
            "- Cache-only references are labeled separately from website-schema references.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def default_sources(repo: Path) -> list[dict[str, Any]]:
    artifacts = repo / "artifacts"
    states = artifacts / "experiment_matrix" / "skill_v3_fc27133"
    prefix = "agentic_no_streamcluster_skillv3_no_rmw_fc27133"
    return [
        {
            "model": "Qwen3.6-27B",
            "model_key": "qwen3_6_27b",
            "model_revision": "Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "strategy": "flash",
            "summary_path": str(
                artifacts
                / f"{prefix}_qwen36_27b_base_local_flash_skills5_nocosim_shapeaudit1_20260729.summary.json"
            ),
            "state_path": str(states / "qwen_state.json"),
        },
        {
            "model": "Qwen3.6-27B",
            "model_key": "qwen3_6_27b",
            "model_revision": "Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "strategy": "multistep",
            "summary_path": str(
                artifacts
                / f"{prefix}_qwen36_27b_base_local_dynamic_skills5_nocosim_shapeaudit1_20260729.summary.json"
            ),
            "state_path": str(states / "qwen_state.json"),
        },
        {
            "model": "Gemma-4-31B",
            "model_key": "gemma4_31b",
            "model_revision": "google/gemma-4-31B-it@vllm-0.20.2-20260730",
            "strategy": "flash",
            "summary_path": str(
                artifacts
                / f"{prefix}_gemma4_31b_base_remote_flash_skills5_nocosim_shapeaudit1_20260729.summary.json"
            ),
            "state_path": str(states / "gemma_state.json"),
        },
        {
            "model": "Gemma-4-31B",
            "model_key": "gemma4_31b",
            "model_revision": "google/gemma-4-31B-it@vllm-0.20.2-20260730",
            "strategy": "multistep",
            "summary_path": str(
                artifacts
                / f"{prefix}_gemma4_31b_base_remote_dynamic_skills5_nocosim_shapeaudit1_endpoint_rerun1_20260730.summary.json"
            ),
            "state_path": str(states / "gemma_dynamic_rerun1_state.json"),
        },
    ]


def validate_matrix(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 540:
        raise ValueError(f"expected 540 sweep rows, found {len(rows)}")
    keys = {
        (row["model_key"], row["strategy"], row["skill_mode"], row["benchmark"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("duplicate model/strategy/skill-mode/benchmark rows")
    benchmarks = {row["benchmark"] for row in rows}
    if len(benchmarks) != 27:
        raise ValueError(f"expected 27 benchmarks, found {len(benchmarks)}")
    modes = {row["skill_mode"] for row in rows}
    if modes != set(SKILL_MODES):
        raise ValueError(f"unexpected skill modes: {sorted(modes)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("/home/luo00466/schema_records.jsonl")
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo / "results_sweeps" / "skill_v3_qwen_gemma_matrix_20260731"
    )
    reference_cache = (
        args.reference_cache.resolve()
        if args.reference_cache
        else repo / "artifacts" / "reference_validation_cache"
    )
    sources = default_sources(repo)
    for source in sources:
        for field in ("summary_path", "state_path"):
            path = Path(source[field])
            if not path.is_file():
                raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    for source in sources:
        rows.extend(normalize_source(source))
    validate_matrix(rows)
    references = load_schema_references(args.schema.resolve())
    extend_references_from_cache(references, reference_cache)
    active_benchmarks = {row["benchmark"] for row in rows}
    references = {
        benchmark: reference
        for benchmark, reference in references.items()
        if benchmark in active_benchmarks
    }
    attach_references(rows, references)
    attach_skillless_effects(rows)
    rows.sort(
        key=lambda row: (
            row["model_key"],
            STRATEGY_ORDER.get(row["strategy"], 99),
            MODE_ORDER.get(row["skill_mode"], 99),
            row["benchmark"],
        )
    )
    winners = build_winners(rows)
    setup_summaries = build_setup_summaries(rows, winners)
    wide = build_wide(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_rows.csv", rows)
    write_jsonl(output_dir / "all_rows.jsonl", rows)
    write_csv(output_dir / "worst_cycles_wide.csv", wide)
    write_csv(output_dir / "setup_summary.csv", setup_summaries)
    write_csv(output_dir / "winners.csv", winners)
    write_readme(
        output_dir / "README.md", rows, setup_summaries, winners, references
    )

    manifest = {
        "schema_version": "c2hls.skill-v3-sweep-table-manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "toolchain": {
            "vitis_version": "2023.2",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
            "cosim": False,
        },
        "skill_catalog": {
            "id": "skill_v3_no_rmw_fc27133",
            "catalog_count": 123,
            "positive_count": 84,
            "avoid_count": 39,
        },
        "invariants": {
            "row_count": len(rows),
            "benchmark_count": len({row["benchmark"] for row in rows}),
            "setup_count": len({row["setup_id"] for row in rows}),
            "winner_count": len(winners),
            "valid_comparison_count": sum(
                row.get("comparison_valid") is True for row in rows
            ),
            "cosim_run_count": sum(row.get("cosim_status") == "passed" for row in rows),
            "reference_count": len(references),
            "website_schema_reference_count": sum(
                reference["source_kind"] == "website_schema"
                for reference in references.values()
            ),
            "cache_only_reference_count": sum(
                reference["source_kind"]
                == "reference_validation_cache_not_schema"
                for reference in references.values()
            ),
            "reference_isolation_audit_failures": sum(
                row.get("reference_isolation_audit_passed") is False for row in rows
            ),
        },
        "inputs": [
            {
                **source,
                "summary_sha256": sha256_file(Path(source["summary_path"])),
                "state_sha256": sha256_file(Path(source["state_path"])),
            }
            for source in sources
        ],
        "reference_schema": {
            "path": str(args.schema.resolve()),
            "sha256": sha256_file(args.schema.resolve()),
        },
        "outputs": {},
    }
    for name in (
        "README.md",
        "all_rows.csv",
        "all_rows.jsonl",
        "worst_cycles_wide.csv",
        "setup_summary.csv",
        "winners.csv",
    ):
        path = output_dir / name
        manifest["outputs"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), **manifest["invariants"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
