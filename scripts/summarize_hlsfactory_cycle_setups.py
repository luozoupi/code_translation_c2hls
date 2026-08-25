#!/usr/bin/env python3
"""Compare HLSFactory CSYNTH cycles across traceable experiment setups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def benchmark_name(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    if not name:
        return ""
    return name if name.startswith("hlsfactory_") else f"hlsfactory_{name}"


def geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


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
            benchmark = benchmark_name(group_path[-1] if group_path else "")
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
                    f"duplicate schema reference for {benchmark} at line {line_number}"
                )
            assignments = as_dict(synth.get("UserAssignments"))
            run = as_dict(record.get("run"))
            references[benchmark] = {
                "benchmark": benchmark,
                "cycles": cycles,
                "source_kind": "website_schema",
                "source_path": str(path.resolve()),
                "source_line": line_number,
                "device": run.get("device"),
                "vitis_version": run.get("vitis_version"),
                "clock_ns": positive_number(assignments.get("TargetClockPeriod")),
            }
    return references


def extend_references_from_cache(
    references: dict[str, dict[str, Any]], cache_dir: Path
) -> dict[str, list[float]]:
    candidates: dict[str, list[tuple[str, float, Path, dict[str, Any]]]] = defaultdict(
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
        benchmark = benchmark_name(configuration.get("benchmark"))
        cycles = positive_number(
            report.get("latency_cycles_worst") or report.get("latency_cycles")
        )
        csim = as_dict(validation.get("csim"))
        if (
            not benchmark
            or cycles is None
            or validation.get("benchmark_ready") is not True
            or csim.get("passed") is not True
        ):
            continue
        candidates[benchmark].append(
            (str(data.get("created_at") or ""), cycles, path, configuration)
        )

    ambiguities: dict[str, list[float]] = {}
    for benchmark, values in candidates.items():
        distinct = sorted({cycles for _, cycles, _, _ in values})
        if len(distinct) > 1:
            ambiguities[benchmark] = distinct
        if benchmark in references:
            continue
        created_at, cycles, path, configuration = sorted(values)[-1]
        references[benchmark] = {
            "benchmark": benchmark,
            "cycles": cycles,
            "source_kind": "reference_validation_cache_not_schema",
            "source_path": str(path.resolve()),
            "source_line": None,
            "device": configuration.get("part"),
            "vitis_version": configuration.get("vitis_version"),
            "clock_ns": positive_number(configuration.get("clock_ns")),
            "created_at": created_at,
        }
    return ambiguities


def selected_step(current: dict[str, Any], cycles: float | None) -> dict[str, Any]:
    if cycles is None:
        return {}
    steps = [step for step in as_list(current.get("step_cycles")) if isinstance(step, dict)]
    for step in steps:
        if positive_number(step.get("cycles")) == cycles:
            return step
    return {}


def csim_passed(value: Any) -> bool:
    if value is True:
        return True
    record = as_dict(value)
    return (
        record.get("passed") is True
        or record.get("success") is True
        or record.get("status") in {"pass", "passed"}
    )


def load_selected_result(current: dict[str, Any]) -> dict[str, Any]:
    value = current.get("json")
    if not isinstance(value, str) or not value:
        return {}
    path = Path(value)
    if not path.is_file():
        return {}
    try:
        data = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    final_report = as_dict(data.get("final_report"))
    final_cycles = positive_number(final_report.get("latency_cycles"))
    if final_cycles is None:
        return {}

    promotion = as_dict(data.get("best_so_far_promotion"))
    selected_name = promotion.get("from_step_name")
    selected: dict[str, Any] = {}
    if isinstance(selected_name, str) and selected_name != "baseline":
        selected = next(
            (
                step
                for step in as_list(data.get("steps"))
                if isinstance(step, dict) and step.get("step_name") == selected_name
            ),
            {},
        )
    if not selected and selected_name != "baseline":
        selected = next(
            (
                step
                for step in reversed(as_list(data.get("steps")))
                if isinstance(step, dict)
                and positive_number(as_dict(step.get("report")).get("latency_cycles"))
                == final_cycles
            ),
            {},
        )
        if selected:
            selected_name = selected.get("step_name")

    baseline_report = as_dict(data.get("baseline_report"))
    baseline_cycles = positive_number(baseline_report.get("latency_cycles"))
    if not selected and selected_name is None and final_cycles == baseline_cycles:
        selected_name = "baseline"
    selected_csim = (
        data.get("baseline_csim")
        if selected_name == "baseline"
        else selected.get("csim")
    )
    return {
        "path": str(path.resolve()),
        "data": data,
        "cycles": final_cycles,
        "phase_b_initial_cycles": baseline_cycles,
        "selected_step": selected,
        "selected_step_name": selected_name,
        "selection_score": positive_number(promotion.get("score")),
        "selected_csim_passed": csim_passed(selected_csim),
    }


def normalize_agentic_source(
    source: dict[str, Any], manifest_dir: Path
) -> list[dict[str, Any]]:
    path = resolve_path(source["path"], manifest_dir)
    data = load_json(path)
    mode_map = as_dict(source.get("mode_map"))
    rows: list[dict[str, Any]] = []
    for raw in as_list(data.get("rows")):
        if not isinstance(raw, dict):
            continue
        current = as_dict(raw.get("current"))
        raw_mode = str(raw.get("skill_mode") or "unknown")
        mode = str(source.get("mode_override") or mode_map.get(raw_mode) or raw_mode)
        selected_result = load_selected_result(current)
        if selected_result:
            result_data = as_dict(selected_result.get("data"))
            cycles = positive_number(selected_result.get("cycles"))
            phase_b_cycles = positive_number(
                selected_result.get("phase_b_initial_cycles")
            )
            step = as_dict(selected_result.get("selected_step"))
            evaluation = as_dict(result_data.get("evaluation_status"))
            run_success = result_data.get("success") is True
            selected_csim_ok = selected_result.get("selected_csim_passed") is True
            audit = as_dict(result_data.get("reference_isolation_audit"))
            cycle_source = "selected_final_report"
            selected_step_name = selected_result.get("selected_step_name")
            selection_score = selected_result.get("selection_score")
            source_result_path = selected_result.get("path")
        else:
            cycles = positive_number(as_dict(current.get("best")).get("cycles"))
            phase_b_cycles = positive_number(current.get("baseline_cycles"))
            step = selected_step(current, cycles)
            evaluation = as_dict(current.get("evaluation_status"))
            run_success = current.get("success") is True
            selected_csim_ok = (
                step.get("csim") is True
                or (
                    not step
                    and current.get("baseline_csim") is True
                    and cycles == phase_b_cycles
                )
            )
            audit = as_dict(current.get("reference_isolation_audit"))
            cycle_source = "summary_best_fallback"
            selected_step_name = as_dict(current.get("best")).get("step")
            selection_score = None
            source_result_path = current.get("json")
        if evaluation:
            valid = (
                run_success
                and evaluation.get("correctness_status") == "passed"
                and evaluation.get("synthesis_status") == "passed"
                and cycles is not None
            )
            correctness_status = evaluation.get("correctness_status")
            synthesis_status = evaluation.get("synthesis_status")
        else:
            valid = (
                run_success
                and cycles is not None
                and selected_csim_ok
            )
            correctness_status = "passed" if valid else "failed_or_unknown"
            synthesis_status = "passed" if valid else "failed_or_unknown"

        prompt = as_dict(step.get("skill_prompt"))
        injected_ids = [
            value
            for value in as_list(prompt.get("injected_skill_ids"))
            if isinstance(value, str)
        ]
        injection_count = prompt.get("injected_skill_count")
        if not isinstance(injection_count, int):
            injection_count = len(injected_ids)
        injection_known = bool(prompt) or selected_step_name == "baseline"
        benchmark = benchmark_name(raw.get("bench"))
        setup_id = f"{source['family']}__{mode}"
        rows.append(
            {
                "record_kind": "agentic",
                "setup_id": setup_id,
                "family": source["family"],
                "setup_label": f"{source['label']} / {mode}",
                "model": source.get("model"),
                "training": source.get("training"),
                "strategy": source.get("strategy"),
                "skill_mode": mode,
                "skill_version": source.get("skill_version"),
                "benchmark": benchmark,
                "cycles": cycles,
                "phase_b_initial_cycles": phase_b_cycles,
                "cycle_source": cycle_source,
                "selected_step_name": selected_step_name,
                "selection_score": selection_score,
                "valid_csim_csynth": valid,
                "correctness_status": correctness_status,
                "synthesis_status": synthesis_status,
                "skill_injection_known": injection_known,
                "skill_injected_count": injection_count,
                "skill_injected_ids": injected_ids,
                "reference_isolation_audit_passed": (
                    audit.get("passed") if audit else None
                ),
                "source_path": str(path.resolve()),
                "source_result_path": source_result_path,
                "one_shot_key": source.get("one_shot_key"),
                "allow_phase_b_one_shot_proxy": source.get(
                    "allow_phase_b_one_shot_proxy", False
                ),
                "control_mode": source.get("control_mode", "skillless"),
            }
        )
    return rows


def normalize_direct_source(
    source: dict[str, Any], manifest_dir: Path
) -> list[dict[str, Any]]:
    path = resolve_path(source["path"], manifest_dir)
    data = load_json(path)
    rows: list[dict[str, Any]] = []
    for raw in as_list(data.get("rows")):
        if not isinstance(raw, dict):
            continue
        csim = as_dict(raw.get("csim"))
        synth = as_dict(raw.get("synth"))
        cycles = positive_number(synth.get("latency_cycles"))
        valid = (
            csim.get("status") == "pass"
            and synth.get("status") == "pass"
            and cycles is not None
        )
        rows.append(
            {
                "record_kind": "one_shot",
                "setup_id": source["setup_id"],
                "family": source["setup_id"],
                "setup_label": source["label"],
                "model": source.get("model"),
                "training": source.get("training"),
                "strategy": "one_shot",
                "skill_mode": "one_shot",
                "skill_version": None,
                "benchmark": benchmark_name(raw.get("benchmark")),
                "cycles": cycles,
                "phase_b_initial_cycles": None,
                "valid_csim_csynth": valid,
                "correctness_status": csim.get("status"),
                "synthesis_status": synth.get("status"),
                "skill_injection_known": True,
                "skill_injected_count": 0,
                "skill_injected_ids": [],
                "reference_isolation_audit_passed": None,
                "source_path": str(path.resolve()),
                "source_result_path": raw.get("generated_code_path"),
                "one_shot_key": source["key"],
                "allow_phase_b_one_shot_proxy": False,
                "control_mode": None,
            }
        )
    return rows


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


def attach_references(
    rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]
) -> None:
    for row in rows:
        reference = references.get(row["benchmark"])
        row["reference_cycles"] = reference.get("cycles") if reference else None
        row["reference_source_kind"] = (
            reference.get("source_kind") if reference else None
        )
        cycles = positive_number(row.get("cycles"))
        reference_cycles = positive_number(row.get("reference_cycles"))
        row["speedup_vs_reference"] = (
            reference_cycles / cycles
            if row.get("valid_csim_csynth")
            and cycles is not None
            and reference_cycles is not None
            else None
        )


def summarize_setups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["setup_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for setup_id, setup_rows in sorted(grouped.items()):
        first = setup_rows[0]
        valid = [row for row in setup_rows if row.get("valid_csim_csynth")]
        comparable = [
            row
            for row in valid
            if positive_number(row.get("speedup_vs_reference")) is not None
        ]
        speedups = [float(row["speedup_vs_reference"]) for row in comparable]
        summaries.append(
            {
                "setup_id": setup_id,
                "setup_label": first["setup_label"],
                "model": first.get("model"),
                "training": first.get("training"),
                "strategy": first.get("strategy"),
                "skill_mode": first.get("skill_mode"),
                "completed": len(setup_rows),
                "valid_csim_csynth": len(valid),
                "reference_comparable": len(comparable),
                "geomean_speedup_vs_reference": geomean(speedups),
                "median_speedup_vs_reference": (
                    statistics.median(speedups) if speedups else None
                ),
                "beats_reference_over_1pct": sum(value > 1.01 for value in speedups),
                "ties_reference_within_1pct": sum(
                    1 / 1.01 <= value <= 1.01 for value in speedups
                ),
                "loses_reference_over_1pct": sum(
                    value < 1 / 1.01 for value in speedups
                ),
                "reference_isolation_audited": sum(
                    row.get("reference_isolation_audit_passed") is not None
                    for row in setup_rows
                ),
                "reference_isolation_audit_passes": sum(
                    row.get("reference_isolation_audit_passed") is True
                    for row in setup_rows
                ),
                "reference_isolation_audit_failures": sum(
                    row.get("reference_isolation_audit_passed") is False
                    for row in setup_rows
                ),
            }
        )
    return summaries


def build_pair_effects(
    rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    agentic = [row for row in rows if row["record_kind"] == "agentic"]
    direct_by_key_benchmark = {
        (row["one_shot_key"], row["benchmark"]): row
        for row in rows
        if row["record_kind"] == "one_shot" and row.get("valid_csim_csynth")
    }
    by_family_mode_benchmark = {
        (row["family"], row["skill_mode"], row["benchmark"]): row
        for row in agentic
        if row.get("valid_csim_csynth")
    }
    effects: list[dict[str, Any]] = []
    for skilled in agentic:
        if not skilled.get("valid_csim_csynth"):
            continue
        control_mode = skilled.get("control_mode")
        if not control_mode or skilled["skill_mode"] == control_mode:
            continue
        control = by_family_mode_benchmark.get(
            (skilled["family"], control_mode, skilled["benchmark"])
        )
        if not control:
            continue
        one_shot = direct_by_key_benchmark.get(
            (skilled.get("one_shot_key"), skilled["benchmark"])
        )
        if one_shot:
            one_shot_cycles = positive_number(one_shot.get("cycles"))
            one_shot_source = "executed_direct_one_shot"
            one_shot_setup_id = one_shot["setup_id"]
        elif skilled.get("allow_phase_b_one_shot_proxy"):
            one_shot_cycles = positive_number(control.get("phase_b_initial_cycles"))
            one_shot_source = "phase_b_initial_proxy"
            one_shot_setup_id = None
        else:
            one_shot_cycles = None
            one_shot_source = None
            one_shot_setup_id = None
        skilled_cycles = positive_number(skilled.get("cycles"))
        control_cycles = positive_number(control.get("cycles"))
        if skilled_cycles is None or control_cycles is None:
            continue
        speedup_control = control_cycles / skilled_cycles
        speedup_one_shot = (
            one_shot_cycles / skilled_cycles if one_shot_cycles is not None else None
        )
        injection_count = int(skilled.get("skill_injected_count") or 0)
        skill_exposure_confirmed = (
            skilled.get("skill_injection_known") is True and injection_count > 0
        )
        effects.append(
            {
                "family": skilled["family"],
                "model": skilled.get("model"),
                "training": skilled.get("training"),
                "strategy": skilled.get("strategy"),
                "benchmark": skilled["benchmark"],
                "skill_mode": skilled["skill_mode"],
                "skilled_cycles": skilled_cycles,
                "skillless_cycles": control_cycles,
                "one_shot_cycles": one_shot_cycles,
                "one_shot_source": one_shot_source,
                "one_shot_setup_id": one_shot_setup_id,
                "reference_cycles": skilled.get("reference_cycles"),
                "reference_source_kind": skilled.get("reference_source_kind"),
                "speedup_vs_skillless": speedup_control,
                "speedup_vs_one_shot": speedup_one_shot,
                "speedup_vs_reference": skilled.get("speedup_vs_reference"),
                "skill_exposure_confirmed": skill_exposure_confirmed,
                "skill_injected_count": injection_count,
                "selected_step_name": skilled.get("selected_step_name"),
                "cycle_source": skilled.get("cycle_source"),
                "reference_isolation_audit_passed": skilled.get(
                    "reference_isolation_audit_passed"
                ),
                "great_gain": (
                    skill_exposure_confirmed
                    and speedup_control >= threshold
                    and speedup_one_shot is not None
                    and speedup_one_shot >= threshold
                ),
                "source_path": skilled.get("source_path"),
                "control_source_path": control.get("source_path"),
            }
        )
    return effects


def summarize_pairs(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        grouped[(effect["family"], effect["skill_mode"])].append(effect)
    summaries: list[dict[str, Any]] = []
    for (family, mode), values in sorted(grouped.items()):
        vs_control = [float(value["speedup_vs_skillless"]) for value in values]
        vs_one_shot = [
            float(value["speedup_vs_one_shot"])
            for value in values
            if positive_number(value.get("speedup_vs_one_shot")) is not None
        ]
        summaries.append(
            {
                "family": family,
                "skill_mode": mode,
                "paired_with_skillless": len(vs_control),
                "paired_with_one_shot": len(vs_one_shot),
                "geomean_speedup_vs_skillless": geomean(vs_control),
                "geomean_speedup_vs_one_shot": geomean(vs_one_shot),
                "wins_vs_skillless_over_1pct": sum(value > 1.01 for value in vs_control),
                "losses_vs_skillless_over_1pct": sum(
                    value < 1 / 1.01 for value in vs_control
                ),
                "great_gain_count": sum(value.get("great_gain") is True for value in values),
            }
        )
    return summaries


def csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else value


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


def fmt_speedup(value: Any) -> str:
    number = positive_number(value)
    return f"{number:.2f}x" if number is not None else "-"


def fmt_cycles(value: Any) -> str:
    number = positive_number(value)
    return f"{number:,.0f}" if number is not None else "-"


def write_markdown(
    path: Path,
    schema_path: Path,
    references: dict[str, dict[str, Any]],
    setup_summaries: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    threshold: float,
    cache_ambiguities: dict[str, list[float]],
) -> None:
    schema_count = sum(
        reference["source_kind"] == "website_schema"
        for reference in references.values()
    )
    cache_count = len(references) - schema_count
    great = sorted(
        [effect for effect in effects if effect.get("great_gain")],
        key=lambda effect: min(
            float(effect["speedup_vs_skillless"]),
            float(effect["speedup_vs_one_shot"]),
        ),
        reverse=True,
    )
    strong_direct = [
        effect
        for effect in great
        if effect.get("one_shot_source") == "executed_direct_one_shot"
        and positive_number(effect.get("speedup_vs_reference")) is not None
        and float(effect["speedup_vs_reference"]) >= threshold
    ]
    lines = [
        "# HLSFactory Cycle Comparison",
        "",
        f"- Canonical reference: `{schema_path.resolve()}`",
        f"- Reference coverage: **{schema_count} website-schema kernels** plus **{cache_count} cache-only kernels**.",
        "- Common target: Vitis 2023.2, U280, 3.33 ns.",
        "- Valid row rule: generated CSim pass and CSynth pass with a positive worst-case latency.",
        f"- Great-gain rule: the selected implementation has explicit injected-skill metadata and at least **{threshold:.1f}x** fewer cycles than both the matched skillless control and one-shot comparator.",
        "- Agentic cycle value: the promoted `final_report`, which is the implementation emitted by the orchestrator after latency/interval feasibility scoring.",
        "",
        "## Setup Results vs Reference",
        "",
        "| setup | valid | reference matched | geomean vs reference | wins | ties | losses | isolation audit (audited/fail) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in setup_summaries:
        lines.append(
            f"| {summary['setup_label']} | "
            f"{summary['valid_csim_csynth']}/{summary['completed']} | "
            f"{summary['reference_comparable']} | "
            f"{fmt_speedup(summary['geomean_speedup_vs_reference'])} | "
            f"{summary['beats_reference_over_1pct']} | "
            f"{summary['ties_reference_within_1pct']} | "
            f"{summary['loses_reference_over_1pct']} | "
            f"{summary['reference_isolation_audited']}/"
            f"{summary['reference_isolation_audit_failures']} |"
        )

    lines.extend(
        [
            "",
            "## Paired Skill Effects",
            "",
            "| family | skill mode | paired control | paired one-shot | geomean vs skillless | geomean vs one-shot | control wins | control losses | great gains |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in pair_summaries:
        lines.append(
            f"| {summary['family']} | {summary['skill_mode']} | "
            f"{summary['paired_with_skillless']} | {summary['paired_with_one_shot']} | "
            f"{fmt_speedup(summary['geomean_speedup_vs_skillless'])} | "
            f"{fmt_speedup(summary['geomean_speedup_vs_one_shot'])} | "
            f"{summary['wins_vs_skillless_over_1pct']} | "
            f"{summary['losses_vs_skillless_over_1pct']} | "
            f"{summary['great_gain_count']} |"
        )

    lines.extend(
        [
            "",
            f"## Strongest Direct-One-Shot Gains ({len(strong_direct)})",
            "",
            "The table shows the top 15. The paired-effects CSV contains every qualifying row.",
            "",
            "| family | benchmark | skill mode | selected step | skilled cycles | skillless cycles | direct one-shot | schema/reference | vs skillless | vs one-shot | vs reference |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for effect in strong_direct[:15]:
        lines.append(
            f"| {effect['family']} | {effect['benchmark']} | "
            f"{effect['skill_mode']} | {effect.get('selected_step_name') or '-'} | "
            f"{fmt_cycles(effect['skilled_cycles'])} | "
            f"{fmt_cycles(effect['skillless_cycles'])} | "
            f"{fmt_cycles(effect['one_shot_cycles'])} | "
            f"{fmt_cycles(effect['reference_cycles'])} | "
            f"{fmt_speedup(effect['speedup_vs_skillless'])} | "
            f"{fmt_speedup(effect['speedup_vs_one_shot'])} | "
            f"{fmt_speedup(effect['speedup_vs_reference'])} |"
        )

    lines.extend(
        [
            "",
            f"## Great Gains ({len(great)})",
            "",
            "| family | benchmark | skill mode | selected step | skilled cycles | skillless cycles | one-shot cycles | schema/reference cycles | vs skillless | vs one-shot | vs reference | one-shot source |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for effect in great:
        lines.append(
            f"| {effect['family']} | {effect['benchmark']} | "
            f"{effect['skill_mode']} | {effect.get('selected_step_name') or '-'} | "
            f"{fmt_cycles(effect['skilled_cycles'])} | "
            f"{fmt_cycles(effect['skillless_cycles'])} | "
            f"{fmt_cycles(effect['one_shot_cycles'])} | "
            f"{fmt_cycles(effect['reference_cycles'])} | "
            f"{fmt_speedup(effect['speedup_vs_skillless'])} | "
            f"{fmt_speedup(effect['speedup_vs_one_shot'])} | "
            f"{fmt_speedup(effect['speedup_vs_reference'])} | "
            f"{effect['one_shot_source']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- These are CSYNTH latency estimates, not measured runtime. The July skill-v2 and SFT sweeps intentionally did not run RTL COSIM.",
            "- Skill and skillless rows are separate model samples. Large paired differences are candidates for repeated trials, not yet deterministic causal estimates.",
            "- A lower-latency historical step is not substituted for `final_report` when the controller selected another step for a better interval/feasibility score.",
            "- `phase_b_initial_proxy` is the framework's initial generated implementation, not a separately invoked direct one-shot process.",
            "- `heat-3d` and `seidel-2d` references come from locally validated reference-cache reports because they are absent from the website schema.",
            "- `doitgen` is not part of paired conclusions because its current reference validation failed.",
            "- Advisory reference-isolation audit failures remain visible in the setup table and must be resolved before publication-grade claims.",
            "- Older July 8 and historical runs have no reference-isolation audit field; `0/0` means unaudited, not passed.",
        ]
    )
    if cache_ambiguities:
        lines.append(
            f"- Cache-cycle ambiguities were observed for: {', '.join(sorted(cache_ambiguities))}."
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--great-threshold", type=float, default=2.0)
    args = parser.parse_args()

    if args.great_threshold <= 1:
        parser.error("--great-threshold must be greater than 1")
    manifest = load_json(args.manifest)
    manifest_dir = args.manifest.resolve().parent
    references = load_schema_references(args.schema)
    cache_ambiguities = extend_references_from_cache(
        references, args.reference_cache
    )

    rows: list[dict[str, Any]] = []
    for source in as_list(manifest.get("agentic_sources")):
        rows.extend(normalize_agentic_source(source, manifest_dir))
    for source in as_list(manifest.get("one_shot_sources")):
        rows.extend(normalize_direct_source(source, manifest_dir))
    attach_references(rows, references)
    setup_summaries = summarize_setups(rows)
    effects = build_pair_effects(rows, args.great_threshold)
    pair_summaries = summarize_pairs(effects)

    output = {
        "schema_version": "c2hls.hlsfactory-cycle-comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "schema_reference": str(args.schema.resolve()),
        "reference_cache": str(args.reference_cache.resolve()),
        "great_gain_threshold": args.great_threshold,
        "reference_count": len(references),
        "website_schema_reference_count": sum(
            reference["source_kind"] == "website_schema"
            for reference in references.values()
        ),
        "cache_only_reference_count": sum(
            reference["source_kind"] == "reference_validation_cache_not_schema"
            for reference in references.values()
        ),
        "reference_cache_cycle_ambiguities": cache_ambiguities,
        "references": [references[key] for key in sorted(references)],
        "setup_summaries": setup_summaries,
        "pair_summaries": pair_summaries,
        "great_gains": [effect for effect in effects if effect.get("great_gain")],
        "strong_direct_one_shot_gains": sorted(
            [
                effect
                for effect in effects
                if effect.get("great_gain")
                and effect.get("one_shot_source") == "executed_direct_one_shot"
                and positive_number(effect.get("speedup_vs_reference")) is not None
                and float(effect["speedup_vs_reference"]) >= args.great_threshold
            ],
            key=lambda effect: min(
                float(effect["speedup_vs_skillless"]),
                float(effect["speedup_vs_one_shot"]),
            ),
            reverse=True,
        ),
        "pair_effects": effects,
        "rows": rows,
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    rows_path = prefix.with_name(f"{prefix.name}.rows.csv")
    effects_path = prefix.with_name(f"{prefix.name}.paired_effects.csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(rows_path, rows)
    write_csv(effects_path, effects)
    write_markdown(
        markdown_path,
        args.schema,
        references,
        setup_summaries,
        pair_summaries,
        effects,
        args.great_threshold,
        cache_ambiguities,
    )
    for path in (json_path, rows_path, effects_path, markdown_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
