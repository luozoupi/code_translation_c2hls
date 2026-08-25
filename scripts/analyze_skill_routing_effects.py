#!/usr/bin/env python3
"""Audit skill routing and paired CSYNTH effects from agentic sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODES = (
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _infer_mode(path: Path, data: dict[str, Any]) -> str:
    run_mode = _as_dict(data.get("run")).get("skill_mode")
    if isinstance(run_mode, str) and run_mode:
        return run_mode
    parent = path.parent.name
    for mode in sorted(DEFAULT_MODES, key=len, reverse=True):
        if parent.endswith(f"_{mode}"):
            return mode
    return "unknown"


def _collect_skill_prompts(value: Any) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "skill_prompt" and isinstance(child, dict):
                    prompts.append(child)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for prompt in prompts:
        encoded = json.dumps(prompt, sort_keys=True, separators=(",", ":"))
        if encoded not in seen:
            unique.append(prompt)
            seen.add(encoded)
    return unique


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if isinstance(value, str) and value})


def _step_metric_changed(step: dict[str, Any]) -> bool:
    comparison = _as_dict(step.get("vs_previous"))
    for metric in ("latency_cycles", "interval", "bram", "dsp", "ff", "lut"):
        values = _as_dict(comparison.get(metric))
        generated = _number(values.get("generated"))
        previous = _number(values.get("ground_truth"))
        if generated is not None and previous is not None and generated != previous:
            return True
    return False


def _route_evidence(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    audit_events = 0
    no_match_events = 0
    selected_count = 0
    eligible_count = 0
    hard_evidence_count = 0
    specialty_compatible_count = 0
    bottlenecks: list[str] = []
    task_features: list[str] = []
    selected_ids: list[str] = []

    for prompt in prompts:
        audit = _as_dict(prompt.get("router_audit"))
        if not audit:
            continue
        audit_events += 1
        no_match_events += int(bool(audit.get("no_match")))
        current_selected = _dedupe_strings(audit.get("selected_skill_ids") or [])
        selected_ids.extend(current_selected)
        selected_count += len(current_selected)
        bottlenecks.extend(_dedupe_strings(audit.get("bottleneck_kinds") or []))
        task_features.extend(_dedupe_strings(audit.get("task_features") or []))
        scores = {
            score.get("skill_id"): score
            for score in _as_list(audit.get("scores"))
            if isinstance(score, dict) and isinstance(score.get("skill_id"), str)
        }
        for skill_id in current_selected:
            score = _as_dict(scores.get(skill_id))
            eligible_count += int(score.get("eligible") is True)
            hard_evidence_count += int(score.get("hard_evidence") is True)
            specialty_compatible_count += int(
                score.get("specialty_compatible") is True
            )

    return {
        "route_audit_events": audit_events,
        "route_no_match_events": no_match_events,
        "route_selected_count": selected_count,
        "route_eligible_selected_count": eligible_count,
        "route_hard_evidence_selected_count": hard_evidence_count,
        "route_specialty_compatible_selected_count": specialty_compatible_count,
        "route_selected_skill_ids": _dedupe_strings(selected_ids),
        "route_bottleneck_kinds": _dedupe_strings(bottlenecks),
        "route_task_features": _dedupe_strings(task_features),
    }


def _summarize_result(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "result_path": str(path),
            "benchmark": path.name.removesuffix("_multistep_results.json"),
            "mode": "unknown",
            "run_success": False,
            "valid_csim_csynth": False,
            "artifact_error": str(exc),
        }

    benchmark = data.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        benchmark = path.name.removesuffix("_multistep_results.json")
    mode = _infer_mode(path, data)
    run = _as_dict(data.get("run"))
    evaluation = _as_dict(data.get("evaluation_status"))
    baseline_report = _as_dict(data.get("baseline_report"))
    final_report = _as_dict(data.get("final_report"))
    steps = [step for step in _as_list(data.get("steps")) if isinstance(step, dict)]
    prompts = _collect_skill_prompts(steps)
    provenance = _as_dict(run.get("skill_library_provenance"))
    store = _as_dict(provenance.get("store"))

    baseline_cycles = _positive_number(baseline_report.get("latency_cycles"))
    final_cycles = _positive_number(final_report.get("latency_cycles"))
    run_success = data.get("success") is True
    correctness = evaluation.get("correctness_status")
    synthesis = evaluation.get("synthesis_status")
    valid = (
        run_success
        and correctness == "passed"
        and synthesis == "passed"
        and final_cycles is not None
    )

    synthesis_events = _as_list(
        _as_dict(data.get("synthesis_evaluations")).get("events")
    )
    optimization_events = [
        event
        for event in synthesis_events
        if isinstance(event, dict)
        and str(event.get("label") or "").startswith("[Step:")
    ]
    successful_steps = [step for step in steps if step.get("success") is True]
    prompted_successful_steps = [
        step
        for step in successful_steps
        if _as_dict(step.get("skill_prompt")).get("injected") is True
    ]
    changed_successful_steps = [
        step for step in successful_steps if _step_metric_changed(step)
    ]

    injected_ids = _dedupe_strings(
        skill_id
        for prompt in prompts
        for skill_id in _as_list(prompt.get("injected_skill_ids"))
    )
    matched_ids = _dedupe_strings(
        skill_id
        for prompt in prompts
        for skill_id in _as_list(prompt.get("matched_skill_ids"))
    )
    avoid_ids = _dedupe_strings(
        skill_id
        for prompt in prompts
        for skill_id in _as_list(prompt.get("avoid_skill_ids"))
    )
    prompt_modes = _dedupe_strings(prompt.get("prompt_mode") for prompt in prompts)
    prompt_scopes = _dedupe_strings(
        prompt.get("prompt_scope") for prompt in prompts
    )
    action_only_events = sum(
        int(prompt.get("prompt_mode") == "action_only") for prompt in prompts
    )
    avoid_suppressed_events = sum(
        int(prompt.get("avoid_skills_suppressed") is True) for prompt in prompts
    )
    injection_counts = [
        int(prompt.get("injected_skill_count"))
        if isinstance(prompt.get("injected_skill_count"), int)
        else len(_as_list(prompt.get("injected_skill_ids")))
        for prompt in prompts
    ]
    skillless_prompt_integrity = all(
        not _as_list(prompt.get("injected_skill_ids"))
        and not _as_list(prompt.get("matched_skill_ids"))
        and prompt.get("source") == "skillless"
        and prompt.get("reason") == "explicit_empty_skill_list"
        for prompt in prompts
    )
    loaded_skill_count = provenance.get("loaded_skill_count")
    if not isinstance(loaded_skill_count, int):
        loaded_skill_count = store.get("skill_count")

    if not valid and not optimization_events:
        application_state = "run_failed_before_optimization"
        tracking_complete = True
    elif mode == "skillless":
        application_state = "skillless_control"
        tracking_complete = (
            loaded_skill_count == 0
            and not injected_ids
            and (not prompts or skillless_prompt_integrity)
        )
    elif prompted_successful_steps:
        application_state = "skill_prompted_step_accepted"
        tracking_complete = True
    elif prompts:
        application_state = "skill_prompted_step_rejected"
        tracking_complete = True
    elif optimization_events:
        application_state = "rejected_step_prompt_untracked"
        tracking_complete = False
    else:
        application_state = "no_skill_application_observed"
        tracking_complete = False

    route = _route_evidence(prompts)
    speedup_vs_baseline = (
        baseline_cycles / final_cycles
        if baseline_cycles is not None and final_cycles is not None
        else None
    )
    return {
        "result_path": str(path),
        "benchmark": benchmark,
        "mode": mode,
        "run_success": run_success,
        "valid_csim_csynth": valid,
        "correctness_status": correctness,
        "synthesis_status": synthesis,
        "cosim_status": evaluation.get("cosim_execution_status"),
        "baseline_cycles": baseline_cycles,
        "final_cycles": final_cycles,
        "speedup_vs_phase_b_baseline": speedup_vs_baseline,
        "accepted_step_count": len(successful_steps),
        "metric_changed_accepted_step_count": len(changed_successful_steps),
        "optimization_evaluation_count": len(optimization_events),
        "skill_prompt_event_count": len(prompts),
        "skill_prompt_tracking_complete": tracking_complete,
        "skill_application_state": application_state,
        "loaded_skill_count": loaded_skill_count,
        "library_sha256": store.get("sha256"),
        "injection_counts": injection_counts,
        "injected_skill_ids": injected_ids,
        "matched_skill_ids": matched_ids,
        "avoid_skill_ids": avoid_ids,
        "prompt_modes": prompt_modes,
        "prompt_scopes": prompt_scopes,
        "action_only_prompt_event_count": action_only_events,
        "avoid_suppressed_prompt_event_count": avoid_suppressed_events,
        "skillless_prompt_integrity": (
            skillless_prompt_integrity if mode == "skillless" else None
        ),
        "run_fingerprint_sha256": _as_dict(data.get("run_fingerprint")).get(
            "sha256"
        ),
        "artifact_error": None,
        **route,
    }


def _geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("valid_csim_csynth")]
    injection_counts = [
        count
        for row in rows
        for count in row.get("injection_counts") or []
        if isinstance(count, int)
    ]
    selected = sum(int(row.get("route_selected_count") or 0) for row in rows)
    eligible = sum(
        int(row.get("route_eligible_selected_count") or 0) for row in rows
    )
    hard = sum(
        int(row.get("route_hard_evidence_selected_count") or 0) for row in rows
    )
    compatible = sum(
        int(row.get("route_specialty_compatible_selected_count") or 0)
        for row in rows
    )
    prompt_modes = Counter(
        prompt_mode
        for row in rows
        for prompt_mode in row.get("prompt_modes") or []
    )
    prompt_events = sum(
        int(row.get("skill_prompt_event_count") or 0) for row in rows
    )
    return {
        "completed": len(rows),
        "valid_csim_csynth": len(valid_rows),
        "invalid_or_failed": len(rows) - len(valid_rows),
        "runs_with_accepted_optimization": sum(
            int(int(row.get("accepted_step_count") or 0) > 0) for row in rows
        ),
        "runs_improved_over_phase_b_baseline": sum(
            int(
                isinstance(row.get("speedup_vs_phase_b_baseline"), (int, float))
                and row["speedup_vs_phase_b_baseline"] > 1.0
            )
            for row in valid_rows
        ),
        "skill_prompt_tracking_complete": sum(
            int(row.get("skill_prompt_tracking_complete") is True) for row in rows
        ),
        "skill_prompt_tracking_gaps": sum(
            int(row.get("skill_prompt_tracking_complete") is False) for row in rows
        ),
        "prompt_event_count": prompt_events,
        "prompt_mode_distribution": dict(sorted(prompt_modes.items())),
        "action_only_prompt_event_count": sum(
            int(row.get("action_only_prompt_event_count") or 0) for row in rows
        ),
        "avoid_suppressed_prompt_event_count": sum(
            int(row.get("avoid_suppressed_prompt_event_count") or 0) for row in rows
        ),
        "unique_avoid_skill_count": len(
            {
                skill_id
                for row in rows
                for skill_id in row.get("avoid_skill_ids") or []
            }
        ),
        "injection_count_distribution": {
            str(key): value
            for key, value in sorted(Counter(injection_counts).items())
        },
        "unique_injected_skill_count": len(
            {
                skill_id
                for row in rows
                for skill_id in row.get("injected_skill_ids") or []
            }
        ),
        "route_audit_events": sum(
            int(row.get("route_audit_events") or 0) for row in rows
        ),
        "route_no_match_events": sum(
            int(row.get("route_no_match_events") or 0) for row in rows
        ),
        "route_selected_count": selected,
        "route_selected_eligible_fraction": (
            eligible / selected if selected else None
        ),
        "route_selected_hard_evidence_fraction": hard / selected if selected else None,
        "route_selected_specialty_compatible_fraction": (
            compatible / selected if selected else None
        ),
        "skillless_integrity_passes": sum(
            int(
                row.get("mode") == "skillless"
                and row.get("skill_prompt_tracking_complete") is True
            )
            for row in rows
        ),
    }


def _paired_effects(
    rows: list[dict[str, Any]], modes: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (row.get("benchmark"), row.get("mode")): row
        for row in rows
        if row.get("valid_csim_csynth")
        and _positive_number(row.get("final_cycles")) is not None
    }
    baselines = {
        benchmark: row
        for (benchmark, mode), row in by_key.items()
        if mode == "skillless"
    }
    pairs: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for mode in modes:
        if mode == "skillless":
            continue
        mode_pairs: list[dict[str, Any]] = []
        for benchmark, baseline in sorted(baselines.items()):
            treatment = by_key.get((benchmark, mode))
            if not treatment:
                continue
            baseline_cycles = _positive_number(baseline.get("final_cycles"))
            treatment_cycles = _positive_number(treatment.get("final_cycles"))
            if baseline_cycles is None or treatment_cycles is None:
                continue
            speedup = baseline_cycles / treatment_cycles
            pair = {
                "benchmark": benchmark,
                "mode": mode,
                "skillless_cycles": baseline_cycles,
                "mode_cycles": treatment_cycles,
                "speedup_vs_skillless": speedup,
                "outcome": (
                    "win"
                    if speedup > 1.01
                    else "loss"
                    if speedup < (1.0 / 1.01)
                    else "tie_within_1pct"
                ),
                "application_state": treatment.get("skill_application_state"),
                "injected_skill_ids": treatment.get("injected_skill_ids") or [],
            }
            pairs.append(pair)
            mode_pairs.append(pair)
        speedups = [pair["speedup_vs_skillless"] for pair in mode_pairs]
        outcomes = Counter(pair["outcome"] for pair in mode_pairs)
        summaries[mode] = {
            "paired_benchmarks": len(mode_pairs),
            "geomean_speedup_vs_skillless": _geomean(speedups),
            "median_speedup_vs_skillless": (
                statistics.median(speedups) if speedups else None
            ),
            "wins_over_1pct": outcomes["win"],
            "ties_within_1pct": outcomes["tie_within_1pct"],
            "losses_over_1pct": outcomes["loss"],
            "best": (
                max(mode_pairs, key=lambda pair: pair["speedup_vs_skillless"])
                if mode_pairs
                else None
            ),
            "worst": (
                min(mode_pairs, key=lambda pair: pair["speedup_vs_skillless"])
                if mode_pairs
                else None
            ),
        }
    return pairs, summaries


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _format_float(value: Any) -> str:
    number = _number(value)
    return f"{number:.3f}" if number is not None else "-"


def _write_markdown(
    path: Path,
    source: Path,
    expected_per_mode: int,
    modes: tuple[str, ...],
    mode_summaries: dict[str, Any],
    pair_summaries: dict[str, Any],
    total_completed: int,
) -> None:
    expected_total = expected_per_mode * len(modes)
    lines = [
        "# Skill Routing and CSYNTH Effect Audit",
        "",
        f"- Source: `{source}`",
        f"- Completed result files: **{total_completed}/{expected_total}**",
        "- Validation: generated CSim and Vitis 2023.2 CSynth; COSIM is intentionally not run in this sweep.",
        "- Effect estimate: final selected cycles for each skill mode divided against the independently sampled `skillless` run for the same benchmark.",
        "",
        "## Progress and Traceability",
        "",
        "| mode | complete | valid | failed | accepted optimization | tracking gaps | prompt modes | avoid IDs | injection counts |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for mode in modes:
        summary = mode_summaries.get(mode) or {}
        lines.append(
            f"| {mode} | {summary.get('completed', 0)}/{expected_per_mode} | "
            f"{summary.get('valid_csim_csynth', 0)} | "
            f"{summary.get('invalid_or_failed', 0)} | "
            f"{summary.get('runs_with_accepted_optimization', 0)} | "
            f"{summary.get('skill_prompt_tracking_gaps', 0)} | "
            f"{json.dumps(summary.get('prompt_mode_distribution') or {}, sort_keys=True)} | "
            f"{summary.get('unique_avoid_skill_count', 0)} | "
            f"{json.dumps(summary.get('injection_count_distribution') or {}, sort_keys=True)} |"
        )

    lines.extend(
        [
            "",
            "## Paired CSYNTH Effect",
            "",
            "| mode vs skillless | paired | geomean speedup | median speedup | wins | ties | losses |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in modes:
        if mode == "skillless":
            continue
        summary = pair_summaries.get(mode) or {}
        lines.append(
            f"| {mode} | {summary.get('paired_benchmarks', 0)} | "
            f"{_format_float(summary.get('geomean_speedup_vs_skillless'))} | "
            f"{_format_float(summary.get('median_speedup_vs_skillless'))} | "
            f"{summary.get('wins_over_1pct', 0)} | "
            f"{summary.get('ties_within_1pct', 0)} | "
            f"{summary.get('losses_over_1pct', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Router Evidence",
            "",
            "| mode | audited selections | eligible | hard-evidence | specialty-compatible | no-match events |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in modes:
        summary = mode_summaries.get(mode) or {}
        lines.append(
            f"| {mode} | {summary.get('route_selected_count', 0)} | "
            f"{_format_float(summary.get('route_selected_eligible_fraction'))} | "
            f"{_format_float(summary.get('route_selected_hard_evidence_fraction'))} | "
            f"{_format_float(summary.get('route_selected_specialty_compatible_fraction'))} | "
            f"{summary.get('route_no_match_events', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- A routed/injected skill is prompt exposure, not proof that the model followed every action.",
            "- `skill_prompted_step_accepted` plus a metric change is application evidence, but causal attribution still requires repeated paired trials because Anthropic does not honor a fixed seed here.",
            "- A baseline fallback is an intent-to-treat outcome. It is not counted as an accepted skill application.",
            "- Rejected attempts whose prompt metadata is absent are labeled as tracking gaps rather than zero-effect skill applications.",
            "- CSYNTH latency is a performance proxy. Runtime conclusions require selected RTL COSIM or hardware execution.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-per-mode", type=int, default=28)
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated ordered skill modes.",
    )
    args = parser.parse_args()

    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    paths = sorted(args.results_root.glob("*/*_multistep_results.json"))
    rows = [_summarize_result(path) for path in paths]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mode") or "unknown")].append(row)
    mode_summaries = {
        mode: _mode_summary(grouped.get(mode, [])) for mode in modes
    }
    pairs, pair_summaries = _paired_effects(rows, modes)
    expected_total = args.expected_per_mode * len(modes)

    output = {
        "schema_version": "c2hls.skill-routing-effects.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_results_root": str(args.results_root.resolve()),
        "expected_per_mode": args.expected_per_mode,
        "expected_total": expected_total,
        "completed_total": len(rows),
        "remaining_total": max(expected_total - len(rows), 0),
        "modes": list(modes),
        "mode_summaries": mode_summaries,
        "paired_effect_summaries": pair_summaries,
        "rows": rows,
        "paired_effects": pairs,
        "interpretation": {
            "performance_metric": "vitis_csynth_latency_cycles",
            "control_mode": "skillless",
            "cosim_required_for_runtime_claim": True,
            "prompt_exposure_is_not_semantic_application_proof": True,
            "anthropic_seed_supported": False,
        },
    }

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        args.results_root.resolve(),
        args.expected_per_mode,
        modes,
        mode_summaries,
        pair_summaries,
        len(rows),
    )
    print(json_path)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
