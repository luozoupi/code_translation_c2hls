#!/usr/bin/env python3
"""Forensic audit of skill exposure, prompt load, and CSYNTH outcomes.

This intentionally distinguishes prompt exposure from semantic application.
Legacy artifacts predate exact rendered-skill telemetry, so the analyzer
reconstructs the renderer's four-skill cap for non-all-positive prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODES = (
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
)
STRATEGIES = ("flash", "multistep")
LEGACY_RENDER_LIMIT = 4


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None


def _geomean(values: Iterable[float]) -> float | None:
    usable = [value for value in values if value > 0 and math.isfinite(value)]
    if not usable:
        return None
    return math.exp(sum(math.log(value) for value in usable) / len(usable))


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for ordered_index in order[index:end]:
            ranks[ordered_index] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_delta, right_delta, strict=True)
    ) / denominator


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _skill_prompts(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _as_dict(step.get("skill_prompt"))
        for step in _as_list(data.get("steps"))
        if isinstance(step, dict) and isinstance(step.get("skill_prompt"), dict)
    ]


def _actual_rendered_ids(prompt: dict[str, Any]) -> list[str]:
    explicit = prompt.get("rendered_skill_ids")
    if isinstance(explicit, list):
        return [str(value) for value in explicit if value]
    injected = [
        str(value)
        for value in _as_list(prompt.get("injected_skill_ids"))
        if value
    ]
    scope = str(prompt.get("prompt_scope") or prompt.get("source") or "")
    if scope in {"all_positive", "all-positive", "positive_all", "positive-all"}:
        return injected
    return injected[:LEGACY_RENDER_LIMIT]


def _prompt_characters(data: dict[str, Any]) -> int:
    prompts = _as_list(
        _as_dict(_as_dict(data.get("run")).get("prompt_hashes")).get("prompts")
    )
    return sum(
        int(prompt.get("characters") or 0)
        for prompt in prompts
        if isinstance(prompt, dict)
    )


def _phase_b_event(data: dict[str, Any]) -> dict[str, Any]:
    events = _as_list(
        _as_dict(data.get("synthesis_evaluations")).get("events")
    )
    return next(
        (
            event
            for event in events
            if isinstance(event, dict) and event.get("label") == "[Phase B]"
        ),
        {},
    )


def _failure_class(data: dict[str, Any]) -> str:
    failures = [
        step
        for step in _as_list(data.get("steps"))
        if isinstance(step, dict) and step.get("success") is not True
    ]
    if not failures:
        return "none"
    text = " ".join(
        str(
            step.get("attempt_error")
            or step.get("error")
            or step.get("exception_type")
            or ""
        ).lower()
        for step in failures
    )
    if "budget" in text:
        return "budget"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if any(token in text for token in ("csim", "correctness", "mismatch")):
        return "csim_or_correctness"
    if any(token in text for token in ("synthesis", "compile", "pragma")):
        return "compile_or_synthesis"
    if "no code" in text:
        return "malformed_output"
    return "no_feasible_or_noop"


def _semantic_families(code: str) -> list[str]:
    lowered = code.lower()
    families: set[str] = set()
    if "max_widen_bitwidth" in lowered or re.search(
        r"ap_u?int\s*<\s*512", lowered
    ):
        families.add("coalescing")
    if "#pragma hls dataflow" in lowered or (
        "ping" in lowered and "pong" in lowered
    ):
        families.add("dataflow")
    if "#pragma hls pipeline" in lowered:
        families.add("pipeline")
    if re.search(r"\btile\w*\b", lowered):
        families.add("tiling")
    if "#pragma hls unroll" in lowered:
        families.add("unroll")
    if "#pragma hls array_partition" in lowered:
        families.add("partition")
    if "#pragma hls dependence" in lowered:
        families.add("dependence")
    return sorted(families)


def _load_skill_tiers(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("skills", []) if isinstance(payload, dict) else payload
    return {
        str(skill.get("id")): str(skill.get("confidence") or "")
        for skill in raw
        if isinstance(skill, dict) and skill.get("id")
    }


def load_rows(matrix_csv: Path, skill_library: Path) -> list[dict[str, Any]]:
    tiers = _load_skill_tiers(skill_library)
    rows: list[dict[str, Any]] = []
    with matrix_csv.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            result_path = Path(source["source_result_path"])
            data = json.loads(result_path.read_text(encoding="utf-8"))
            prompts = _skill_prompts(data)
            rendered_ids = [
                skill_id
                for prompt in prompts
                for skill_id in _actual_rendered_ids(prompt)
            ]
            recorded_ids = [
                str(skill_id)
                for prompt in prompts
                for skill_id in _as_list(prompt.get("injected_skill_ids"))
                if skill_id
            ]
            positive_rendered = [
                skill_id
                for skill_id in rendered_ids
                if tiers.get(skill_id) != "avoid"
            ]
            avoid_rendered = [
                skill_id
                for skill_id in rendered_ids
                if tiers.get(skill_id) == "avoid"
            ]
            usage = _as_dict(data.get("llm_usage"))
            phase_b = _phase_b_event(data)
            final_cycles = _positive(source.get("latency_cycles"))
            baseline_cycles = _positive(source.get("phase_b_initial_cycles"))
            code = str(data.get("hls_code") or "")
            recorded_count = len(recorded_ids)
            rendered_count = len(rendered_ids)
            rows.append(
                {
                    "benchmark": source.get("benchmark"),
                    "problem": source.get("problem"),
                    "strategy": source.get("strategy"),
                    "skill_mode": source.get("skill_mode"),
                    "valid_csim_csynth": (
                        source.get("valid_csim_csynth") == "True"
                    ),
                    "final_cycles": final_cycles,
                    "phase_b_cycles": baseline_cycles,
                    "speedup_vs_phase_b": (
                        baseline_cycles / final_cycles
                        if baseline_cycles and final_cycles
                        else None
                    ),
                    "skill_prompt_events": len(prompts),
                    "recorded_skill_exposures": recorded_count,
                    "rendered_skill_exposures": rendered_count,
                    "rendered_positive_exposures": len(positive_rendered),
                    "rendered_avoid_exposures": len(avoid_rendered),
                    "recorded_unique_skill_count": len(set(recorded_ids)),
                    "rendered_unique_skill_count": len(set(rendered_ids)),
                    "rendered_unique_positive_count": len(
                        set(positive_rendered)
                    ),
                    "legacy_recorded_overcount": max(
                        recorded_count - rendered_count, 0
                    ),
                    "rendered_skill_ids": sorted(set(rendered_ids)),
                    "input_tokens": int(usage.get("input_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                    "llm_calls": int(usage.get("calls") or 0),
                    "prompt_characters": _prompt_characters(data),
                    "phase_b_code_sha256": phase_b.get("code_sha256"),
                    "phase_b_report_sha256": phase_b.get("report_sha256"),
                    "selected_code_sha256": data.get("selected_code_sha256"),
                    "failure_class": _failure_class(data),
                    "final_equals_phase_b": (
                        final_cycles == baseline_cycles
                        if final_cycles and baseline_cycles
                        else None
                    ),
                    "semantic_families": _semantic_families(code),
                    "semantic_family_count": len(_semantic_families(code)),
                    "pragma_count": len(
                        re.findall(
                            r"^\s*#\s*pragma\s+hls\b",
                            code,
                            flags=re.IGNORECASE | re.MULTILINE,
                        )
                    ),
                    "result_path": str(result_path),
                }
            )
    return rows


def add_paired_speedups(rows: list[dict[str, Any]]) -> None:
    controls = {
        (row["benchmark"], row["strategy"]): row
        for row in rows
        if row["skill_mode"] == "skillless"
        and row["valid_csim_csynth"]
        and row["final_cycles"]
    }
    for row in rows:
        control = controls.get((row["benchmark"], row["strategy"]))
        row["speedup_vs_skillless"] = (
            control["final_cycles"] / row["final_cycles"]
            if control
            and row["valid_csim_csynth"]
            and row["final_cycles"]
            else None
        )
        row["same_phase_b_code_as_skillless"] = (
            bool(control)
            and bool(row.get("phase_b_code_sha256"))
            and row.get("phase_b_code_sha256")
            == control.get("phase_b_code_sha256")
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_setup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setup[(str(row["strategy"]), str(row["skill_mode"]))].append(row)

    setups: dict[str, Any] = {}
    for strategy in STRATEGIES:
        for mode in MODES:
            group = by_setup[(strategy, mode)]
            valid = [row for row in group if row["valid_csim_csynth"]]
            speedups = [
                float(row["speedup_vs_skillless"])
                for row in valid
                if row.get("speedup_vs_skillless")
            ]
            setups[f"{strategy}/{mode}"] = {
                "rows": len(group),
                "valid": len(valid),
                "geomean_speedup_vs_skillless": _geomean(speedups),
                "median_speedup_vs_skillless": (
                    statistics.median(speedups) if speedups else None
                ),
                "wins_over_1pct": sum(value > 1.01 for value in speedups),
                "ties_within_1pct": sum(
                    1.0 / 1.01 <= value <= 1.01 for value in speedups
                ),
                "losses_over_1pct": sum(
                    value < 1.0 / 1.01 for value in speedups
                ),
                "median_rendered_positive_exposures": (
                    statistics.median(
                        row["rendered_positive_exposures"] for row in valid
                    )
                    if valid
                    else None
                ),
                "median_input_tokens": (
                    statistics.median(row["input_tokens"] for row in valid)
                    if valid
                    else None
                ),
                "median_prompt_characters": (
                    statistics.median(
                        row["prompt_characters"] for row in valid
                    )
                    if valid
                    else None
                ),
                "phase_b_fallbacks": sum(
                    row.get("final_equals_phase_b") is True for row in valid
                ),
                "failure_classes": dict(
                    Counter(row["failure_class"] for row in valid)
                ),
                "legacy_recorded_overcount_total": sum(
                    row["legacy_recorded_overcount"] for row in valid
                ),
                "median_semantic_family_count": (
                    statistics.median(
                        row["semantic_family_count"] for row in valid
                    )
                    if valid
                    else None
                ),
            }

    correlations: dict[str, Any] = {}
    for strategy in STRATEGIES:
        valid = [
            row
            for row in rows
            if row["strategy"] == strategy
            and row["valid_csim_csynth"]
            and row.get("speedup_vs_skillless")
        ]
        count_values = [
            float(row["rendered_positive_exposures"]) for row in valid
        ]
        log_speedups = [
            math.log(float(row["speedup_vs_skillless"])) for row in valid
        ]
        per_benchmark: list[float] = []
        for benchmark in sorted({str(row["benchmark"]) for row in valid}):
            group = [row for row in valid if row["benchmark"] == benchmark]
            coefficient = _spearman(
                [
                    float(row["rendered_positive_exposures"])
                    for row in group
                ],
                [
                    math.log(float(row["speedup_vs_skillless"]))
                    for row in group
                ],
            )
            if coefficient is not None:
                per_benchmark.append(coefficient)
        correlations[strategy] = {
            "global_pearson_count_vs_log_speedup": _pearson(
                count_values, log_speedups
            ),
            "global_spearman_count_vs_log_speedup": _spearman(
                count_values, log_speedups
            ),
            "within_benchmark_spearman_median": (
                statistics.median(per_benchmark)
                if per_benchmark
                else None
            ),
            "within_benchmark_positive": sum(
                value > 0 for value in per_benchmark
            ),
            "within_benchmark_negative": sum(
                value < 0 for value in per_benchmark
            ),
        }

    baseline_consistency: dict[str, Any] = {}
    for strategy in STRATEGIES:
        strategy_rows = [
            row
            for row in rows
            if row["strategy"] == strategy and row["valid_csim_csynth"]
        ]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in strategy_rows:
            groups[str(row["benchmark"])].append(row)
        baseline_consistency[strategy] = {
            "benchmarks": len(groups),
            "exact_phase_b_code_shared_by_all_modes": sum(
                len(
                    {
                        row.get("phase_b_code_sha256")
                        for row in group
                        if row.get("phase_b_code_sha256")
                    }
                )
                == 1
                for group in groups.values()
            ),
        }

    return {
        "schema_version": "c2hls.skill-dose-forensic.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "valid_row_count": sum(row["valid_csim_csynth"] for row in rows),
        "benchmark_count": len({row["benchmark"] for row in rows}),
        "valid_benchmark_count": len(
            {
                row["benchmark"]
                for row in rows
                if row["valid_csim_csynth"]
            }
        ),
        "setup_summaries": setups,
        "correlations": correlations,
        "baseline_consistency": baseline_consistency,
        "interpretation": {
            "injected_means_prompt_exposure": True,
            "semantic_application_proven": False,
            "legacy_matched_metadata_can_overcount_rendered_skills": True,
            "anthropic_seed_supported": False,
            "causal_confirmation_requires_frozen_phase_b": True,
        },
    }


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            "|".join(value)
            if isinstance(value, list)
            else value
        )
        for key, value in row.items()
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_jsonable_row(row) for row in rows)


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    valid = frame[
        frame["valid_csim_csynth"]
        & frame["speedup_vs_skillless"].notna()
    ].copy()
    valid["log2_speedup"] = valid["speedup_vs_skillless"].map(math.log2)
    colors = {
        "skillless": "#555555",
        "matched": "#0072B2",
        "smart_best_fit": "#009E73",
        "smart_exhaustive": "#E69F00",
        "all_positive": "#CC79A7",
    }
    outputs: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, strategy in zip(axes, STRATEGIES, strict=True):
        subset = valid[valid["strategy"] == strategy]
        for mode in MODES:
            group = subset[subset["skill_mode"] == mode]
            axis.scatter(
                group["rendered_positive_exposures"],
                group["log2_speedup"],
                s=26,
                alpha=0.62,
                label=mode,
                color=colors[mode],
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_title(strategy.title())
        axis.set_xlabel("Rendered positive skill exposures")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("log2 speedup versus skillless")
    axes[1].legend(fontsize=8, loc="best")
    figure.tight_layout()
    path = output_dir / "skill_count_vs_speedup.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    setup = (
        valid.groupby(["strategy", "skill_mode"], observed=True)
        .agg(
            median_input_tokens=("input_tokens", "median"),
            validity=("valid_csim_csynth", "mean"),
        )
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    labels = [
        f"{row.strategy}\n{row.skill_mode}"
        for row in setup.itertuples()
    ]
    axis.bar(
        range(len(setup)),
        setup["median_input_tokens"],
        color=[colors[value] for value in setup["skill_mode"]],
    )
    axis.set_xticks(range(len(setup)), labels, rotation=35, ha="right")
    axis.set_ylabel("Median input tokens")
    axis.set_title("Prompt cost by setup")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "skill_setup_prompt_tokens.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    return outputs


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return "-"


def write_markdown(
    path: Path,
    matrix_csv: Path,
    summary: dict[str, Any],
    plot_paths: list[Path],
) -> None:
    lines = [
        "# Skill-Dose Forensic Audit",
        "",
        f"- Source matrix: `{matrix_csv}`",
        f"- Valid rows: **{summary['valid_row_count']}/{summary['row_count']}**",
        f"- Benchmarks: **{summary['benchmark_count']} total; "
        f"{summary['valid_benchmark_count']} with valid CSim/CSynth**",
        "- Metric: generated Vitis 2023.2 CSYNTH latency cycles.",
        "- `rendered` means present in the model-visible prompt; it does not prove application.",
        "",
        "## Paired Effects",
        "",
        "| setup | valid | rendered positive exposures (median) | geomean speedup vs skillless | wins | ties | losses | Phase-B fallbacks | input tokens (median) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        for mode in MODES:
            item = summary["setup_summaries"][f"{strategy}/{mode}"]
            lines.append(
                f"| {strategy}/{mode} | {item['valid']} | "
                f"{_fmt(item['median_rendered_positive_exposures'], 1)} | "
                f"{_fmt(item['geomean_speedup_vs_skillless'])} | "
                f"{item['wins_over_1pct']} | {item['ties_within_1pct']} | "
                f"{item['losses_over_1pct']} | {item['phase_b_fallbacks']} | "
                f"{_fmt(item['median_input_tokens'], 0)} |"
            )

    lines.extend(
        [
            "",
            "## Causal Defects Found",
            "",
            "1. The all-positive prompt exposes 42 recipes as a menu but still produces only one candidate per optimization step.",
            "2. Legacy matched prompts record all requested IDs even though the renderer displays at most four.",
            "3. Matched counts include avoid-tier rules, so they are not positive-skill cardinalities.",
            "4. Action-only rendering removes applicability and safety constraints from every positive skill.",
            "5. Smart exhaustive scores the catalog but injects at most three skills; it does not synthesize every skill separately.",
            "6. Exact Phase-B code is shared across all five modes on only "
            f"{summary['baseline_consistency']['flash']['exact_phase_b_code_shared_by_all_modes']}/"
            f"{summary['baseline_consistency']['flash']['benchmarks']} flash benchmarks and "
            f"{summary['baseline_consistency']['multistep']['exact_phase_b_code_shared_by_all_modes']}/"
            f"{summary['baseline_consistency']['multistep']['benchmarks']} multistep benchmarks.",
            "7. Current artifacts do not record declared or semantically verified skill application.",
            "",
            "## Conclusion",
            "",
            "The existing sweep rejects a monotonic prompt-dose interpretation. More skill text is not more applied or synthesized transformations. A causal follow-up must freeze the exact Phase-B kernel, use nested rendered sets, repeat stochastic model calls, and track application separately from exposure.",
            "",
            "## Figures",
            "",
        ]
    )
    for plot in plot_paths:
        lines.append(f"- `{plot}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--skill-library", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.matrix_csv, args.skill_library)
    add_paired_speedups(rows)
    result = summarize(rows)
    result["source_matrix_csv"] = str(args.matrix_csv.resolve())
    result["source_matrix_sha256"] = hashlib.sha256(
        args.matrix_csv.read_bytes()
    ).hexdigest()
    result["rows"] = rows

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    plot_dir = args.plot_dir or args.output_prefix.parent / (
        args.output_prefix.name + "_figures"
    )
    plot_paths = write_plots(plot_dir, rows)
    result["plots"] = [str(path) for path in plot_paths]

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    write_rows_csv(csv_path, rows)
    write_markdown(md_path, args.matrix_csv.resolve(), result, plot_paths)
    print(json_path)
    print(csv_path)
    print(md_path)
    for path in plot_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
