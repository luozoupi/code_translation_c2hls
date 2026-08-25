#!/usr/bin/env python3
"""Export Sonnet 4.6 GEMM multistep trajectories and QoR evidence.

The July 24 campaign uses a setup-specific optimization order.  This script
therefore treats the x axis as an ordinal and preserves each setup's labels.
It reports both each synthesized candidate and the best feasible candidate
seen by that point.  The latter matches the controller's final promotion rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAMPAIGN = Path(
    "results_sweeps/"
    "agentic_no_streamcluster_hpca_skillv2_hlsfactory_sonnet46_"
    "multistep_skills5_csim_csynth_20260724"
)
DEFAULT_OUTPUT = Path(
    "results_sweeps/sonnet46_gemm_multistep_trajectory_20260819"
)
SETUP_ORDER = [
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
]
SETUP_LABELS = {
    "skillless": "Skillless",
    "matched": "Matched",
    "smart_best_fit": "Smart best-fit",
    "smart_exhaustive": "Smart exhaustive",
    "all_positive": "All positive",
}
STEP_LABELS = {
    "baseline": "base",
    "coalescing": "coal",
    "doublebuffer": "dblbuf",
    "pipeline": "pipe",
    "tiling": "tile",
    "unroll": "unroll",
}
REPORT_FIELDS = (
    "latency_cycles",
    "latency_cycles_worst",
    "interval",
    "dsp",
    "bram",
    "lut",
    "ff",
    "uram",
    "slack_ns",
    "estimated_clock_period_ns",
    "requested_clock_period_ns",
    "fmax_mhz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--benchmark", default="hlsfactory_gemm")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def positive_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_int(value: Any) -> str:
    number = positive_number(value)
    return "N/A" if number is None else f"{int(round(number)):,}"


def format_ratio(value: Any) -> str:
    number = positive_number(value)
    return "N/A" if number is None else f"{number:.2f}x"


def extract_pipeline_iis(report: dict[str, Any]) -> tuple[float | None, float | None, int]:
    scopes = (((report.get("feedback") or {}).get("scopes")) or [])
    values = [
        positive_number(scope.get("pipeline_ii"))
        for scope in scopes
        if isinstance(scope, dict)
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None, None, 0
    return min(values), max(values), len(values)


def candidate_report(step: dict[str, Any]) -> tuple[dict[str, Any], str]:
    report = step.get("report")
    if isinstance(report, dict) and report:
        return report, "report"
    rejected = step.get("rejected_report")
    if isinstance(rejected, dict) and rejected:
        return rejected, "rejected_report"
    return {}, "none"


def status_for_step(step: dict[str, Any], report_source: str) -> str:
    if step.get("step_name") == "baseline":
        return "baseline"
    if step.get("success") is True:
        return "synthesized"
    error = str(step.get("attempt_error") or "")
    if report_source == "rejected_report" and "no_op" in error:
        return "rejected_noop"
    if "budget_exhausted" in error:
        return "not_attempted_budget"
    if "csim_failed" in error or "golden-output" in error:
        return "csim_failed"
    if "Synthesis failed" in error or step.get("step_effect") == "synth_failed":
        return "synthesis_failed"
    return "failed"


def best_envelope(data: dict[str, Any], step_count: int) -> list[float | None]:
    history = data.get("best_so_far_history") or []
    observations: list[tuple[int, float]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        step_index = item.get("step_index")
        score = positive_number(item.get("score"))
        if isinstance(step_index, int) and score is not None:
            observations.append((step_index + 1, score))

    envelope: list[float | None] = []
    for ordinal in range(step_count):
        eligible = [score for index, score in observations if index <= ordinal]
        envelope.append(min(eligible) if eligible else None)
    return envelope


def load_setup(path: Path, benchmark: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("benchmark") != benchmark:
        raise ValueError(f"{path}: expected benchmark {benchmark!r}")
    run = data.get("run") or {}
    if run.get("model") != "claude-sonnet-4-6":
        raise ValueError(f"{path}: expected claude-sonnet-4-6")
    if run.get("skill_mode") not in SETUP_ORDER:
        raise ValueError(f"{path}: unsupported skill mode {run.get('skill_mode')!r}")
    data["_source_path"] = str(path.resolve())
    data["_source_sha256"] = sha256_file(path)
    return data


def collect_records(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = data.get("run") or {}
    setup = str(run.get("skill_mode"))
    steps = data.get("generated_step_history") or []
    if not steps or steps[0].get("step_name") != "baseline":
        raise ValueError(f"{data['_source_path']}: missing baseline trajectory entry")

    reference_report = data.get("ground_truth_report") or {}
    reference_cycles = positive_number(
        reference_report.get("latency_cycles_worst")
        or reference_report.get("latency_cycles")
    )
    envelope = best_envelope(data, len(steps))
    records: list[dict[str, Any]] = []

    for ordinal, step in enumerate(steps):
        report, report_source = candidate_report(step)
        nominal = positive_number(report.get("latency_cycles"))
        worst = positive_number(report.get("latency_cycles_worst"))
        qor_cycles = worst or nominal
        min_ii, max_ii, ii_count = extract_pipeline_iis(report)
        feasibility = step.get("feasibility") or {}
        csim = step.get("csim") or {}
        incumbent = envelope[ordinal]
        record: dict[str, Any] = {
            "setup": setup,
            "setup_label": SETUP_LABELS[setup],
            "step_ordinal": ordinal,
            "step_name": step.get("step_name"),
            "step_label": STEP_LABELS.get(str(step.get("step_name")), str(step.get("step_name"))),
            "status": status_for_step(step, report_source),
            "success": step.get("success") is True,
            "step_effect": step.get("step_effect"),
            "report_source": report_source,
            "candidate_latency_cycles": nominal,
            "candidate_worst_cycles": worst,
            "candidate_interval": positive_number(report.get("interval")),
            "candidate_qor_cycles": qor_cycles,
            "candidate_speedup_vs_reference": (
                reference_cycles / qor_cycles
                if reference_cycles is not None and qor_cycles is not None
                else None
            ),
            "best_seen_qor_cycles": incumbent,
            "best_seen_speedup_vs_reference": (
                reference_cycles / incumbent
                if reference_cycles is not None and incumbent is not None
                else None
            ),
            "reference_worst_cycles": reference_cycles,
            "feasible": feasibility.get("feasible"),
            "csim_status": csim.get("status"),
            "min_pipeline_ii": min_ii,
            "max_pipeline_ii": max_ii,
            "pipeline_scope_count": ii_count,
            "error": str(step.get("attempt_error") or ""),
            "source_file": data["_source_path"],
        }
        for field in REPORT_FIELDS:
            if field in {"latency_cycles", "latency_cycles_worst", "interval"}:
                continue
            record[field] = finite_number(report.get(field))
        records.append(record)

    final_report = data.get("final_report") or {}
    final_cycles = positive_number(
        final_report.get("latency_cycles_worst")
        or final_report.get("latency_cycles")
    )
    final_speedup = (
        reference_cycles / final_cycles
        if reference_cycles is not None and final_cycles is not None
        else None
    )
    valid_candidates = [
        record for record in records[1:]
        if record["candidate_qor_cycles"] is not None and record["status"] == "synthesized"
    ]
    regressions = 0
    previous = records[0]["candidate_qor_cycles"]
    for record in valid_candidates:
        current = record["candidate_qor_cycles"]
        if previous is not None and current is not None and current > previous:
            regressions += 1
        previous = current

    best_history = data.get("best_so_far_history") or []
    best_item = min(
        (item for item in best_history if positive_number(item.get("score")) is not None),
        key=lambda item: positive_number(item.get("score")) or float("inf"),
        default={},
    )
    summary = {
        "setup": setup,
        "setup_label": SETUP_LABELS[setup],
        "step_sequence": " -> ".join(str(record["step_name"]) for record in records),
        "optimization_steps": len(records) - 1,
        "synthesized_steps": len(valid_candidates),
        "failed_or_unattempted_steps": len(records) - 1 - len(valid_candidates),
        "raw_qor_regressions": regressions,
        "reference_worst_cycles": reference_cycles,
        "final_latency_cycles": positive_number(final_report.get("latency_cycles")),
        "final_worst_cycles": final_cycles,
        "final_speedup_vs_reference": final_speedup,
        "best_step": best_item.get("step_name"),
        "best_step_index": best_item.get("step_index"),
        "best_score": positive_number(best_item.get("score")),
        "best_promoted_at_end": bool((data.get("best_so_far_promotion") or {}).get("promoted")),
        "final_dsp": finite_number(final_report.get("dsp")),
        "final_bram": finite_number(final_report.get("bram")),
        "final_lut": finite_number(final_report.get("lut")),
        "final_ff": finite_number(final_report.get("ff")),
        "final_uram": finite_number(final_report.get("uram")),
        "final_fmax_mhz": finite_number(final_report.get("fmax_mhz")),
        "reference_isolation_audit_passed": (data.get("reference_isolation_audit") or {}).get("passed"),
        "reference_isolation_findings": (data.get("reference_isolation_audit") or {}).get("finding_count"),
        "source_file": data["_source_path"],
        "source_sha256": data["_source_sha256"],
        "run_fingerprint": ((data.get("run_fingerprint") or {}).get("sha256")),
    }
    return records, summary


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scan_architecture(setup_dirs: list[Path]) -> dict[str, Any]:
    patterns = {
        "explicit_systolic_term": re.compile(r"\bsystolic\b", re.IGNORECASE),
        "hls_stream": re.compile(r"hls\s*::\s*stream"),
        "dataflow_pragma": re.compile(r"#\s*pragma\s+HLS\s+DATAFLOW", re.IGNORECASE),
        "pe_array_identifier": re.compile(r"\b(?:PE|pe)\s*\["),
        "array_partition": re.compile(r"#\s*pragma\s+HLS\s+ARRAY_PARTITION", re.IGNORECASE),
        "pipeline_pragma": re.compile(r"#\s*pragma\s+HLS\s+PIPELINE", re.IGNORECASE),
        "unroll_pragma": re.compile(r"#\s*pragma\s+HLS\s+UNROLL", re.IGNORECASE),
    }
    counts = {name: 0 for name in patterns}
    files_scanned = 0
    for setup_dir in setup_dirs:
        for path in sorted(setup_dir.rglob("*.cpp")):
            files_scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in patterns.items():
                counts[name] += len(pattern.findall(text))
    systolic_evidence = any(
        counts[name] > 0
        for name in (
            "explicit_systolic_term",
            "hls_stream",
            "dataflow_pragma",
            "pe_array_identifier",
        )
    )
    return {
        "benchmark": "hlsfactory_gemm",
        "classification": (
            "systolic architecture evidence found"
            if systolic_evidence
            else "tiled/partitioned/pipelined GEMM; not verified as a systolic array"
        ),
        "systolic_architecture_evidence": systolic_evidence,
        "files_scanned": files_scanned,
        "token_counts": counts,
        "rule": (
            "A matrix-multiplication workload is not itself a systolic implementation. "
            "This audit requires an explicit PE array, stream/dataflow topology, or "
            "systolic declaration in generated code."
        ),
    }


def setup_colors() -> dict[str, str]:
    return {
        "skillless": "#1f77b4",
        "matched": "#ff7f0e",
        "smart_best_fit": "#2ca02c",
        "smart_exhaustive": "#d62728",
        "all_positive": "#9467bd",
    }


def plot_overview(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = setup_colors()
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    for setup in SETUP_ORDER:
        rows = [row for row in records if row["setup"] == setup]
        x = [row["step_ordinal"] for row in rows]
        raw_cycles = [row["candidate_qor_cycles"] or math.nan for row in rows]
        best_cycles = [row["best_seen_qor_cycles"] or math.nan for row in rows]
        raw_speedup = [row["candidate_speedup_vs_reference"] or math.nan for row in rows]
        best_speedup = [row["best_seen_speedup_vs_reference"] or math.nan for row in rows]
        color = colors[setup]
        axes[0].plot(x, best_cycles, color=color, linewidth=2.2)
        axes[0].scatter(x, raw_cycles, facecolors="white", edgecolors=color, s=48, linewidths=1.5, zorder=3)
        axes[1].plot(x, best_speedup, color=color, linewidth=2.2)
        axes[1].scatter(x, raw_speedup, facecolors="white", edgecolors=color, s=48, linewidths=1.5, zorder=3)
        for axis, y_values in ((axes[0], best_cycles), (axes[1], best_speedup)):
            for row, y_value in zip(rows, y_values):
                if row["step_ordinal"] > 0 and row["candidate_qor_cycles"] is None and math.isfinite(y_value):
                    axis.scatter(row["step_ordinal"], y_value, marker="x", color=color, s=55, linewidths=1.7, zorder=4)

    max_step = max(row["step_ordinal"] for row in records)
    for axis in axes:
        axis.set_xticks(range(max_step + 1))
        axis.set_xlabel("Optimization-step ordinal (setup-specific order)")
        axis.grid(True, which="both", axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Worst-case CSynth cycles (log scale)")
    axes[0].set_title("Candidate and best-feasible trajectory")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Speedup over validated HLSFactory reference (log scale)")
    axes[1].set_title("Reference-relative trajectory")

    legend = [
        Line2D([0], [0], color=colors[setup], lw=2.2, label=SETUP_LABELS[setup])
        for setup in SETUP_ORDER
    ]
    legend.extend([
        Line2D([0], [0], color="#333333", lw=2.2, label="Best feasible seen"),
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="white", lw=0, label="Raw synthesized candidate"),
        Line2D([0], [0], color="#333333", marker="x", lw=0, label="No valid CSynth point"),
    ])
    figure.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("Sonnet 4.6 multistep GEMM trajectories, Vitis 2023.2, U280, 3.33 ns", fontsize=14)
    figure.savefig(output / "trajectory_overview.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "trajectory_overview.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_small_multiples(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = setup_colors()
    figure, axes = plt.subplots(2, len(SETUP_ORDER), figsize=(22, 8.5), constrained_layout=True)
    for column, setup in enumerate(SETUP_ORDER):
        rows = [row for row in records if row["setup"] == setup]
        x = [row["step_ordinal"] for row in rows]
        labels = [f"{row['step_ordinal']}\n{row['step_label']}" for row in rows]
        raw_cycles = [row["candidate_qor_cycles"] or math.nan for row in rows]
        best_cycles = [row["best_seen_qor_cycles"] or math.nan for row in rows]
        raw_speedup = [row["candidate_speedup_vs_reference"] or math.nan for row in rows]
        best_speedup = [row["best_seen_speedup_vs_reference"] or math.nan for row in rows]
        color = colors[setup]
        for row_index, (raw, best) in enumerate(((raw_cycles, best_cycles), (raw_speedup, best_speedup))):
            axis = axes[row_index][column]
            axis.plot(x, best, color=color, linewidth=2.4, marker="o", markersize=4, label="best feasible")
            axis.scatter(x, raw, facecolors="white", edgecolors=color, s=50, linewidths=1.5, zorder=3, label="raw candidate")
            for row, y_value in zip(rows, best):
                if row["step_ordinal"] > 0 and row["candidate_qor_cycles"] is None and math.isfinite(y_value):
                    axis.scatter(row["step_ordinal"], y_value, marker="x", color="#202020", s=50, linewidths=1.5, zorder=4)
            axis.set_xticks(x, labels, fontsize=8)
            axis.set_yscale("log")
            axis.grid(True, which="both", axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        axes[0][column].set_title(SETUP_LABELS[setup], fontsize=11)
        axes[1][column].set_xlabel("Step ordinal and operation")
    axes[0][0].set_ylabel("Worst-case CSynth cycles\n(log scale)")
    axes[1][0].set_ylabel("Speedup over reference\n(log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle("Setup-specific Sonnet 4.6 GEMM step order and QoR trajectory", fontsize=15)
    figure.savefig(output / "trajectory_by_setup.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "trajectory_by_setup.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_resource_latency(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = setup_colors()
    figure, axis = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    baseline_drawn = False
    for setup in SETUP_ORDER:
        for row in records:
            if row["setup"] != setup or row["candidate_qor_cycles"] is None or row["dsp"] is None:
                continue
            if row["step_name"] == "baseline":
                if baseline_drawn:
                    continue
                baseline_drawn = True
                color = "#202020"
            else:
                color = colors[setup]
            axis.scatter(row["dsp"], row["candidate_qor_cycles"], color=color, s=55, alpha=0.88)
            axis.annotate(
                f"{row['step_label']}",
                (row["dsp"], row["candidate_qor_cycles"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
    axis.set_yscale("log")
    axis.set_xlabel("DSP count")
    axis.set_ylabel("Worst-case CSynth cycles (log scale)")
    axis.set_title("GEMM resource-latency observations: DSP growth is not a monotone latency guarantee")
    axis.grid(True, which="both", alpha=0.25)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[setup], label=SETUP_LABELS[setup])
        for setup in SETUP_ORDER
    ]
    legend.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#202020", label="Common baseline"))
    axis.legend(handles=legend, frameon=False, ncol=2)
    figure.savefig(output / "resource_latency_scatter.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "resource_latency_scatter.pdf", bbox_inches="tight")
    plt.close(figure)


def qos_contract() -> dict[str, Any]:
    return {
        "schema_version": "c2hls.qor-design-space.v0-proposal",
        "parent_candidate": {
            "code_sha256": "required",
            "csim_golden_sha256": "required",
            "toolchain_fingerprint": "required",
        },
        "objective": {
            "primary": "minimize latency_cycles_worst",
            "constraints": [
                "csim_passed",
                "csynth_passed",
                "timing_met",
                "resource_fit",
            ],
            "secondary": "Pareto rank on DSP, BRAM, LUT, FF, URAM, and slack",
        },
        "knobs": [
            {"name": "unroll_factor", "kind": "integer_divisor", "scope": "loop_id"},
            {"name": "partition_factor", "kind": "integer_divisor", "scope": "array_dimension"},
            {"name": "tile_size", "kind": "integer_tuple", "scope": "loop_nest"},
            {"name": "pe_count", "kind": "integer_divisor", "scope": "compute_region"},
            {"name": "target_ii", "kind": "positive_integer", "scope": "loop_id"},
            {"name": "buffer_count", "kind": "enum", "values": [1, 2]},
            {"name": "operator_binding", "kind": "enum", "values": ["auto", "dsp", "fabric"]},
        ],
        "observation": {
            "required": [
                "changed_knob",
                "changed_value",
                "fixed_knob_values",
                "source_diff_sha256",
                "csim_status",
                "csynth_status",
                "latency_cycles_worst",
                "interval",
                "achieved_pipeline_ii_by_scope",
                "dsp",
                "bram",
                "lut",
                "ff",
                "uram",
                "slack_ns",
                "estimated_clock_period_ns",
            ]
        },
        "search_policy": {
            "stage_1": "one-factor-at-a-time bracket using 3-5 legal values",
            "stage_2": "interaction-aware refinement near Pareto knees",
            "stage_3": "promote only a feasible nondominated winner",
            "reference_blind": True,
        },
    }


def render_report(
    summaries: list[dict[str, Any]],
    architecture: dict[str, Any],
    run_meta: dict[str, Any],
) -> str:
    lines = [
        "# Sonnet 4.6 GEMM Multistep Trajectory",
        "",
        "## Scope and architecture classification",
        "",
        (
            "The closest HLSFactory workload to a systolic-array benchmark in this "
            "campaign is `hlsfactory_gemm`. GEMM is the algorithm, not proof of a "
            "systolic implementation. The generated files contain tiling/local "
            "buffers, array partitioning, pipelining, and some unrolling, but this "
            "audit found no explicit PE array, `hls::stream`, `DATAFLOW`, or systolic "
            "declaration. It is therefore classified as **tiled/partitioned/pipelined "
            "GEMM, not a verified systolic array**."
        ),
        "",
        f"- Generated C++ files scanned: {architecture['files_scanned']}",
        f"- Architecture token counts: `{json.dumps(architecture['token_counts'], sort_keys=True)}`",
        f"- Model/toolchain: `{run_meta.get('model')}`, Vitis `{run_meta.get('vitis_version')}`, "
        f"`{run_meta.get('part')}`, `{run_meta.get('clock_ns')}` ns",
        "",
        "## Final results",
        "",
        "The primary metric is `latency_cycles_worst`, matching the controller's "
        "best-so-far score. Speedup is the validated HLSFactory reference's worst "
        "CSynth cycles divided by candidate worst cycles. Failed CSim/CSynth and "
        "budget-exhausted steps are left missing rather than imputed.",
        "",
        "| Setup | Ordered path | Final worst cycles | Speedup vs ref | DSP | BRAM | Raw regressions | Best step |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['setup_label']} | `{item['step_sequence']}` | "
            f"{format_int(item['final_worst_cycles'])} | "
            f"{format_ratio(item['final_speedup_vs_reference'])} | "
            f"{format_int(item['final_dsp'])} | {format_int(item['final_bram'])} | "
            f"{item['raw_qor_regressions']} | `{item['best_step']}` |"
        )
    lines.extend([
        "",
        "## What changes along the steps",
        "",
        "Yes, the measured y values change substantially, but not monotonically for "
        "every raw candidate. `all_positive` falls from 54,471,422 reference cycles "
        "to 366,397 after coalescing and 74,557 after pipeline. In `skillless`, the "
        "double-buffer candidate has lower nominal latency than unroll but worse "
        "worst-case latency/interval (1,276,801 versus 906,061), before tiling reaches "
        "355,968. In `smart_best_fit`, double buffering reports 193,743 nominal cycles "
        "but 381,440 worst-case cycles; that is worse than coalescing at 361,952, so "
        "the final best-so-far promotion correctly restores coalescing. This is why "
        "the plots show hollow raw-candidate markers and a solid best-feasible envelope.",
        "",
        "The best-feasible curve is non-increasing by construction. That does not "
        "mean each transformation improved QoR; it means the controller retained the "
        "best feasible candidate after observing regressions or failures.",
        "",
        "## Artifact caveats",
        "",
        "- The five setups use different step orders, so a shared x coordinate means "
        "  ordinal, not the same transformation. Use `trajectory_by_setup.*` for exact labels.",
        "- These runs did not execute reference COSIM. Their reference CSim and CSynth passed.",
        "- Every inspected result has `reference_blind=true`, but the stored "
        "  `reference_isolation_audit` reports findings. Treat this as forensic QoR "
        "  evidence, not a leakage-clean publication result, until those findings are adjudicated.",
        "- This is one benchmark and one sample per setup. It demonstrates trajectory "
        "  behavior, not a general causal relationship between a transform and QoR.",
        "",
        "## QoR agent upgrade",
        "",
        "The current `QualityRepairAgent` asks an LLM for one unconstrained code revision "
        "per turn. It sees the current code, a synthesis summary, target/device "
        "utilization, timing evidence, parsed bottlenecks, and benchmark context. The "
        "acceptance rule is feasibility first and lower worst-case latency second. It "
        "does not represent design knobs, hold other knobs fixed, estimate sensitivity, "
        "or retain a Pareto frontier.",
        "",
        "A controlled design-space evaluator is workable and should be added beside, "
        "not embedded inside, free-form LLM repair:",
        "",
        "1. **Typed knob extraction.** Let the LLM propose a bounded schema, then validate "
        "   it deterministically. Knobs include unroll/partition factors, tile sizes, PE "
        "   count, requested II, buffering, and operator binding. DSP count is normally "
        "   an outcome, not a direct knob.",
        "2. **Frozen-parent mutations.** Generate every local candidate from the same "
        "   code hash and alter exactly one typed value during the first sweep. Use AST "
        "   or pragma-aware edits, and record the diff hash and all fixed values.",
        "3. **Gated evaluation.** Run CSim, then CSynth only after CSim passes. Parse "
        "   nominal/worst latency, top-level interval, achieved II per loop, timing, and "
        "   all U280 resource counts. A requested `II=1` is not evidence that II=1 was achieved.",
        "4. **Directional hypotheses.** Do not require every metric to move in one "
        "   positive direction. For example, increasing unroll may reduce cycles while "
        "   increasing DSP, then hit a memory/timing cliff. Test expected signs and "
        "   saturation with at least 3-5 legal values using Spearman rank trend and "
        "   explicit monotonicity-violation counts.",
        "5. **Pareto and constrained selection.** Reject incorrect, timing-failing, or "
        "   device-overfit points. Among feasible points, retain the latency-area-slack "
        "   Pareto frontier. Select minimum worst-case latency subject to explicit "
        "   budgets; do not hide policy in an arbitrary weighted sum.",
        "6. **Interaction refinement.** One-factor-at-a-time identifies local direction "
        "   but misses interactions such as unroll x partition. After bracketing, spend "
        "   the remaining budget on a small factorial or Bayesian refinement around "
        "   frontier knees, with early stop at plateaus or validity cliffs.",
        "7. **Evidence report.** Emit the parent hash, knob table, raw measurements, "
        "   achieved-versus-requested II, rank correlations, Pareto membership, winner "
        "   constraints, and why rejected alternatives lost. Keep external reference "
        "   cycles reporting-only and unavailable to the planner or selector.",
        "",
        "A practical first ablation for GEMM is `unroll_factor in {1,2,4,8,16}` crossed "
        "only after the OFAT stage with matching partition factors, followed by tile "
        "sizes that divide the loop bounds. The result should be a resource-latency "
        "frontier, not an assumption that more DSP or lower requested II always improves latency.",
        "",
        "## Files",
        "",
        "- `trajectory_by_setup.png` / `.pdf`: exact setup-specific step labels",
        "- `trajectory_overview.png` / `.pdf`: overlaid ordinal trajectories",
        "- `resource_latency_scatter.png` / `.pdf`: observed DSP-latency tradeoff",
        "- `trajectory_records.csv`: full per-step metrics and statuses",
        "- `setup_summary.csv`: final setup comparison",
        "- `analysis.json`: machine-readable analysis",
        "- `qor_design_space_contract.example.json`: proposed typed DOE contract",
        "- `manifest.json`: source and output hashes",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    campaign = args.campaign.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    result_paths = sorted(
        campaign.glob(f"{args.benchmark}_sonnet_*/{args.benchmark}_multistep_results.json")
    )
    data_by_setup: dict[str, dict[str, Any]] = {}
    setup_dirs: list[Path] = []
    for path in result_paths:
        data = load_setup(path, args.benchmark)
        setup = str((data.get("run") or {}).get("skill_mode"))
        if setup in data_by_setup:
            raise ValueError(f"duplicate setup {setup!r}: {path}")
        data_by_setup[setup] = data
        setup_dirs.append(path.parent)
    missing = [setup for setup in SETUP_ORDER if setup not in data_by_setup]
    if missing:
        raise ValueError(f"campaign lacks setups: {', '.join(missing)}")

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for setup in SETUP_ORDER:
        setup_records, setup_summary = collect_records(data_by_setup[setup])
        records.extend(setup_records)
        summaries.append(setup_summary)

    architecture = scan_architecture(setup_dirs)
    first_run = data_by_setup[SETUP_ORDER[0]].get("run") or {}
    run_meta = {
        "model": first_run.get("model"),
        "vitis_version": first_run.get("vitis_version"),
        "part": first_run.get("part"),
        "clock_ns": first_run.get("clock_ns"),
        "reference_blind": first_run.get("reference_blind"),
    }

    record_fields = [
        "setup", "setup_label", "step_ordinal", "step_name", "step_label",
        "status", "success", "step_effect", "report_source", "feasible",
        "csim_status", "candidate_latency_cycles", "candidate_worst_cycles",
        "candidate_interval", "candidate_qor_cycles", "candidate_speedup_vs_reference",
        "best_seen_qor_cycles", "best_seen_speedup_vs_reference", "reference_worst_cycles",
        "dsp", "bram", "lut", "ff", "uram", "slack_ns",
        "estimated_clock_period_ns", "requested_clock_period_ns", "fmax_mhz",
        "min_pipeline_ii", "max_pipeline_ii", "pipeline_scope_count", "error", "source_file",
    ]
    summary_fields = list(summaries[0].keys())
    write_csv(output / "trajectory_records.csv", records, record_fields)
    write_csv(output / "setup_summary.csv", summaries, summary_fields)

    analysis = {
        "schema_version": "c2hls.sonnet-gemm-trajectory.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "benchmark": args.benchmark,
        "run": run_meta,
        "architecture_audit": architecture,
        "setups": summaries,
        "records": records,
    }
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "qor_design_space_contract.example.json").write_text(
        json.dumps(qos_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        render_report(summaries, architecture, run_meta),
        encoding="utf-8",
    )

    plot_overview(records, output)
    plot_small_multiples(records, output)
    plot_resource_latency(records, output)

    source_manifest = [
        {
            "path": data_by_setup[setup]["_source_path"],
            "sha256": data_by_setup[setup]["_source_sha256"],
            "setup": setup,
        }
        for setup in SETUP_ORDER
    ]
    output_manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            output_manifest.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        "schema_version": "c2hls.analysis-manifest.v1",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sources": source_manifest,
        "outputs": output_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} trajectory records for {len(summaries)} setups to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
