#!/usr/bin/env python3
"""Export Sonnet skill-v2 latency tables and large comparison bar charts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator


REPO = Path("/home/luo00466/code_translation-c2hls-hpca2027")
DEFAULT_INPUT = (
    REPO
    / "artifacts/analysis/hlsfactory_cycle_setup_comparison_20260725.rows.csv"
)
DEFAULT_OUTPUT = (
    REPO / "results_sweeps/sonnet46_skillv2_latency_matrix_20260726"
)
FAMILIES = {"sonnet46_skillv2_flash", "sonnet46_skillv2_multistep"}
STRATEGIES = ["flash", "multistep"]
SKILL_MODES = [
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
]
SKILL_LABELS = {
    "skillless": "Skillless",
    "matched": "Matched",
    "smart_best_fit": "Smart best-fit",
    "smart_exhaustive": "Smart exhaustive",
    "all_positive": "All-positive",
}
SKILL_ABBREVIATIONS = {
    "skillless": "SL",
    "matched": "MA",
    "smart_best_fit": "BF",
    "smart_exhaustive": "EX",
    "all_positive": "AP",
}
SKILL_COLORS = {
    "skillless": "#666666",
    "matched": "#0072B2",
    "smart_best_fit": "#009E73",
    "smart_exhaustive": "#E69F00",
    "all_positive": "#CC79A7",
}
CLOCK_NS = 3.33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def geometric_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric) & (numeric > 0)]
    if numeric.empty:
        return None
    return float(np.exp(np.log(numeric).mean()))


def compact_number(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    number = float(value)
    absolute = abs(number)
    if absolute >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:.0f}"


def problem_label(benchmark: str) -> str:
    return benchmark.removeprefix("hlsfactory_")


def axis_problem_label(benchmark: str) -> str:
    label = problem_label(benchmark)
    if "_" in label:
        return label.replace("_", "\n")
    return label


def combination_order() -> list[tuple[str, str]]:
    return [
        (strategy, skill_mode)
        for strategy in STRATEGIES
        for skill_mode in SKILL_MODES
    ]


def combination_label(strategy: str, skill_mode: str) -> str:
    return f"{strategy.title()} | {SKILL_LABELS[skill_mode]}"


def combination_key(strategy: str, skill_mode: str) -> str:
    return f"{strategy}__{skill_mode}"


def combination_abbreviation(strategy: str, skill_mode: str) -> str:
    prefix = "F" if strategy == "flash" else "M"
    return f"{prefix}-{SKILL_ABBREVIATIONS[skill_mode]}"


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame[
        (frame["model"] == "claude-sonnet-4-6")
        & frame["family"].isin(FAMILIES)
    ].copy()
    frame["valid_csim_csynth"] = as_bool(frame["valid_csim_csynth"])
    frame["skill_injection_known"] = as_bool(frame["skill_injection_known"])
    frame["reference_isolation_audit_passed"] = as_bool(
        frame["reference_isolation_audit_passed"]
    )
    for column in [
        "cycles",
        "reference_cycles",
        "speedup_vs_reference",
        "phase_b_initial_cycles",
        "skill_injected_count",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["latency_cycles"] = frame["cycles"].where(
        frame["valid_csim_csynth"]
    )
    frame["latency_ns_at_3_33ns"] = frame["latency_cycles"] * CLOCK_NS
    frame["reference_latency_ns_at_3_33ns"] = (
        frame["reference_cycles"] * CLOCK_NS
    )
    frame["problem"] = frame["benchmark"].map(problem_label)
    frame["combination"] = [
        combination_key(strategy, skill_mode)
        for strategy, skill_mode in zip(
            frame["strategy"], frame["skill_mode"], strict=True
        )
    ]

    expected_combinations = {
        combination_key(strategy, skill_mode)
        for strategy, skill_mode in combination_order()
    }
    actual_combinations = set(frame["combination"])
    if actual_combinations != expected_combinations:
        raise ValueError(
            "unexpected setup combinations: "
            f"missing={sorted(expected_combinations - actual_combinations)} "
            f"extra={sorted(actual_combinations - expected_combinations)}"
        )
    duplicate = frame.duplicated(["benchmark", "combination"], keep=False)
    if duplicate.any():
        rows = frame.loc[duplicate, ["benchmark", "combination"]]
        raise ValueError(f"duplicate benchmark/setup rows:\n{rows}")
    benchmarks = sorted(frame["benchmark"].unique())
    if len(benchmarks) != 28:
        raise ValueError(f"expected 28 benchmarks, found {len(benchmarks)}")
    if len(frame) != 280:
        raise ValueError(f"expected 280 matrix cells, found {len(frame)}")
    return frame


def export_long_table(frame: pd.DataFrame, output_dir: Path) -> Path:
    columns = [
        "benchmark",
        "problem",
        "strategy",
        "skill_mode",
        "setup_label",
        "correctness_status",
        "synthesis_status",
        "valid_csim_csynth",
        "latency_cycles",
        "latency_ns_at_3_33ns",
        "reference_cycles",
        "reference_latency_ns_at_3_33ns",
        "speedup_vs_reference",
        "cycle_source",
        "selected_step_name",
        "phase_b_initial_cycles",
        "skill_injection_known",
        "skill_injected_count",
        "skill_injected_ids",
        "reference_isolation_audit_passed",
        "reference_source_kind",
        "source_path",
        "source_result_path",
    ]
    ordered = frame.copy()
    ordered["strategy"] = pd.Categorical(
        ordered["strategy"], categories=STRATEGIES, ordered=True
    )
    ordered["skill_mode"] = pd.Categorical(
        ordered["skill_mode"], categories=SKILL_MODES, ordered=True
    )
    ordered = ordered.sort_values(["benchmark", "strategy", "skill_mode"])
    path = output_dir / "sonnet46_skillv2_latency_records_long.csv"
    ordered[columns].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def build_wide_table(frame: pd.DataFrame) -> pd.DataFrame:
    reference = (
        frame.groupby("benchmark", sort=True)["reference_cycles"]
        .first()
        .rename("gold_reference_cycles")
    )
    wide = frame.pivot(
        index="benchmark", columns="combination", values="latency_cycles"
    )
    ordered_columns = [
        combination_key(strategy, skill_mode)
        for strategy, skill_mode in combination_order()
    ]
    wide = wide.reindex(columns=ordered_columns)
    wide = reference.to_frame().join(wide).reset_index()
    wide.insert(1, "problem", wide["benchmark"].map(problem_label))
    return wide


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in headers:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif column.endswith("cycles"):
                values.append(f"{float(value):,.0f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def export_wide_tables(
    wide: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    csv_path = output_dir / "sonnet46_skillv2_latency_cycles_wide.csv"
    markdown_path = output_dir / "sonnet46_skillv2_latency_cycles_wide.md"
    wide.to_csv(csv_path, index=False)
    markdown_path.write_text(markdown_table(wide), encoding="utf-8")
    return csv_path, markdown_path


def export_setup_summary(frame: pd.DataFrame, output_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for strategy, skill_mode in combination_order():
        selected = frame[
            (frame["strategy"] == strategy)
            & (frame["skill_mode"] == skill_mode)
        ]
        valid = selected[selected["valid_csim_csynth"]]
        records.append(
            {
                "strategy": strategy,
                "skill_mode": skill_mode,
                "setup": combination_label(strategy, skill_mode),
                "attempted": int(len(selected)),
                "csim_passed": int(
                    (selected["correctness_status"] == "passed").sum()
                ),
                "csynth_passed": int(
                    (selected["synthesis_status"] == "passed").sum()
                ),
                "valid_csim_csynth": int(len(valid)),
                "geomean_latency_cycles": geometric_mean(
                    valid["latency_cycles"]
                ),
                "median_latency_cycles": (
                    float(valid["latency_cycles"].median())
                    if not valid.empty
                    else None
                ),
                "geomean_speedup_vs_reference": geometric_mean(
                    valid["speedup_vs_reference"]
                ),
                "wins_vs_reference": int(
                    (valid["speedup_vs_reference"] > 1).sum()
                ),
            }
        )
    path = output_dir / "sonnet46_skillv2_setup_summary.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path, records


def style_axes(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: compact_number(value))
    )
    ax.grid(axis="y", which="major", color="#D6D6D6", linewidth=0.8)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.45)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def legend_handles() -> list[Any]:
    handles: list[Any] = []
    for strategy, skill_mode in combination_order():
        handles.append(
            Patch(
                facecolor=SKILL_COLORS[skill_mode],
                edgecolor="#222222",
                linewidth=0.5,
                hatch="" if strategy == "flash" else "///",
                label=combination_label(strategy, skill_mode),
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="D",
            markersize=5,
            linestyle="--",
            linewidth=1.2,
            label="Gold/reference",
        )
    )
    return handles


def chart_arrays(
    frame: pd.DataFrame,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    benchmarks = sorted(frame["benchmark"].unique())
    values: dict[str, np.ndarray] = {}
    for strategy, skill_mode in combination_order():
        selected = (
            frame[
                (frame["strategy"] == strategy)
                & (frame["skill_mode"] == skill_mode)
            ]
            .set_index("benchmark")
            .reindex(benchmarks)
        )
        values[combination_key(strategy, skill_mode)] = selected[
            "latency_cycles"
        ].to_numpy(dtype=float)
    reference = (
        frame.groupby("benchmark")["reference_cycles"]
        .first()
        .reindex(benchmarks)
        .to_numpy(dtype=float)
    )
    return benchmarks, values, reference


def plot_grouped_chart(
    frame: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    benchmarks, values, reference = chart_arrays(frame)
    combinations = combination_order()
    x = np.arange(len(benchmarks), dtype=float)
    group_width = 0.84
    bar_width = group_width / len(combinations)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "hatch.linewidth": 0.7,
        }
    )
    fig, ax = plt.subplots(figsize=(44, 15))
    for index, (strategy, skill_mode) in enumerate(combinations):
        offset = (index - (len(combinations) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            values[combination_key(strategy, skill_mode)],
            width=bar_width * 0.94,
            color=SKILL_COLORS[skill_mode],
            edgecolor="#222222",
            linewidth=0.35,
            hatch="" if strategy == "flash" else "///",
            zorder=3,
        )

    ax.plot(
        x,
        reference,
        color="#111111",
        marker="D",
        markersize=4.5,
        linestyle="--",
        linewidth=1.2,
        zorder=5,
    )
    finite_values = np.concatenate(
        [array[np.isfinite(array) & (array > 0)] for array in values.values()]
        + [reference[np.isfinite(reference) & (reference > 0)]]
    )
    ax.set_ylim(finite_values.min() / 2.2, finite_values.max() * 2.2)
    ax.set_xlim(-0.7, len(benchmarks) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [axis_problem_label(benchmark) for benchmark in benchmarks],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Worst-case latency (cycles, log scale)", fontsize=12)
    ax.set_xlabel("HLSFactory PolyBench problem", fontsize=12)
    style_axes(ax)
    for boundary in np.arange(0.5, len(benchmarks) - 0.5, 1):
        ax.axvline(boundary, color="#F0F0F0", linewidth=0.6, zorder=0)

    missing_mask = np.ones(len(benchmarks), dtype=bool)
    for array in values.values():
        missing_mask &= ~np.isfinite(array)
    for index in np.where(missing_mask)[0]:
        reference_value = reference[index]
        y = (
            reference_value * 1.5
            if math.isfinite(reference_value)
            else finite_values.min()
        )
        ax.text(
            index,
            y,
            "no valid\ncandidate",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8B0000",
        )

    fig.suptitle(
        "Sonnet 4.6 HLSFactory CSYNTH Latency Across All Skill-v2 Combinations",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        "Vitis 2023.2 | Xilinx U280 | 3.33 ns | bars require CSIM and CSYNTH pass",
        fontsize=12,
        pad=16,
        fontweight="normal",
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=6,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.01,
        0.015,
        "Flash bars are solid; multistep bars are hatched. Missing bars are "
        "failed or unexecuted CSIM/CSYNTH results. Current fresh rerun is not included.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.19)

    png_path = output_dir / "sonnet46_skillv2_latency_grouped_bar_log.png"
    pdf_path = output_dir / "sonnet46_skillv2_latency_grouped_bar_log.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_faceted_chart(
    frame: pd.DataFrame, output_dir: Path
) -> tuple[Path, Path]:
    benchmarks, values, reference = chart_arrays(frame)
    combinations = combination_order()
    abbreviations = [
        combination_abbreviation(strategy, skill_mode)
        for strategy, skill_mode in combinations
    ]
    colors = [SKILL_COLORS[skill_mode] for _, skill_mode in combinations]
    hatches = ["" if strategy == "flash" else "///" for strategy, _ in combinations]

    fig, axes = plt.subplots(7, 4, figsize=(28, 34))
    for benchmark_index, (ax, benchmark) in enumerate(
        zip(axes.flat, benchmarks, strict=True)
    ):
        row_values = np.array(
            [
                values[combination_key(strategy, skill_mode)][benchmark_index]
                for strategy, skill_mode in combinations
            ],
            dtype=float,
        )
        bars = ax.bar(
            np.arange(len(combinations)),
            row_values,
            color=colors,
            edgecolor="#222222",
            linewidth=0.45,
            width=0.78,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)

        reference_value = reference[benchmark_index]
        finite = row_values[np.isfinite(row_values) & (row_values > 0)]
        scale_values = finite
        if math.isfinite(reference_value) and reference_value > 0:
            scale_values = np.append(scale_values, reference_value)
            ax.axhline(
                reference_value,
                color="#111111",
                linestyle="--",
                linewidth=1.0,
                zorder=4,
            )
        ax.set_title(problem_label(benchmark), fontsize=13, pad=7)
        ax.set_xticks(np.arange(len(combinations)))
        ax.set_xticklabels(abbreviations, rotation=55, ha="right", fontsize=7)

        if finite.size:
            lower = scale_values.min() / 2.4
            upper = scale_values.max() * 5.0
            ax.set_ylim(lower, upper)
            for bar, value in zip(bars, row_values, strict=True):
                if not math.isfinite(value) or value <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.12,
                    compact_number(value),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.5,
                    color="#222222",
                )
        else:
            lower = reference_value / 4 if reference_value > 0 else 1
            upper = reference_value * 4 if reference_value > 0 else 10
            ax.set_ylim(lower, upper)
            ax.text(
                0.5,
                0.5,
                "No valid generated\nCSIM + CSYNTH result",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#8B0000",
            )
        if math.isfinite(reference_value) and reference_value > 0:
            ax.text(
                0.02,
                0.97,
                f"gold {compact_number(reference_value)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color="#111111",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
            )
        style_axes(ax)
        if benchmark_index % 4 == 0:
            ax.set_ylabel("Cycles (log)", fontsize=9)

    fig.suptitle(
        "Sonnet 4.6 HLSFactory CSYNTH Latency by Problem",
        fontsize=24,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.979),
        ncol=6,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.006,
        "Vitis 2023.2 | Xilinx U280 | 3.33 ns | F = flash, M = multistep, "
        "SL = skillless, MA = matched, BF = smart best-fit, "
        "EX = smart exhaustive, AP = all-positive",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.015, 0.02, 0.995, 0.955), h_pad=2.2, w_pad=1.4)

    png_path = output_dir / "sonnet46_skillv2_latency_faceted_bar_log.png"
    pdf_path = output_dir / "sonnet46_skillv2_latency_faceted_bar_log.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_manifest(
    *,
    input_path: Path,
    output_dir: Path,
    frame: pd.DataFrame,
    setup_summary: list[dict[str, Any]],
    files: list[Path],
) -> Path:
    valid = frame[frame["valid_csim_csynth"]]
    manifest = {
        "schema_version": "c2hls.sonnet-latency-visualization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path.resolve()),
            "sha256": sha256(input_path),
            "families": sorted(FAMILIES),
            "model": "claude-sonnet-4-6",
        },
        "target": {
            "vitis_version": "2023.2",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": CLOCK_NS,
        },
        "matrix": {
            "benchmarks": int(frame["benchmark"].nunique()),
            "combinations": int(frame["combination"].nunique()),
            "cells": int(len(frame)),
            "valid_csim_csynth": int(len(valid)),
            "invalid_or_unexecuted": int(len(frame) - len(valid)),
            "cosim_included": False,
            "pending_skillless_rerun_included": False,
        },
        "setup_summary": setup_summary,
        "caveats": [
            "CSIM is functional and does not report hardware cycles.",
            "Plotted latency is CSYNTH worst-case latency in cycles.",
            "The y-axis is logarithmic because the latency range spans several orders of magnitude.",
            "The fresh 2026-07-26 multistep-skillless rerun is pending and is not included.",
            "Reference-isolation audit findings remain advisory for this July 24 matrix.",
        ],
        "files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(input_path)
    long_csv = export_long_table(frame, output_dir)
    wide = build_wide_table(frame)
    wide_csv, wide_markdown = export_wide_tables(wide, output_dir)
    summary_csv, summary_records = export_setup_summary(frame, output_dir)
    grouped_png, grouped_pdf = plot_grouped_chart(frame, output_dir)
    faceted_png, faceted_pdf = plot_faceted_chart(frame, output_dir)
    files = [
        long_csv,
        wide_csv,
        wide_markdown,
        summary_csv,
        grouped_png,
        grouped_pdf,
        faceted_png,
        faceted_pdf,
    ]
    manifest = write_manifest(
        input_path=input_path,
        output_dir=output_dir,
        frame=frame,
        setup_summary=summary_records,
        files=files,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "benchmarks": int(frame["benchmark"].nunique()),
                "combinations": int(frame["combination"].nunique()),
                "cells": int(len(frame)),
                "valid": int(frame["valid_csim_csynth"].sum()),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
