#!/usr/bin/env python3
"""Consolidate GEMM linked-control campaigns into a per-step rollout audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


RESOURCE_KEYS = ("dsp", "bram", "lut", "ff", "uram")
METRIC_KEYS = (
    "latency_cycles_worst",
    "interval",
    "estimated_clock_period_ns",
    "achieved_pipeline_ii_max",
    *RESOURCE_KEYS,
)
STEP_ORDER = (
    "parallel_lanes",
    "pipeline_ii",
    "k_strip_size",
    "ping_pong_depth",
    "axi_width",
    "multiplier_implementation",
)
STEP_SPECS = {
    "parallel_lanes": {
        "case_id": "gemm_striped_compute_lanes_expanded",
        "label": "Parallel lanes",
        "control": "GEMM_LANES",
        "axis": "linked lanes",
        "origin_step": "unroll",
        "numeric": True,
        "log_x": True,
        "primary_family": "striped_accumulators",
    },
    "pipeline_ii": {
        "case_id": "gemm_pipeline_compute_ii",
        "label": "Compute pipeline II",
        "control": "GEMM_COMPUTE_II",
        "axis": "requested II",
        "origin_step": "pipeline",
        "numeric": True,
        "log_x": False,
        "primary_family": "gemm_pipeline_compute_ii",
    },
    "k_strip_size": {
        "case_id": "gemm_tiling_k_strip",
        "label": "K strip size",
        "control": "GEMM_K_TILE",
        "axis": "K strip size",
        "origin_step": "tiling",
        "numeric": True,
        "log_x": True,
        "primary_family": "gemm_tiling_k_strip",
    },
    "ping_pong_depth": {
        "case_id": "gemm_doublebuffer_stream_depth",
        "label": "Ping-pong tile depth",
        "control": "GEMM_BUFFER_DEPTH",
        "axis": "tile depth",
        "origin_step": "doublebuffer",
        "numeric": True,
        "log_x": True,
        "primary_family": "gemm_doublebuffer_stream_depth",
    },
    "axi_width": {
        "case_id": "gemm_coalescing_axi_width",
        "label": "AXI width cap",
        "control": "GEMM_AXI_BITS",
        "axis": "width (bits)",
        "origin_step": "coalescing",
        "numeric": True,
        "log_x": True,
        "primary_family": "gemm_coalescing_axi_width",
    },
    "multiplier_implementation": {
        "case_id": "gemm_resource_mul_limit",
        "label": "Multiplier implementation",
        "control": "GEMM_MUL_IMPL",
        "axis": "implementation",
        "origin_step": "resource",
        "numeric": False,
        "log_x": False,
        "primary_family": "gemm_resource_mul_limit",
    },
}
CASE_TO_STEP = {
    str(spec["case_id"]): step_id for step_id, spec in STEP_SPECS.items()
}
COLORS = {
    "parallel_lanes": "#0072B2",
    "pipeline_ii": "#D55E00",
    "k_strip_size": "#009E73",
    "ping_pong_depth": "#CC79A7",
    "axi_width": "#E69F00",
    "multiplier_implementation": "#6C757D",
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_sum(row: dict[str, Any]) -> float:
    resources = [_number(row.get(key)) for key in RESOURCE_KEYS]
    return (
        sum(value for value in resources if value is not None)
        if all(value is not None for value in resources)
        else float("inf")
    )


def _selection_key(row: dict[str, Any]) -> tuple[float, float, str]:
    cycles = _number(row.get("latency_cycles_worst"))
    return (
        cycles if row.get("feasible") is True and cycles is not None else float("inf"),
        _resource_sum(row),
        str(row.get("candidate_id") or ""),
    )


def _copy_metrics(source: dict[str, Any]) -> dict[str, Any]:
    return {key: _number(source.get(key)) for key in METRIC_KEYS}


def _attach_parent_ratios(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["comparison_family"]), []).append(row)
    for family, selected in grouped.items():
        parents = [row for row in selected if row.get("is_parent") is True]
        if len(parents) != 1:
            raise ValueError(f"expected one frozen parent for {family}, found {len(parents)}")
        parent = parents[0]
        for row in selected:
            row["parent_candidate_id"] = parent.get("candidate_id")
            for metric in METRIC_KEYS:
                value = _number(row.get(metric))
                parent_value = _number(parent.get(metric))
                row[f"parent_{metric}"] = parent_value
                row[f"{metric}_ratio"] = (
                    value / parent_value
                    if value is not None and parent_value not in (None, 0)
                    else None
                )
            cycles = _number(row.get("latency_cycles_worst"))
            parent_cycles = _number(parent.get("latency_cycles_worst"))
            row["speedup_vs_parent"] = (
                parent_cycles / cycles
                if cycles not in (None, 0) and parent_cycles is not None
                else None
            )


def _normalize_lane_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        architecture = str(record.get("architecture") or "unknown")
        primary = architecture == STEP_SPECS["parallel_lanes"]["primary_family"]
        requested = _number(record.get("requested_lanes"))
        rows.append({
            "schema_version": "c2hls.gemm-parameter-rollout-measurement.v1",
            "benchmark": "hlsfactory_gemm",
            "step_id": "parallel_lanes",
            "step_label": STEP_SPECS["parallel_lanes"]["label"],
            "origin_step": "unroll",
            "control_name": "GEMM_LANES",
            "case_id": "gemm_striped_compute_lanes_expanded",
            "comparison_family": f"parallel_lanes:{architecture}",
            "architecture": architecture,
            "architecture_label": record.get("architecture_label"),
            "included_in_step_comparison": primary,
            "parameter_value": int(requested) if requested is not None else None,
            "parameter_label": str(int(requested)) if requested is not None else "N/A",
            "parameter_numeric": requested,
            "effective_parameter_value": _number(record.get("effective_lanes")),
            "is_parent": record.get("is_family_parent") is True,
            "candidate_id": record.get("candidate_id"),
            "code_sha256": record.get("code_sha256"),
            "status": "feasible" if record.get("feasible") is True else "invalid",
            "feasible": record.get("feasible") is True,
            "csim_status": record.get("csim_status"),
            "compiler_control_status": (
                "target_realized" if record.get("feasible") is True else "not_realized"
            ),
            "compiler_realization_evidence": record.get(
                "compiler_realization_evidence"
            ),
            **_copy_metrics(record),
        })
    _attach_parent_ratios(rows)
    return rows


def _normalize_step_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        case_id = str(record.get("case_id") or "")
        step_id = CASE_TO_STEP.get(case_id)
        if step_id is None:
            raise ValueError(f"unrecognized GEMM rollout case: {case_id}")
        spec = STEP_SPECS[step_id]
        numeric = _number(record.get("parameter_value")) if spec["numeric"] else None
        rows.append({
            "schema_version": "c2hls.gemm-parameter-rollout-measurement.v1",
            "benchmark": record.get("benchmark") or "hlsfactory_gemm",
            "step_id": step_id,
            "step_label": spec["label"],
            "origin_step": spec["origin_step"],
            "control_name": spec["control"],
            "case_id": case_id,
            "comparison_family": case_id,
            "architecture": None,
            "architecture_label": None,
            "included_in_step_comparison": True,
            "parameter_value": record.get("parameter_value"),
            "parameter_label": str(record.get("parameter_label") or "N/A"),
            "parameter_numeric": numeric,
            "effective_parameter_value": record.get("compiler_effective_value"),
            "is_parent": record.get("is_parent") is True,
            "candidate_id": record.get("candidate_id"),
            "code_sha256": record.get("code_sha256"),
            "status": record.get("status"),
            "feasible": record.get("feasible") is True,
            "csim_status": record.get("csim_status"),
            "compiler_control_status": record.get("compiler_control_status"),
            "compiler_realization_evidence": record.get(
                "compiler_realization_evidence"
            ),
            **_copy_metrics(record),
        })
    _attach_parent_ratios(rows)
    return rows


def _primary_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {step_id: [] for step_id in STEP_ORDER}
    for row in rows:
        if row.get("included_in_step_comparison") is True:
            grouped[str(row["step_id"])].append(row)
    missing = [step_id for step_id, selected in grouped.items() if not selected]
    if missing:
        raise ValueError(f"missing primary GEMM rollout steps: {', '.join(missing)}")
    return grouped


def _step_summaries(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for step_id, selected in _primary_groups(rows).items():
        if STEP_SPECS[step_id]["numeric"]:
            selected = sorted(
                selected,
                key=lambda row: float(row["parameter_numeric"]),
            )
        parent = next(row for row in selected if row.get("is_parent") is True)
        best = min(selected, key=_selection_key)
        values = [row["parameter_label"] for row in selected]
        summaries.append({
            "schema_version": "c2hls.gemm-parameter-rollout-summary.v1",
            "benchmark": "hlsfactory_gemm",
            "step_id": step_id,
            "step_label": STEP_SPECS[step_id]["label"],
            "origin_step": STEP_SPECS[step_id]["origin_step"],
            "control_name": STEP_SPECS[step_id]["control"],
            "tested_values": values,
            "point_count_including_parent": len(selected),
            "candidate_count": len(selected) - 1,
            "feasible_point_count": sum(row.get("feasible") is True for row in selected),
            "parent_parameter_value": parent.get("parameter_value"),
            "parent_parameter_label": parent.get("parameter_label"),
            "parent_candidate_id": parent.get("candidate_id"),
            "parent_cycles": parent.get("latency_cycles_worst"),
            "best_parameter_value": best.get("parameter_value"),
            "best_parameter_label": best.get("parameter_label"),
            "best_candidate_id": best.get("candidate_id"),
            "best_cycles": best.get("latency_cycles_worst"),
            "best_speedup_vs_parent": best.get("speedup_vs_parent"),
            "best_is_parent": best.get("is_parent") is True,
            "best_dsp": best.get("dsp"),
            "best_bram": best.get("bram"),
            "best_lut": best.get("lut"),
            "best_ff": best.get("ff"),
            "best_uram": best.get("uram"),
            "best_compiler_control_status": best.get("compiler_control_status"),
        })
    return summaries


def _write_rows(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    (output_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _parameter_positions(
    selected: list[dict[str, Any]], spec: dict[str, Any]
) -> tuple[list[float], list[str]]:
    if spec["numeric"]:
        values = [float(row["parameter_numeric"]) for row in selected]
        return values, [row["parameter_label"] for row in selected]
    return list(range(len(selected))), [row["parameter_label"] for row in selected]


def _plot_parameter_curves(rows: list[dict[str, Any]], output_dir: Path) -> None:
    grouped = _primary_groups(rows)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for axis, step_id in zip(axes.flat, STEP_ORDER):
        spec = STEP_SPECS[step_id]
        selected = sorted(
            grouped[step_id],
            key=lambda row: (
                _number(row.get("parameter_numeric"))
                if spec["numeric"]
                else next(
                    index
                    for index, value in enumerate(
                        [item["parameter_label"] for item in grouped[step_id]]
                    )
                    if value == row["parameter_label"]
                )
            ),
        )
        x_values, labels = _parameter_positions(selected, spec)
        valid = [
            (x, row)
            for x, row in zip(x_values, selected)
            if row.get("feasible") is True
            and _number(row.get("latency_cycles_worst")) is not None
        ]
        axis.plot(
            [item[0] for item in valid],
            [item[1]["latency_cycles_worst"] for item in valid],
            color=COLORS[step_id],
            marker="o",
            linewidth=2,
        )
        parent = next(row for row in selected if row.get("is_parent") is True)
        best = min(selected, key=_selection_key)
        parent_x = x_values[selected.index(parent)]
        best_x = x_values[selected.index(best)]
        axis.scatter(
            [parent_x], [parent["latency_cycles_worst"]], marker="s", s=72,
            facecolor="white", edgecolor="black", linewidth=1.3, zorder=4,
            label="frozen parent",
        )
        axis.scatter(
            [best_x], [best["latency_cycles_worst"]], marker="*", s=145,
            color="#F0B429", edgecolor="black", linewidth=0.7, zorder=5,
            label="selected best",
        )
        invalid = [
            (x, row) for x, row in zip(x_values, selected)
            if row.get("feasible") is not True
        ]
        if invalid:
            floor = min(item[1]["latency_cycles_worst"] for item in valid) * 0.78
            axis.scatter(
                [item[0] for item in invalid], [floor] * len(invalid), marker="x",
                s=70, color="#B22222", linewidth=2, label="CSynth invalid",
            )
        axis.set_title(str(spec["label"]), fontsize=11)
        axis.set_xlabel(str(spec["axis"]))
        axis.set_ylabel("CSynth cycles")
        axis.set_yscale("log")
        if spec["numeric"] and spec["log_x"]:
            axis.set_xscale("log", base=2)
        rotation = 25 if not spec["numeric"] or len(labels) > 8 else 0
        axis.set_xticks(
            x_values,
            labels=labels,
            rotation=rotation,
            ha="right" if rotation else "center",
        )
        axis.grid(True, which="both", alpha=0.24)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    deduped = dict(zip(labels, handles))
    fig.legend(
        deduped.values(), deduped.keys(), loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.01), frameon=False,
    )
    fig.suptitle("GEMM one-factor parameter effects", y=1.045, fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output_dir, "gemm_parameter_effect_curves")


def _plot_parent_best(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [str(item["step_label"]) for item in summaries]
    positions = np.arange(len(labels))
    parent = np.asarray([float(item["parent_cycles"]) for item in summaries])
    best = np.asarray([float(item["best_cycles"]) for item in summaries])
    width = 0.36
    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    axis.bar(positions - width / 2, parent, width, color="#6C757D", label="parent")
    axis.bar(positions + width / 2, best, width, color="#009E73", label="best")
    for index, item in enumerate(summaries):
        axis.text(
            positions[index] + width / 2,
            best[index] * 1.13,
            f"{float(item['best_speedup_vs_parent']):.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_yscale("log")
    axis.set_xticks(positions, labels=labels, rotation=22, ha="right")
    axis.set_ylabel("worst-case CSynth cycles")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend()
    axis.set_title("Frozen parent versus selected per-step point")
    _save_figure(fig, output_dir, "gemm_parent_vs_best_cycles")


def _plot_winner_resources(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path
) -> None:
    grouped = _primary_groups(rows)
    matrix = []
    for summary in summaries:
        selected = grouped[str(summary["step_id"])]
        best = next(
            row for row in selected
            if row.get("candidate_id") == summary.get("best_candidate_id")
        )
        matrix.append([
            _number(best.get(f"{resource}_ratio")) or 0 for resource in RESOURCE_KEYS
        ])
    values = np.asarray(matrix, dtype=float)
    log_values = np.log2(np.clip(values, 1e-6, None))
    limit = max(1.0, float(np.max(np.abs(log_values))))
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    image = axis.imshow(log_values, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index, row_index, f"{values[row_index, column_index]:.2f}x",
                ha="center", va="center", fontsize=8,
            )
    axis.set_xticks(np.arange(len(RESOURCE_KEYS)), labels=[key.upper() for key in RESOURCE_KEYS])
    axis.set_yticks(
        np.arange(len(summaries)), labels=[item["step_label"] for item in summaries]
    )
    axis.set_title("Selected-point resources relative to each frozen parent")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("log2(resource ratio)")
    fig.tight_layout()
    _save_figure(fig, output_dir, "gemm_winner_resource_ratios")


def _format_value(value: Any) -> str:
    numeric = _number(value)
    if numeric is None:
        return "N/A"
    return f"{int(numeric):,}" if numeric.is_integer() else f"{numeric:.3f}"


def _report(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]], inputs: dict[str, Path]
) -> str:
    primary = [row for row in rows if row.get("included_in_step_comparison") is True]
    candidate_count = sum(not row.get("is_parent") for row in primary)
    feasible_candidate_count = sum(
        not row.get("is_parent") and row.get("feasible") is True for row in primary
    )
    lines = [
        "# GEMM Per-Step Parameter Rollout",
        "",
        "Toolchain: **Vitis 2023.2**, **xcu280-fsvh2892-2L-e**, **3.33 ns**  ",
        "Search: **reference-blind**, **CSim + CSynth**, **no COSIM**, **no model calls**  ",
        f"Primary OFAT points including parents: **{len(primary)}**  ",
        f"Candidate variants: **{candidate_count}**; CSynth-feasible: "
        f"**{feasible_candidate_count}/{candidate_count}**",
        "",
        "## Best point from each step",
        "",
        "| Optimization step | Control | Tested values | Feasible | Parent | Best | Parent cycles | Best cycles | Speedup |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['step_label']} | `{summary['control_name']}` | "
            f"{', '.join(summary['tested_values'])} | "
            f"{summary['feasible_point_count']}/{summary['point_count_including_parent']} | "
            f"{summary['parent_parameter_label']} | {summary['best_parameter_label']} | "
            f"{_format_value(summary['parent_cycles'])} | "
            f"{_format_value(summary['best_cycles'])} | "
            f"{float(summary['best_speedup_vs_parent']):.3f}x |"
        )
    lines.extend([
        "",
        "## What the expanded ranges show",
        "",
        "- **Parallel lanes:** the striped control is genuinely active and strongly "
        "non-monotonic. The 80-lane point reaches the compiler's II=1 schedule and "
        "wins latency, but at a large DSP/LUT/FF cost. This is a latency winner, not "
        "automatically a resource-constrained winner.",
        "- **Pipeline II:** requests 1-4 are blocked at achieved II=5 by the reported "
        "memory dependency. Requests above 5 are realized and progressively increase "
        "latency. The flat lower range is therefore compiler evidence, not an inactive knob.",
        "- **K strip size:** size 4 crosses a scheduler/parallelism threshold and sharply "
        "reduces cycles while increasing DSP use. Larger values return to the parent-like "
        "schedule.",
        "- **Ping-pong depth:** depth 2 reaches II=1 and wins; the remaining values expose "
        "a non-monotonic curve rather than a presumed linear relationship.",
        "- **AXI width:** latency is effectively flat, but width 64 materially reduces "
        "BRAM, LUT, and FF. This control affects implementation cost more than this "
        "kernel's compute-bound latency.",
        "- **Multiplier implementation:** `fabric`, `meddsp`, and `fulldsp` are realized "
        "with distinct operator/resource mappings and the same improved latency. "
        "`primitivedsp` passes CSim but is unsupported for this U280 CSynth target and "
        "remains invalid with no imputed cycles.",
        "",
        "## Interpretation boundary",
        "",
        "Each row is a one-factor-at-a-time measurement against that step's own frozen "
        "GEMM parent. The parents represent different valid source structures, so their "
        "absolute cycles must not be treated as one sequential optimization trajectory. "
        "The winning parameter values are also not assumed to compose safely; a combined "
        "candidate requires a fresh CSim/CSynth validation and resource/timing gates.",
        "",
        "The direct scalar-reduction lane family and the original compiler-forced "
        "complete-unroll kernel are retained in `gemm_parameter_rollout_measurements.*` "
        "as supplemental evidence. Only the striped independent-accumulator family is "
        "used for the primary parallel-lane OFAT comparison.",
        "",
        "## Inputs",
        "",
        f"- Lane measurements: `{inputs['lane'].resolve()}`",
        f"- Per-step measurements: `{inputs['steps'].resolve()}`",
        f"- Per-step summary: `{inputs['summary'].resolve()}`",
    ])
    return "\n".join(lines) + "\n"


def _verify_step_winners(
    summaries: list[dict[str, Any]], existing_summary: dict[str, Any]
) -> None:
    expected = {
        str(case["case_id"]): case for case in (existing_summary.get("cases") or [])
    }
    for summary in summaries:
        if summary["step_id"] == "parallel_lanes":
            continue
        case_id = str(STEP_SPECS[str(summary["step_id"])]["case_id"])
        prior = expected.get(case_id)
        if prior is None:
            raise ValueError(f"step summary is missing case {case_id}")
        if str(prior.get("winner_candidate_id")) != str(summary["best_candidate_id"]):
            raise ValueError(f"winner mismatch for {case_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-measurements", type=Path, required=True)
    parser.add_argument("--step-measurements", type=Path, required=True)
    parser.add_argument("--step-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lane_records = json.loads(args.lane_measurements.read_text())
    step_records = json.loads(args.step_measurements.read_text())
    existing_summary = json.loads(args.step_summary.read_text())
    if not isinstance(lane_records, list) or not isinstance(step_records, list):
        raise ValueError("rollout measurement inputs must be JSON arrays")

    rows = [
        *_normalize_lane_rows(lane_records),
        *_normalize_step_rows(step_records),
    ]
    summaries = _step_summaries(rows)
    _verify_step_winners(summaries, existing_summary)
    _write_rows(args.output_dir, "gemm_parameter_rollout_measurements", rows)
    _write_rows(args.output_dir, "gemm_parameter_rollout_best_by_step", summaries)

    inputs = {
        "lane": args.lane_measurements,
        "steps": args.step_measurements,
        "summary": args.step_summary,
    }
    (args.output_dir / "report.md").write_text(_report(rows, summaries, inputs))
    _plot_parameter_curves(rows, args.output_dir)
    _plot_parent_best(summaries, args.output_dir)
    _plot_winner_resources(rows, summaries, args.output_dir)

    manifest = {
        "schema_version": "c2hls.gemm-parameter-rollout-audit.v1",
        "benchmark": "hlsfactory_gemm",
        "reference_blind": True,
        "cosim": False,
        "model_calls": 0,
        "toolchain": {
            "vitis_version": "2023.2",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
        },
        "input_artifacts": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "primary_step_count": len(summaries),
        "primary_measurement_count": sum(
            row.get("included_in_step_comparison") is True for row in rows
        ),
        "all_measurement_count": len(rows),
        "output_artifacts": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "sha256": _sha256(args.output_dir / name),
            }
            for name in (
                "report.md",
                "gemm_parameter_rollout_measurements.csv",
                "gemm_parameter_rollout_measurements.json",
                "gemm_parameter_rollout_measurements.jsonl",
                "gemm_parameter_rollout_best_by_step.csv",
                "gemm_parameter_rollout_best_by_step.json",
                "gemm_parameter_rollout_best_by_step.jsonl",
                "gemm_parameter_effect_curves.png",
                "gemm_parameter_effect_curves.pdf",
                "gemm_parent_vs_best_cycles.png",
                "gemm_parent_vs_best_cycles.pdf",
                "gemm_winner_resource_ratios.png",
                "gemm_winner_resource_ratios.pdf",
            )
        },
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
