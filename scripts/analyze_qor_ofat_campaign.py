#!/usr/bin/env python3
"""Aggregate a QoR OFAT campaign and render parameter-effect visuals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

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
CASE_LABELS = {
    "atax_pipeline_ii": "ATAX: pipeline II",
    "bicg_unroll_factor": "BiCG: unroll factor",
    "gemver_tile_size": "GEMVER: tile size",
    "gemm_partition_factor": "GEMM: partition factor",
    "2mm_axi_widen": "2MM: AXI width cap",
    "trisolv_dataflow_ablation": "TRISOLV: DATAFLOW",
}
COLORS = {
    "latency_cycles_worst": "#0072B2",
    "interval": "#D55E00",
    "estimated_clock_period_ns": "#009E73",
    "achieved_pipeline_ii_max": "#CC79A7",
    "dsp": "#0072B2",
    "bram": "#E69F00",
    "lut": "#009E73",
    "ff": "#D55E00",
    "uram": "#CC79A7",
}
LINE_STYLES = {
    "latency_cycles_worst": "-",
    "interval": "--",
    "estimated_clock_period_ns": "-",
    "achieved_pipeline_ii_max": ":",
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parameter_label(value: Any, *, kind: str) -> str:
    if kind.endswith("_enabled"):
        return "enabled" if value in (1, "enabled") else "disabled"
    if value is None:
        return "unknown"
    return str(value)


def _row_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    return {key: _number(metrics.get(key)) for key in METRIC_KEYS}


def _measurement_rows(case_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(case_path.read_text())
    sweep = payload.get("design_sweep") or {}
    knob = (sweep.get("discovered_knobs") or [{}])[0]
    case_id = case_path.stem
    parent = sweep.get("parent") or {}
    parent_value = (
        knob.get("current_value")
        if knob.get("current_value") is not None
        else knob.get("current_label")
    )
    base = {
        "schema_version": "c2hls.qor-ofat-measurement.v1",
        "case_id": case_id,
        "case_label": CASE_LABELS.get(case_id, case_id),
        "benchmark": payload.get("benchmark"),
        "knob_id": knob.get("knob_id"),
        "knob_kind": knob.get("kind"),
        "knob_name": knob.get("name"),
        "source_result_sha256": (payload.get("source_result") or {}).get("sha256"),
    }
    rows = [{
        **base,
        "candidate_id": "frozen_parent",
        "is_parent": True,
        "parameter_value": parent_value,
        "parameter_label": _parameter_label(parent_value, kind=str(knob.get("kind"))),
        "status": parent.get("status"),
        "feasible": parent.get("feasible") is True,
        "csim_status": (parent.get("csim") or {}).get("status"),
        "pareto_frontier": parent.get("pareto_frontier") is True,
        **_row_metrics(parent),
    }]
    for candidate in sweep.get("candidates") or []:
        changed = (candidate.get("changed_knobs") or [{}])[0]
        value = changed.get("to")
        rows.append({
            **base,
            "candidate_id": candidate.get("candidate_id"),
            "is_parent": False,
            "parameter_value": value,
            "parameter_label": _parameter_label(value, kind=str(knob.get("kind"))),
            "status": candidate.get("status"),
            "feasible": candidate.get("feasible") is True,
            "csim_status": (candidate.get("csim") or {}).get("status"),
            "pareto_frontier": candidate.get("pareto_frontier") is True,
            **_row_metrics(candidate),
        })

    parent_row = rows[0]
    for row in rows:
        for metric in METRIC_KEYS:
            parent_metric = parent_row.get(metric)
            value = row.get(metric)
            row[f"{metric}_ratio"] = (
                value / parent_metric
                if value is not None and parent_metric not in (None, 0)
                else None
            )
            row[f"{metric}_delta"] = (
                value - parent_metric
                if value is not None and parent_metric is not None
                else None
            )
        ratios = [
            row.get(f"{resource}_ratio")
            for resource in RESOURCE_KEYS
            if parent_row.get(resource) not in (None, 0)
        ]
        valid_ratios = [value for value in ratios if value is not None]
        row["mean_resource_ratio"] = (
            sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
        )

    numeric_parent = _number(parent_value)
    if numeric_parent is not None and not str(knob.get("kind")).endswith("_enabled"):
        rows.sort(key=lambda row: (_number(row["parameter_value"]) or float("inf")))
    else:
        rows.sort(key=lambda row: (not row["is_parent"], _number(row["parameter_value"]) or 0))
    for index, row in enumerate(rows):
        row["parameter_order"] = index
    return payload, rows


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _axes(case_count: int) -> tuple[plt.Figure, np.ndarray]:
    columns = 3
    rows = math.ceil(case_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.2 * rows), squeeze=False)
    return fig, axes.flatten()


def _curve_plot(
    grouped: list[tuple[str, list[dict[str, Any]]]],
    output_dir: Path,
    *,
    metrics: tuple[str, ...],
    ratios: bool,
    name: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    fig, axes = _axes(len(grouped))
    for axis, (case_id, rows) in zip(axes, grouped):
        x = [row["parameter_order"] for row in rows]
        labels = [row["parameter_label"] for row in rows]
        for metric in metrics:
            key = f"{metric}_ratio" if ratios else metric
            points = [(row["parameter_order"], row.get(key)) for row in rows]
            valid = [(px, py) for px, py in points if py is not None]
            if not valid:
                continue
            axis.plot(
                [item[0] for item in valid],
                [item[1] for item in valid],
                marker="o",
                linewidth=1.8,
                linestyle=LINE_STYLES.get(metric, "-"),
                color=COLORS[metric],
                label=metric.replace("_", " "),
            )
        parent = next(row for row in rows if row["is_parent"])
        parent_y = parent.get(
            f"{metrics[0]}_ratio" if ratios else metrics[0]
        )
        if parent_y is not None:
            axis.scatter(
                [parent["parameter_order"]], [parent_y],
                s=85, facecolors="white", edgecolors="#111111", linewidths=1.5,
                zorder=5, label="frozen parent",
            )
        for row in rows:
            if row["is_parent"] or row["feasible"]:
                continue
            fallback = parent_y if parent_y is not None else 0
            axis.scatter(
                [row["parameter_order"]], [fallback], marker="x", s=65,
                color="#B2182B", linewidths=2, zorder=6,
            )
        axis.set_title(CASE_LABELS.get(case_id, case_id), fontsize=11)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_xlabel("parameter value")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        if ratios:
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
        if log_y:
            axis.set_yscale("log")
        axis.legend(fontsize=7, frameon=False)
    for axis in axes[len(grouped):]:
        axis.set_visible(False)
    fig.suptitle(ylabel, fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output_dir, name)


def _heatmap(rows: list[dict[str, Any]], output_dir: Path) -> None:
    candidates = [row for row in rows if not row["is_parent"]]
    metrics = (
        "latency_cycles_worst",
        "interval",
        "estimated_clock_period_ns",
        "dsp",
        "bram",
        "lut",
        "ff",
        "uram",
    )
    values = np.full((len(candidates), len(metrics)), np.nan)
    for row_index, row in enumerate(candidates):
        for column_index, metric in enumerate(metrics):
            ratio = row.get(f"{metric}_ratio")
            if ratio is not None:
                values[row_index, column_index] = 100.0 * (ratio - 1.0)
    fig, axis = plt.subplots(figsize=(12, max(6, 0.38 * len(candidates))))
    masked = np.ma.masked_invalid(values)
    color_map = matplotlib.colormaps["RdBu_r"].copy()
    color_map.set_bad("#BDBDBD")
    image = axis.imshow(masked, aspect="auto", cmap=color_map, vmin=-50, vmax=50)
    labels = [
        f"{CASE_LABELS.get(row['case_id'], row['case_id'])} | {row['parameter_label']}"
        + ("" if row["feasible"] else " | invalid")
        for row in candidates
    ]
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.set_xticks(
        range(len(metrics)),
        [metric.replace("latency_cycles_worst", "cycles").replace("_", " ") for metric in metrics],
        rotation=25,
        ha="right",
    )
    axis.set_title("Metric change from frozen parent (%) | gray = unavailable")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("change (%)")
    fig.tight_layout()
    _save_figure(fig, output_dir, "metric_change_heatmap")


def _tradeoff(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))
    case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    for color, case_id in zip(palette, case_ids):
        points = [
            row for row in rows
            if row["case_id"] == case_id
            and row["feasible"]
            and row.get("latency_cycles_worst_ratio") is not None
            and row.get("mean_resource_ratio") is not None
        ]
        axis.scatter(
            [row["mean_resource_ratio"] for row in points],
            [row["latency_cycles_worst_ratio"] for row in points],
            s=[95 if row["is_parent"] else 55 for row in points],
            facecolors=["white" if row["is_parent"] else color for row in points],
            edgecolors=color,
            linewidths=1.5,
            label=CASE_LABELS.get(case_id, case_id),
        )
        candidates = [row for row in points if not row["is_parent"]]
        changed_candidates = [
            row for row in candidates
            if abs(row["latency_cycles_worst_ratio"] - 1.0) > 1e-9
            or abs(row["mean_resource_ratio"] - 1.0) > 1e-9
        ]
        annotate_ids = set()
        if changed_candidates:
            annotate_ids.add(min(
                changed_candidates,
                key=lambda row: row["latency_cycles_worst_ratio"],
            )["candidate_id"])
            annotate_ids.add(min(
                changed_candidates,
                key=lambda row: row["mean_resource_ratio"],
            )["candidate_id"])
        base_offset = {
            "2mm_axi_widen": (-5, -16),
            "atax_pipeline_ii": (7, 8),
            "gemm_partition_factor": (7, -14),
            "gemver_tile_size": (7, 8),
        }.get(case_id, (7, 7))
        offsets = (
            base_offset,
            (base_offset[0], base_offset[1] - 14),
        )
        for annotation_index, row in enumerate(
            item for item in candidates if item["candidate_id"] in annotate_ids
        ):
            axis.annotate(
                row["parameter_label"],
                (row["mean_resource_ratio"], row["latency_cycles_worst_ratio"]),
                xytext=offsets[annotation_index % len(offsets)],
                textcoords="offset points", fontsize=8,
            )
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("mean resource ratio vs parent")
    axis.set_ylabel("worst-cycle ratio vs parent")
    axis.set_title("Latency-resource tradeoff (lower-left is better)")
    axis.grid(color="#E1E1E1", linewidth=0.7)
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "latency_resource_tradeoff")


def _validity(grouped: list[tuple[str, list[dict[str, Any]]]], output_dir: Path) -> None:
    candidates = [(case_id, row) for case_id, rows in grouped for row in rows if not row["is_parent"]]
    fig, axis = plt.subplots(figsize=(12, max(4, 0.35 * len(candidates))))
    values = np.array([[1 if row["feasible"] else 0] for _, row in candidates])
    axis.imshow(values, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#B2182B", "#1B7837"]), vmin=0, vmax=1)
    axis.set_yticks(
        range(len(candidates)),
        [
            f"{CASE_LABELS.get(case_id, case_id)} | {row['parameter_label']} | {row['status']}"
            for case_id, row in candidates
        ],
        fontsize=8,
    )
    axis.set_xticks([0], ["CSim + CSynth + timing + resource fit"])
    axis.set_title("Candidate validity matrix")
    fig.tight_layout()
    _save_figure(fig, output_dir, "candidate_validity_matrix")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
        })
    return {
        "schema_version": "c2hls.qor-ofat-artifact-manifest.v1",
        "artifacts": artifacts,
    }


def _summary(
    payloads: dict[str, dict[str, Any]],
    grouped: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    cases = []
    for case_id, rows in grouped:
        payload = payloads[case_id]
        sweep = payload["design_sweep"]
        parent = next(row for row in rows if row["is_parent"])
        feasible = [row for row in rows if row["feasible"]]
        best = min(feasible, key=_selection_key)
        trend = (sweep.get("knob_trends") or [{}])[0]
        cases.append({
            "case_id": case_id,
            "case_label": CASE_LABELS.get(case_id, case_id),
            "benchmark": payload.get("benchmark"),
            "knob_kind": parent["knob_kind"],
            "knob_name": parent["knob_name"],
            "parent_value": parent["parameter_label"],
            "parent_cycles": parent["latency_cycles_worst"],
            "candidate_count": len(rows) - 1,
            "feasible_candidate_count": sum(
                row["feasible"] for row in rows if not row["is_parent"]
            ),
            "best_parameter_value": best["parameter_label"],
            "best_cycles": best["latency_cycles_worst"],
            "best_cycle_ratio": best["latency_cycles_worst_ratio"],
            "winner_candidate_id": sweep.get("winner_candidate_id"),
            "applied": sweep.get("applied") is True,
            "spearman_value_vs_cycles": trend.get("spearman_value_vs_worst_cycles"),
            "monotonicity_violations": trend.get("monotonicity_violations"),
            "winner_explanation": sweep.get("winner_explanation"),
        })
    toolchains = [payload.get("toolchain") or {} for payload in payloads.values()]
    toolchain_fingerprints = {
        json.dumps(toolchain, sort_keys=True) for toolchain in toolchains
    }
    cycle_improvements = [
        case for case in cases
        if case.get("best_cycle_ratio") is not None
        and case["best_cycle_ratio"] < 1.0 - 1e-12
    ]
    resource_ties = [
        case for case in cases
        if case.get("applied") is True
        and case.get("best_cycle_ratio") is not None
        and abs(case["best_cycle_ratio"] - 1.0) <= 1e-12
    ]
    parent_retained = [case for case in cases if case.get("applied") is not True]
    return {
        "schema_version": "c2hls.qor-ofat-analysis.v1",
        "reference_blind": all(
            payload.get("reference_blind") is True for payload in payloads.values()
        ),
        "model_calls": sum(
            int(payload.get("model_calls") or 0) for payload in payloads.values()
        ),
        "cosim": False,
        "toolchain": toolchains[0] if toolchains else {},
        "toolchain_consistent": len(toolchain_fingerprints) == 1,
        "case_count": len(cases),
        "candidate_count": sum(case["candidate_count"] for case in cases),
        "feasible_candidate_count": sum(
            case["feasible_candidate_count"] for case in cases
        ),
        "measurement_count_including_parents": sum(len(rows) for _, rows in grouped),
        "cycle_improvement_case_count": len(cycle_improvements),
        "resource_tie_improvement_case_count": len(resource_ties),
        "parent_retained_case_count": len(parent_retained),
        "cases": cases,
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, str]:
    cycles = row.get("latency_cycles_worst")
    resources = [row.get(key) for key in RESOURCE_KEYS]
    resource_sum = (
        sum(resources)
        if all(value is not None for value in resources)
        else float("inf")
    )
    return (
        cycles if cycles is not None else float("inf"),
        resource_sum,
        str(row.get("candidate_id") or ""),
    )


def _report(summary: dict[str, Any]) -> str:
    toolchain = summary.get("toolchain") or {}
    lines = [
        "# Multi-Benchmark QoR Parameter Effects",
        "",
        f"Toolchain: **Vitis {toolchain.get('vitis_version', 'unknown')}**, "
        f"**{toolchain.get('part', 'unknown')}**, "
        f"**{toolchain.get('clock_ns', 'unknown')} ns**  ",
        f"Reference-blind: **{summary.get('reference_blind')}** | "
        f"Model calls: **{summary.get('model_calls')}** | "
        f"COSIM: **{summary.get('cosim')}**  ",
        f"Cases: **{summary['case_count']}**  ",
        f"Candidates: **{summary['candidate_count']}**  ",
        f"Feasible candidates: **{summary['feasible_candidate_count']}**  ",
        f"Measurements including parents: **{summary['measurement_count_including_parents']}**",
        "",
        "| Case | Control | Parent | Best tested | Parent cycles | Best cycles | Change | Valid | Monotonic violations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        ratio = case.get("best_cycle_ratio")
        change = f"{100.0 * (ratio - 1.0):+.2f}%" if ratio is not None else "N/A"
        lines.append(
            f"| {case['case_label']} | {case['knob_kind']} | "
            f"{case['parent_value']} | {case['best_parameter_value']} | "
            f"{case['parent_cycles']:.0f} | {case['best_cycles']:.0f} | {change} | "
            f"{case['feasible_candidate_count']}/{case['candidate_count']} | "
            f"{case.get('monotonicity_violations')} |"
        )
    improved = [
        case for case in summary["cases"]
        if case.get("best_cycle_ratio") is not None
        and case["best_cycle_ratio"] < 1.0 - 1e-12
    ]
    resource_ties = [
        case for case in summary["cases"]
        if case.get("applied") is True
        and case.get("best_cycle_ratio") is not None
        and abs(case["best_cycle_ratio"] - 1.0) <= 1e-12
    ]
    retained = [
        case["case_label"] for case in summary["cases"]
        if case.get("applied") is not True
    ]
    lines.extend(["", "## Findings", ""])
    lines.append(
        f"- {len(improved)}/{summary['case_count']} controls produced a lower-cycle "
        "candidate."
    )
    if improved:
        largest = min(improved, key=lambda case: case["best_cycle_ratio"])
        reduction = 100.0 * (1.0 - largest["best_cycle_ratio"])
        lines.append(
            f"- The largest latency response was {largest['case_label']}: "
            f"{largest['parent_value']} to {largest['best_parameter_value']} reduced "
            f"worst-case cycles by {reduction:.2f}%."
        )
    if resource_ties:
        lines.append(
            "- Exact-latency resource tie-break improvements were selected for: "
            + ", ".join(case["case_label"] for case in resource_ties)
            + "."
        )
    if retained:
        lines.append(
            "- The frozen parent remained best for: " + ", ".join(retained) + "."
        )
    lines.extend([
        "",
        "All candidates start from each case's frozen parent. Reference metrics are not "
        "available to generation, mutation, or winner selection. CSim runs before "
        "CSynth; invalid candidates retain no imputed cycles. COSIM is not run.",
        "",
        "The mean-resource tradeoff view is descriptive only. Winner selection uses "
        "minimum feasible worst-case cycles with the framework's deterministic "
        "resource tie-break, not the plotted equal-weight resource mean.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict[str, Any]] = {}
    grouped = []
    all_rows = []
    for case_path in sorted((args.campaign_dir / "cases").glob("*.json")):
        payload, rows = _measurement_rows(case_path)
        payloads[case_path.stem] = payload
        grouped.append((case_path.stem, rows))
        all_rows.extend(rows)
    grouped.sort(key=lambda item: list(CASE_LABELS).index(item[0]) if item[0] in CASE_LABELS else 999)
    if not grouped:
        raise RuntimeError("no completed campaign case JSON files found")

    _write_csv(args.output_dir / "measurements.csv", all_rows)
    _write_jsonl(args.output_dir / "measurements.jsonl", all_rows)
    (args.output_dir / "measurements.json").write_text(
        json.dumps(all_rows, indent=2) + "\n"
    )
    summary = _summary(payloads, grouped)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(_report(summary))

    _curve_plot(
        grouped, args.output_dir,
        metrics=("latency_cycles_worst",), ratios=False,
        name="absolute_cycles_by_parameter", ylabel="worst-case CSynth cycles",
        log_y=True,
    )
    _curve_plot(
        grouped, args.output_dir,
        metrics=("latency_cycles_worst", "interval"), ratios=True,
        name="normalized_latency_by_parameter", ylabel="ratio to frozen parent",
    )
    _curve_plot(
        grouped, args.output_dir,
        metrics=("estimated_clock_period_ns", "achieved_pipeline_ii_max"), ratios=True,
        name="timing_ii_by_parameter", ylabel="ratio to frozen parent",
    )
    _curve_plot(
        grouped, args.output_dir,
        metrics=RESOURCE_KEYS, ratios=True,
        name="resources_by_parameter", ylabel="resource ratio to frozen parent",
    )
    _heatmap(all_rows, args.output_dir)
    _tradeoff(all_rows, args.output_dir)
    _validity(grouped, args.output_dir)
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(_artifact_manifest(args.output_dir), indent=2) + "\n"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "case_count": summary["case_count"],
        "candidate_count": summary["candidate_count"],
        "feasible_candidate_count": summary["feasible_candidate_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
