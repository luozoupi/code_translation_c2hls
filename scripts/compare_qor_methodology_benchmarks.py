#!/usr/bin/env python3
"""Compare per-step QoR-control behavior across GEMM and 2MM."""

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


STEP_ORDER = (
    "parallel_lanes",
    "pipeline_ii",
    "tiling",
    "alternating_buffer_depth",
    "axi_width",
    "multiplier_implementation",
)
STEP_LABELS = {
    "parallel_lanes": "Parallel lanes",
    "pipeline_ii": "Pipeline II",
    "tiling": "Reduction tiling",
    "alternating_buffer_depth": "Alternating buffer depth",
    "axi_width": "AXI width",
    "multiplier_implementation": "Multiplier implementation",
}
CASE_MAP = {
    "gemm": {
        "parallel_lanes": "gemm_striped_compute_lanes_expanded",
        "pipeline_ii": "gemm_pipeline_compute_ii",
        "tiling": "gemm_tiling_k_strip",
        "alternating_buffer_depth": "gemm_doublebuffer_stream_depth",
        "axi_width": "gemm_coalescing_axi_width",
        "multiplier_implementation": "gemm_resource_mul_limit",
    },
    "2mm": {
        "parallel_lanes": "2mm_linked_compute_lanes",
        "pipeline_ii": "2mm_pipeline_compute_ii",
        "tiling": "2mm_tiling_k_strip",
        "alternating_buffer_depth": "2mm_doublebuffer_b_tile_depth",
        "axi_width": "2mm_coalescing_axi_width",
        "multiplier_implementation": "2mm_resource_mul_impl",
    },
}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _case_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("case_id")): case
        for case in (summary.get("cases") or [])
        if isinstance(case, dict)
    }


def _gemm_lane_case(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the GEMM lane case from the consolidated, full-domain rollout."""
    rows = [
        row for row in measurements
        if row.get("step_id") == "parallel_lanes"
        and row.get("included_in_step_comparison") is True
    ]
    parents = [row for row in rows if row.get("is_parent") is True]
    if len(parents) != 1:
        raise ValueError(
            "consolidated GEMM lane measurements must contain exactly one parent"
        )
    parent = parents[0]
    feasible = [
        row for row in rows
        if row.get("feasible") is True
        and _number(row.get("latency_cycles_worst")) is not None
    ]
    if not feasible:
        raise ValueError("consolidated GEMM lane measurements have no feasible point")

    def selection_key(row: dict[str, Any]) -> tuple[float, float, str]:
        resources = [_number(row.get(key)) for key in ("dsp", "bram", "lut", "ff", "uram")]
        resource_sum = (
            sum(float(value) for value in resources)
            if all(value is not None for value in resources)
            else float("inf")
        )
        return (
            float(_number(row.get("latency_cycles_worst")) or float("inf")),
            resource_sum,
            str(row.get("candidate_id") or ""),
        )

    best = min(feasible, key=selection_key)
    ordered = sorted(
        feasible,
        key=lambda row: float(_number(row.get("parameter_numeric")) or 0.0),
    )
    cycles = [float(row["latency_cycles_worst"]) for row in ordered]
    violations = sum(
        1 for previous, current in zip(cycles, cycles[1:])
        if current > previous
    )
    return {
        "case_id": CASE_MAP["gemm"]["parallel_lanes"],
        "case_label": "GEMM: striped independent-accumulator lanes",
        "knob_kind": "compute_lanes",
        "parent_value": parent.get("parameter_label"),
        "best_parameter_value": best.get("parameter_label"),
        "parent_cycles": _number(parent.get("latency_cycles_worst")),
        "best_cycles": _number(best.get("latency_cycles_worst")),
        "candidate_count": len([row for row in rows if row.get("is_parent") is not True]),
        "feasible_candidate_count": len(
            [row for row in feasible if row.get("is_parent") is not True]
        ),
        "monotonicity_violations": violations,
        "winner_candidate_id": best.get("candidate_id"),
        "measurements": rows,
    }


def _comparison_rows(
    gemm_summary: dict[str, Any],
    two_mm_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries = {"gemm": gemm_summary, "2mm": two_mm_summary}
    indexes = {name: _case_index(payload) for name, payload in summaries.items()}
    rows = []
    for step_id in STEP_ORDER:
        row: dict[str, Any] = {
            "schema_version": "c2hls.qor-methodology-comparison.v1",
            "step_id": step_id,
            "step_label": STEP_LABELS[step_id],
        }
        for benchmark in ("gemm", "2mm"):
            case_id = CASE_MAP[benchmark][step_id]
            case = indexes[benchmark].get(case_id)
            if case is None:
                raise ValueError(f"{benchmark} summary is missing {case_id}")
            parent_cycles = _number(case.get("parent_cycles"))
            best_cycles = _number(case.get("best_cycles"))
            ratio = (
                best_cycles / parent_cycles
                if best_cycles is not None and parent_cycles not in (None, 0)
                else None
            )
            prefix = f"{benchmark}_"
            row.update({
                prefix + "case_id": case_id,
                prefix + "parent_value": case.get("parent_value"),
                prefix + "best_value": case.get("best_parameter_value"),
                prefix + "parent_cycles": parent_cycles,
                prefix + "best_cycles": best_cycles,
                prefix + "best_cycle_ratio": ratio,
                prefix + "speedup_vs_parent": (
                    1.0 / ratio if ratio not in (None, 0) else None
                ),
                prefix + "candidate_count": case.get("candidate_count"),
                prefix + "feasible_candidate_count": case.get(
                    "feasible_candidate_count"
                ),
                prefix + "monotonicity_violations": case.get(
                    "monotonicity_violations"
                ),
                prefix + "winner_candidate_id": case.get("winner_candidate_id"),
                prefix + "parent_retained": (
                    case.get("winner_candidate_id") == "frozen_parent"
                ),
            })
        rows.append(row)
    return rows


def _case_measurement(
    summary: dict[str, Any], case_id: str, parameter_label: str
) -> dict[str, Any] | None:
    case = _case_index(summary).get(case_id) or {}
    return next(
        (
            row
            for row in (case.get("measurements") or [])
            if str(row.get("parameter_label")) == parameter_label
        ),
        None,
    )


def _write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    stem = "gemm_vs_2mm_methodology"
    (output_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output_dir / f"{stem}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    positions = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 5.8))
    for offset, benchmark, label, color in (
        (-width / 2, "gemm", "GEMM", "#0072B2"),
        (width / 2, "2mm", "2MM", "#D55E00"),
    ):
        ratios = [float(row[f"{benchmark}_best_cycle_ratio"]) for row in rows]
        axis.bar(positions + offset, ratios, width, color=color, label=label)
        for index, ratio in enumerate(ratios):
            axis.text(
                positions[index] + offset,
                ratio * (1.12 if ratio < 0.8 else 1.025),
                f"{1.0 / ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if ratio > 0.9 else 0,
            )
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_ylim(0.015, 1.35)
    axis.set_xticks(
        positions,
        labels=[row["step_label"] for row in rows],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel("best CSynth cycles / frozen-parent cycles")
    axis.set_title("Per-step QoR response on two matrix kernels")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"gemm_vs_2mm_best_cycle_ratio.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def _format_cycles(value: Any) -> str:
    numeric = _number(value)
    return f"{int(numeric):,}" if numeric is not None else "N/A"


def _report(
    rows: list[dict[str, Any]],
    gemm_summary: dict[str, Any],
    two_mm_summary: dict[str, Any],
    inputs: dict[str, Path],
) -> str:
    gemm_pipeline = _case_measurement(
        gemm_summary, CASE_MAP["gemm"]["pipeline_ii"], "5"
    ) or {}
    two_mm_pipeline = _case_measurement(
        two_mm_summary, CASE_MAP["2mm"]["pipeline_ii"], "5"
    ) or {}
    primitive_statuses = []
    for benchmark, summary in (("GEMM", gemm_summary), ("2MM", two_mm_summary)):
        primitive = _case_measurement(
            summary,
            CASE_MAP[benchmark.lower()]["multiplier_implementation"],
            "primitivedsp",
        ) or {}
        primitive_statuses.append(
            f"{benchmark}={primitive.get('compiler_control_status', 'unknown')}"
        )
    lines = [
        "# QoR Methodology Validation: GEMM versus 2MM",
        "",
        "Both campaigns use Vitis 2023.2, `xcu280-fsvh2892-2L-e`, 3.33 ns, "
        "reference-blind source mutation, CSim-before-CSynth validation, no model "
        "calls, and no COSIM. Every ratio is local to that control family's frozen "
        "parent; absolute parent cycles are not compared across arms.",
        "",
        "## Aligned results",
        "",
        "| Control family | GEMM parent -> best | GEMM cycles | GEMM speedup | 2MM parent -> best | 2MM cycles | 2MM speedup |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step_label']} | {row['gemm_parent_value']} -> "
            f"{row['gemm_best_value']} | {_format_cycles(row['gemm_parent_cycles'])} "
            f"-> {_format_cycles(row['gemm_best_cycles'])} | "
            f"{float(row['gemm_speedup_vs_parent']):.3f}x | "
            f"{row['2mm_parent_value']} -> {row['2mm_best_value']} | "
            f"{_format_cycles(row['2mm_parent_cycles'])} -> "
            f"{_format_cycles(row['2mm_best_cycles'])} | "
            f"{float(row['2mm_speedup_vs_parent']):.3f}x |"
        )
    lines.extend([
        "",
        "## Replicated behavior",
        "",
        "- **Lane count is active and non-monotonic on both kernels.** GEMM and 2MM "
        "both have monotonicity violations, and 2MM has an interior optimum at 10 "
        "lanes. More parallelism therefore cannot replace measured selection.",
        "- **Both pipeline controls hit an achieved-II floor of 5.** Requests below "
        f"the floor do not reduce latency; at requested II=5 the measured maximum "
        f"II is {gemm_pipeline.get('achieved_pipeline_ii_max', 'N/A')} for GEMM and "
        f"{two_mm_pipeline.get('achieved_pipeline_ii_max', 'N/A')} for 2MM. Requests "
        "above the floor increase cycles.",
        "- **AXI width is primarily a resource control in both experiments.** The "
        "selected low-width points change latency by only a few cycles while reducing "
        "BRAM/LUT/FF substantially.",
        "- **Multiplier implementation is compiler-realized in both experiments.** "
        "Fabric, medium-DSP, and full-DSP mappings synthesize with distinct resources; "
        f"the primitive-DSP option remains invalid ({', '.join(primitive_statuses)}).",
        "",
        "## Benchmark-specific behavior",
        "",
        "- GEMM's K-strip control crosses a large scheduler threshold; 2MM's phase-1 "
        "strip sizes 2-70 are latency-flat and differ mainly in small resource changes.",
        "- Alternating-buffer depth has a strong optimum on both benchmarks, but the "
        "winning depths differ. Parameter values are not transferable even when the "
        "control family is.",
        "",
        "## Conclusion",
        "",
        "The second benchmark supports the methodology: exact coupled controls, "
        "compiler evidence, and CSim/CSynth tournament selection expose real and "
        "repeatable QoR structure. It does not support a universal optimization recipe. "
        "The useful invariant is to measure bounded candidates and retain the best "
        "feasible point; the winning value remains benchmark-specific.",
        "",
        "## Inputs",
        "",
        f"- GEMM step-control summary: `{inputs['gemm_steps'].resolve()}`",
        f"- GEMM consolidated lane measurements: `{inputs['gemm_lanes'].resolve()}`",
        f"- 2MM summary: `{inputs['2mm'].resolve()}`",
    ])
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-summary", type=Path, required=True)
    parser.add_argument("--gemm-lane-measurements", type=Path, required=True)
    parser.add_argument("--two-mm-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gemm_summary = dict(json.loads(args.gemm_summary.read_text()))
    gemm_summary["cases"] = [
        _gemm_lane_case(json.loads(args.gemm_lane_measurements.read_text())),
        *(gemm_summary.get("cases") or []),
    ]
    two_mm_summary = json.loads(args.two_mm_summary.read_text())
    rows = _comparison_rows(gemm_summary, two_mm_summary)
    _write_rows(args.output_dir, rows)
    _plot(rows, args.output_dir)
    inputs = {
        "gemm_steps": args.gemm_summary,
        "gemm_lanes": args.gemm_lane_measurements,
        "2mm": args.two_mm_summary,
    }
    (args.output_dir / "methodology_validation_report.md").write_text(
        _report(rows, gemm_summary, two_mm_summary, inputs)
    )
    outputs = [
        path for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "schema_version": "c2hls.qor-methodology-comparison-manifest.v1",
        "reference_blind": True,
        "cosim": False,
        "toolchain": {
            "vitis_version": "2023.2",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "outputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(args.output_dir / "methodology_validation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
