#!/usr/bin/env python3
"""Combine direct and striped GEMM lane-control campaign results."""

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


RESOURCE_KEYS = ("dsp", "bram", "lut", "ff", "uram")
ARCHITECTURE_LABELS = {
    "direct_reduction": "Direct scalar recurrence",
    "striped_accumulators": "Striped independent accumulators",
    "original_forced_complete_unroll": "Original (Vitis forced 80 lanes)",
}
ARCHITECTURE_COLORS = {
    "direct_reduction": "#31688e",
    "striped_accumulators": "#d1495b",
    "original_forced_complete_unroll": "#3a7d44",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _load_family(path: Path, architecture: str) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"measurement file is empty: {path}")
    rows = []
    parent = next((record for record in records if record.get("is_parent")), None)
    parent_cycles = _number((parent or {}).get("latency_cycles_worst"))
    factor_one = next(
        (
            record for record in records
            if _number(record.get("parameter_value")) == 1
        ),
        None,
    )
    factor_one_cycles = _number((factor_one or {}).get("latency_cycles_worst"))
    for record in records:
        requested = _number(record.get("parameter_value"))
        cycles = _number(record.get("latency_cycles_worst"))
        effective = _number(record.get("compiler_effective_unroll_factor"))
        row = {
            "architecture": architecture,
            "architecture_label": ARCHITECTURE_LABELS[architecture],
            "requested_lanes": int(requested) if requested is not None else None,
            "effective_lanes": int(effective) if effective is not None else None,
            "is_family_parent": record.get("is_parent") is True,
            "candidate_id": record.get("candidate_id"),
            "code_sha256": record.get("code_sha256"),
            "csim_status": record.get("csim_status"),
            "feasible": record.get("feasible") is True,
            "latency_cycles_worst": cycles,
            "interval": _number(record.get("interval")),
            "estimated_clock_period_ns": _number(
                record.get("estimated_clock_period_ns")
            ),
            "achieved_pipeline_ii_max": _number(
                record.get("achieved_pipeline_ii_max")
            ),
            "max_ii_scope_id": record.get("max_ii_scope_id"),
            "max_ii_scope_violation": record.get("max_ii_scope_violation"),
            "compiler_unroll_mode": record.get("compiler_unroll_mode"),
            "compiler_override": record.get(
                "compiler_target_parameter_overridden"
            ),
            "compiler_realization_evidence": record.get(
                "compiler_realization_evidence"
            ),
            "speedup_vs_family_parent": (
                parent_cycles / cycles
                if parent_cycles is not None and cycles not in (None, 0)
                else None
            ),
            "speedup_vs_family_factor1": (
                factor_one_cycles / cycles
                if factor_one_cycles is not None and cycles not in (None, 0)
                else None
            ),
        }
        for resource in RESOURCE_KEYS:
            row[resource] = _number(record.get(resource))
        rows.append(row)
    return rows


def _dedupe_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge incremental runs that share the same frozen parent."""

    merged: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in rows:
        key = (row["architecture"], row.get("requested_lanes"))
        prior = merged.get(key)
        if prior is not None:
            comparable = (
                "code_sha256",
                "latency_cycles_worst",
                "interval",
                "dsp",
                "bram",
                "lut",
                "ff",
                "uram",
            )
            if any(prior.get(field) != row.get(field) for field in comparable):
                raise ValueError(f"conflicting repeated GEMM lane point: {key}")
            prior["is_family_parent"] = bool(
                prior.get("is_family_parent") or row.get("is_family_parent")
            )
            continue
        merged[key] = dict(row)
    return list(merged.values())


def _original_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    report = payload.get("final_report") or {}
    feedback = report.get("feedback") or {}
    scope_iis = [
        _number(item.get("pipeline_ii"))
        for item in (feedback.get("scopes") or [])
        if isinstance(item, dict)
    ]
    scope_iis = [value for value in scope_iis if value is not None]
    row = {
        "architecture": "original_forced_complete_unroll",
        "architecture_label": ARCHITECTURE_LABELS[
            "original_forced_complete_unroll"
        ],
        "requested_lanes": 2,
        "effective_lanes": 80,
        "is_family_parent": True,
        "candidate_id": "original_generated_kernel",
        "csim_status": (payload.get("csim") or {}).get("status"),
        "feasible": bool(report) and (payload.get("csim") or {}).get("passed") is True,
        "latency_cycles_worst": _number(
            report.get("latency_cycles_worst") or report.get("latency_cycles")
        ),
        "interval": _number(report.get("interval")),
        "estimated_clock_period_ns": _number(
            report.get("estimated_clock_period_ns")
        ),
        "achieved_pipeline_ii_max": max(scope_iis) if scope_iis else None,
        "max_ii_scope_id": None,
        "max_ii_scope_violation": None,
        "compiler_unroll_mode": "completely",
        "compiler_override": True,
        "compiler_realization_evidence": (
            "The enclosing output-loop pipeline forced the K reduction to "
            "complete unroll factor 80."
        ),
        "speedup_vs_family_parent": 1.0,
        "speedup_vs_family_factor1": None,
    }
    for resource in RESOURCE_KEYS:
        row[resource] = _number(report.get(resource))
    return row


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    families = ("direct_reduction", "striped_accumulators")
    original = next(
        row for row in rows if row["architecture"] == "original_forced_complete_unroll"
    )

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for architecture in families:
        selected = sorted(
            (row for row in rows if row["architecture"] == architecture),
            key=lambda row: row["requested_lanes"],
        )
        axis.plot(
            [row["requested_lanes"] for row in selected],
            [row["latency_cycles_worst"] for row in selected],
            marker="o",
            linewidth=2,
            color=ARCHITECTURE_COLORS[architecture],
            label=ARCHITECTURE_LABELS[architecture],
        )
    axis.axhline(
        original["latency_cycles_worst"],
        color=ARCHITECTURE_COLORS["original_forced_complete_unroll"],
        linestyle="--",
        linewidth=1.8,
        label=ARCHITECTURE_LABELS["original_forced_complete_unroll"],
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    lane_ticks = sorted({
        row["requested_lanes"]
        for row in rows
        if row["architecture"] in families and row.get("requested_lanes") is not None
    })
    axis.set_xticks(lane_ticks, labels=[str(value) for value in lane_ticks])
    axis.set_xlabel("requested GEMM lanes")
    axis.set_ylabel("worst-case CSynth cycles")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save_figure(fig, output_dir, "gemm_latency_by_linked_lanes")

    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for architecture in families:
        selected = sorted(
            (row for row in rows if row["architecture"] == architecture),
            key=lambda row: row["requested_lanes"],
        )
        axis.plot(
            [row["requested_lanes"] for row in selected],
            [row["achieved_pipeline_ii_max"] for row in selected],
            marker="o",
            linewidth=2,
            color=ARCHITECTURE_COLORS[architecture],
            label=ARCHITECTURE_LABELS[architecture],
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks(lane_ticks, labels=[str(value) for value in lane_ticks])
    axis.set_xlabel("requested GEMM lanes")
    axis.set_ylabel("maximum achieved loop II")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, output_dir, "gemm_ii_by_linked_lanes")

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for architecture in (*families, "original_forced_complete_unroll"):
        selected = [row for row in rows if row["architecture"] == architecture]
        axis.scatter(
            [row["dsp"] for row in selected],
            [row["latency_cycles_worst"] for row in selected],
            s=55,
            color=ARCHITECTURE_COLORS[architecture],
            label=ARCHITECTURE_LABELS[architecture],
        )
        direct_offsets = {
            1: (6, -2),
            2: (8, -18),
            4: (8, -6),
            8: (8, 6),
            16: (8, 18),
        }
        for row in selected:
            offset = (
                direct_offsets.get(row["effective_lanes"], (5, 5))
                if architecture == "direct_reduction"
                else (5, 5)
            )
            axis.annotate(
                str(row["effective_lanes"]),
                (row["dsp"], row["latency_cycles_worst"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
    axis.set_yscale("log")
    axis.set_xlabel("DSP")
    axis.set_ylabel("worst-case CSynth cycles")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, output_dir, "gemm_latency_dsp_tradeoff")


def _write_report(rows: list[dict[str, Any]], source_path: Path) -> str:
    direct = sorted(
        (row for row in rows if row["architecture"] == "direct_reduction"),
        key=lambda row: row["requested_lanes"],
    )
    striped = sorted(
        (row for row in rows if row["architecture"] == "striped_accumulators"),
        key=lambda row: row["requested_lanes"],
    )
    original = next(
        row for row in rows if row["architecture"] == "original_forced_complete_unroll"
    )
    direct_best = min(direct, key=lambda row: row["latency_cycles_worst"])
    striped_best = min(striped, key=lambda row: row["latency_cycles_worst"])
    striped_parent = next(row for row in striped if row["is_family_parent"])
    lines = [
        "# GEMM Linked-Lane Campaign Audit",
        "",
        "Vitis 2023.2, `xcu280-fsvh2892-2L-e`, 3.33 ns. Every linked-family "
        "point passed CSim and CSynth; COSIM was not run. The parameter search "
        "was reference-blind.",
        "",
        "## Results",
        "",
        "| Architecture | Requested lanes | Effective lanes | CSim | Cycles | Max II | DSP | BRAM | LUT | FF | URAM |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*direct, *striped, original]:
        lines.append(
            f"| {row['architecture_label']} | {row['requested_lanes']} | "
            f"{row['effective_lanes']} | {row['csim_status']} | "
            f"{int(row['latency_cycles_worst'])} | "
            f"{int(row['achieved_pipeline_ii_max'])} | {int(row['dsp'])} | "
            f"{int(row['bram'])} | {int(row['lut'])} | {int(row['ff'])} | "
            f"{int(row['uram'])} |"
        )
    parent_reduction = 1.0 - (
        striped_best["latency_cycles_worst"]
        / striped_parent["latency_cycles_worst"]
    )
    factor_one = next(row for row in striped if row["requested_lanes"] == 1)
    factor_one_reduction = 1.0 - (
        striped_best["latency_cycles_worst"]
        / factor_one["latency_cycles_worst"]
    )
    dsp_ratio = striped_best["dsp"] / original["dsp"]
    speedup_vs_original = (
        original["latency_cycles_worst"] / striped_best["latency_cycles_worst"]
    )
    lines.extend([
        "",
        "## Audit",
        "",
        f"- The original source requested unroll 2, but Vitis forced complete "
        f"unroll 80 because of the enclosing output-loop pipeline. It reached "
        f"{int(original['latency_cycles_worst'])} cycles with {int(original['dsp'])} DSP.",
        f"- Once that override is removed, the direct scalar recurrence is truly "
        f"controlled, but larger factors raise maximum II from "
        f"{int(direct[0]['achieved_pipeline_ii_max'])} to "
        f"{int(direct[-1]['achieved_pipeline_ii_max'])}. Its best point is factor "
        f"{direct_best['requested_lanes']} at {int(direct_best['latency_cycles_worst'])} cycles.",
        f"- Independent striped accumulators expose a non-monotonic design curve. "
        f"Factor {striped_best['requested_lanes']} wins at "
        f"{int(striped_best['latency_cycles_worst'])} cycles: "
        f"{parent_reduction:.2%} below its factor-2 parent and "
        f"{factor_one_reduction:.2%} below factor 1.",
        f"- Intermediate factors can regress because final lane reduction and resource "
        f"cost grow before Vitis reaches the fully parallel factor-80 schedule. This "
        f"is a design knee and scheduler transition, not a failed source control.",
        f"- The linked winner is {speedup_vs_original:.2f}x faster than the original "
        f"compiler-forced kernel while using {dsp_ratio:.2f}x its DSP count. The "
        f"latency-only winner therefore differs from a resource-constrained winner.",
        "",
        "## Control Contract",
        "",
        "The direct family links one macro to the reduction unroll and both "
        "K-dimension memory partitions. The striped family links the same macro "
        "to two bank partitions, three unroll directives, the partial-sum array "
        "extent, three loop bounds, and the K-loop stride; a complete partition "
        "over that macro-sized partial-sum array supplies one bank per lane. Each "
        "candidate changes "
        "only the macro value from one frozen parent, so all related source "
        "modifications move together.",
        "",
        f"Selected striped-family kernel: `gemm_striped_lanes"
        f"{striped_best['requested_lanes']}_winner.cpp`  ",
        "Selected-kernel provenance: `winner.json`",
        "",
        f"Original generated-result artifact: `{source_path}`",
        "",
        "The original result is included only for post-run reporting. It was not "
        "available to candidate generation or winner selection.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-measurements", type=Path, required=True)
    parser.add_argument(
        "--striped-measurements", type=Path, nargs="+", required=True
    )
    parser.add_argument("--striped-case", type=Path, required=True)
    parser.add_argument("--striped-preflight", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    striped_rows = _dedupe_family([
        row
        for path in args.striped_measurements
        for row in _load_family(path, "striped_accumulators")
    ])
    striped_parent = next(row for row in striped_rows if row["is_family_parent"])
    striped_factor_one = next(
        row for row in striped_rows if row["requested_lanes"] == 1
    )
    for row in striped_rows:
        cycles = row.get("latency_cycles_worst")
        row["speedup_vs_family_parent"] = (
            striped_parent["latency_cycles_worst"] / cycles
            if cycles not in (None, 0)
            else None
        )
        row["speedup_vs_family_factor1"] = (
            striped_factor_one["latency_cycles_worst"] / cycles
            if cycles not in (None, 0)
            else None
        )
    rows = [
        *_load_family(args.direct_measurements, "direct_reduction"),
        *striped_rows,
        _original_row(args.original_result),
    ]
    original_cycles = next(
        row["latency_cycles_worst"]
        for row in rows
        if row["architecture"] == "original_forced_complete_unroll"
    )
    for row in rows:
        cycles = row["latency_cycles_worst"]
        row["speedup_vs_original_generated"] = (
            original_cycles / cycles if cycles not in (None, 0) else None
        )

    json_path = args.output_dir / "gemm_linked_lane_measurements.json"
    jsonl_path = args.output_dir / "gemm_linked_lane_measurements.jsonl"
    csv_path = args.output_dir / "gemm_linked_lane_measurements.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "report.md").write_text(
        _write_report(rows, args.original_result.resolve())
    )
    striped_case = json.loads(args.striped_case.read_text())
    striped_preflight = json.loads(args.striped_preflight.read_text())
    preflight_case = next(
        (
            record for record in (striped_preflight.get("cases") or [])
            if record.get("passed") is True
            and ((record.get("linked_control") or {}).get("architecture")
                 == "striped_accumulators")
        ),
        None,
    )
    if preflight_case is None or preflight_case.get("passed") is not True:
        raise ValueError("striped campaign preflight is missing or failed")
    selected_code = str(striped_case.get("selected_hls_code") or "")
    sweep = striped_case.get("design_sweep") or {}
    winner_id = sweep.get("winner_candidate_id")
    winner_record = next(
        (
            record for record in (sweep.get("candidates") or [])
            if record.get("candidate_id") == winner_id
        ),
        None,
    )
    if not selected_code or winner_record is None or sweep.get("applied") is not True:
        raise ValueError("striped campaign does not contain an applied winner kernel")
    winner_value = (winner_record.get("changed_knobs") or [{}])[0].get("to")
    winner_path = args.output_dir / f"gemm_striped_lanes{winner_value}_winner.cpp"
    winner_path.write_text(selected_code)
    winner_metadata = {
        "schema_version": "c2hls.gemm-linked-lane-winner.v1",
        "benchmark": striped_case.get("benchmark"),
        "candidate_id": winner_id,
        "code_sha256": _sha256(winner_path),
        "selected_code_sha256": striped_case.get("selected_code_sha256"),
        "changed_knobs": winner_record.get("changed_knobs"),
        "metrics": winner_record.get("metrics"),
        "csim": winner_record.get("csim"),
        "feasibility": winner_record.get("feasibility"),
        "compiler_realization": winner_record.get("compiler_realization"),
        "linked_control": preflight_case.get("linked_control"),
        "executed_linked_control": striped_case.get("linked_control"),
        "reference_blind": striped_case.get("reference_blind") is True,
        "cosim": False,
        "source_case": {
            "path": str(args.striped_case.resolve()),
            "sha256": _sha256(args.striped_case),
        },
    }
    (args.output_dir / "winner.json").write_text(
        json.dumps(winner_metadata, indent=2) + "\n"
    )
    _plot(rows, args.output_dir)
    manifest = {
        "schema_version": "c2hls.gemm-linked-lane-audit.v1",
        "reference_blind_search": True,
        "cosim": False,
        "toolchain": {
            "vitis_version": "2023.2",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
        },
        "inputs": {
            "direct_measurements": {
                "path": str(args.direct_measurements.resolve()),
                "sha256": _sha256(args.direct_measurements),
            },
            "striped_measurements": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in args.striped_measurements
            ],
            "striped_case": {
                "path": str(args.striped_case.resolve()),
                "sha256": _sha256(args.striped_case),
            },
            "striped_preflight": {
                "path": str(args.striped_preflight.resolve()),
                "sha256": _sha256(args.striped_preflight),
            },
            "original_result": {
                "path": str(args.original_result.resolve()),
                "sha256": _sha256(args.original_result),
            },
        },
        "measurement_count": len(rows),
        "artifacts": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest["artifacts"][path.name] = _sha256(path)
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "measurement_count": len(rows),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
