#!/usr/bin/env python3
"""Replay and export the completed corrected-v2 setup tournament."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from setup_router import (  # noqa: E402
    CORRECTED_VERSION,
    select_tournament_winner,
    setup_registry,
)


SCHEMA_VERSION = "c2hls.corrected-setup-sweep-report.v1"
DEFAULT_RECORDS = (
    REPO_ROOT
    / "artifacts/corrected_setup_matrix_v2/corrected_matrix_records.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results_sweeps/corrected_v2_setup_tournament_20260728"
)
DEFAULT_ANALYSIS = REPO_ROOT / "artifacts/analysis"
CLOCK_NS = 3.33
SCOPE_LABELS = {
    "skillless": "Skillless",
    "matched_positive": "Matched positive",
    "smart_best_fit_v2": "Smart best-fit v2",
    "smart_exhaustive_v2": "Smart exhaustive v2",
    "all_positive_preconditions": "All-positive + preconditions",
}
SCOPE_SHORT = {
    "skillless": "SL",
    "matched_positive": "MP",
    "smart_best_fit_v2": "BF",
    "smart_exhaustive_v2": "EX",
    "all_positive_preconditions": "AP",
}
COLORS = {
    "corrected_v2:flash:skillless": "#6b6b6b",
    "corrected_v2:flash:matched_positive": "#4c78a8",
    "corrected_v2:flash:smart_best_fit_v2": "#2a9d8f",
    "corrected_v2:flash:smart_exhaustive_v2": "#e9c46a",
    "corrected_v2:flash:all_positive_preconditions": "#d66ba0",
    "corrected_v2:multistep:skillless": "#222222",
    "corrected_v2:multistep:matched_positive": "#1f4e79",
    "corrected_v2:multistep:smart_best_fit_v2": "#146b55",
    "corrected_v2:multistep:smart_exhaustive_v2": "#d97706",
    "corrected_v2:multistep:all_positive_preconditions": "#b23a48",
}
SKILL_FIELDS = (
    "catalog_skill_ids",
    "routed_skill_ids",
    "rendered_skill_ids",
    "declared_applied_skill_ids",
    "verified_applied_skill_ids",
    "synthesized_candidate_skill_ids",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_unique(values: list[Any]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _setup_label(setup: dict[str, Any]) -> str:
    strategy = str(setup["strategy"]).title()
    return f"{strategy} | {SCOPE_LABELS[str(setup['skill_scope'])]}"


def _setup_short(setup: dict[str, Any]) -> str:
    strategy = "F" if setup["strategy"] == "flash" else "M"
    return f"{strategy}-{SCOPE_SHORT[str(setup['skill_scope'])]}"


def _report(detail: dict[str, Any], key: str) -> dict[str, Any]:
    value = detail.get(key)
    return value if isinstance(value, dict) else {}


def _cycles(report: dict[str, Any]) -> int | None:
    value = report.get("latency_cycles_worst")
    if value is None:
        value = report.get("latency_cycles")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _skill_telemetry(detail: dict[str, Any]) -> dict[str, Any]:
    aggregate: dict[str, list[str]] = {field: [] for field in SKILL_FIELDS}
    prompt_characters = 0
    routing_reasons = []
    per_step = []
    history = (
        detail.get("generated_step_history")
        or detail.get("optimization_history")
        or detail.get("steps")
        or []
    )
    for item in history:
        if not isinstance(item, dict):
            continue
        prompt = item.get("skill_prompt")
        routing = item.get("routing_decision")
        if not isinstance(prompt, dict) or not prompt:
            continue
        step = {"step_name": str(item.get("step_name") or "")}
        if isinstance(routing, dict) and routing.get("reason"):
            reason = str(routing["reason"])
            step["routing_reason"] = reason
            routing_reasons.append(reason)
        for field in SKILL_FIELDS:
            values = _ordered_unique(list(prompt.get(field) or []))
            aggregate[field].extend(values)
            step[field] = values
        try:
            rendered_characters = int(
                prompt.get("rendered_prompt_characters") or 0
            )
        except (TypeError, ValueError):
            rendered_characters = 0
        prompt_characters += max(rendered_characters, 0)
        per_step.append(step)
    output: dict[str, Any] = {
        "rendered_prompt_characters": prompt_characters,
        "routing_reasons": _ordered_unique(routing_reasons),
        "steps": per_step,
    }
    for field in SKILL_FIELDS:
        values = _ordered_unique(aggregate[field])
        output[field] = values
        output[field.replace("_ids", "_count")] = len(values)
    return output


def _selected_step(detail: dict[str, Any]) -> str:
    history = detail.get("best_so_far_history") or []
    measured = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        cycles = _cycles(item.get("report") or {})
        if cycles is not None and item.get("step_name"):
            measured.append((cycles, index, str(item["step_name"])))
    return min(measured)[2] if measured else ""


def _load_detail(candidate: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(candidate.get("result_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"missing setup result: {path}")
    detail = json.loads(path.read_text(encoding="utf-8"))
    detail.update(
        {
            "setup_id": candidate["setup_id"],
            "setup_fingerprint": candidate["setup_fingerprint"],
            "code_sha256": candidate["code_sha256"],
            "result_path": str(path),
        }
    )
    return detail


def _reference_cycles(detail: dict[str, Any]) -> int | None:
    validation = detail.get("reference_validation") or {}
    synthesis = validation.get("synthesis") or {}
    return _cycles(synthesis.get("report") or {})


def _llm_input_tokens(detail: dict[str, Any]) -> int | None:
    usage = detail.get("llm_usage") or {}
    for key in ("input_tokens", "prompt_tokens", "total_input_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _build(
    matrix_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    registry = {
        setup.setup_id: setup.to_record()
        for setup in setup_registry(CORRECTED_VERSION)
    }
    expected = set(registry)
    long_rows = []
    reports = []
    winner_rows = []
    for matrix in sorted(
        matrix_records, key=lambda item: str(item["benchmark"])
    ):
        benchmark = str(matrix["benchmark"])
        candidates = list(matrix.get("candidates") or [])
        actual = {str(candidate["setup_id"]) for candidate in candidates}
        if actual != expected:
            raise ValueError(
                f"{benchmark}: setup mismatch "
                f"missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        details = []
        for index, candidate in enumerate(candidates):
            detail = _load_detail(candidate)
            detail["candidate_index"] = index
            detail["setup"] = registry[str(candidate["setup_id"])]
            details.append(detail)

        replay = select_tournament_winner(details)
        replay_winner = replay.get("winner") or {}
        winner_setup_id = str(replay_winner.get("setup_id") or "")
        if winner_setup_id != matrix.get("winner_setup_id"):
            raise ValueError(
                f"{benchmark}: replay winner {winner_setup_id} differs "
                f"from stored {matrix.get('winner_setup_id')}"
            )

        valid_candidates = [
            candidate for candidate in candidates if candidate.get("valid")
        ]
        best_cycles = min(
            int(candidate["latency_cycles"])
            for candidate in valid_candidates
        )
        best_skillless_cycles = min(
            int(candidate["latency_cycles"])
            for candidate in valid_candidates
            if str(candidate["setup_id"]).endswith(":skillless")
        )
        rank_order = sorted(
            valid_candidates,
            key=lambda candidate: (
                int(candidate["latency_cycles"]),
                str(candidate["setup_fingerprint"]),
            ),
        )
        ranks = {
            str(candidate["setup_id"]): rank
            for rank, candidate in enumerate(rank_order, start=1)
        }
        detail_by_setup = {
            str(detail["setup_id"]): detail for detail in details
        }
        reference_cycles = _reference_cycles(details[0])
        for candidate in candidates:
            setup_id = str(candidate["setup_id"])
            setup = registry[setup_id]
            detail = detail_by_setup[setup_id]
            final_report = _report(detail, "final_report")
            baseline_report = _report(detail, "baseline_report")
            latency = int(candidate["latency_cycles"])
            telemetry = _skill_telemetry(detail)
            bottleneck_kinds = _ordered_unique(
                [
                    item.get("kind")
                    for item in (
                        (baseline_report.get("feedback") or {}).get(
                            "bottlenecks"
                        )
                        or []
                    )
                    if isinstance(item, dict)
                ]
            )
            row = {
                "benchmark": benchmark,
                "problem": str(
                    matrix.get("problem")
                    or benchmark.removeprefix("hlsfactory_")
                ),
                "benchmark_lineage": matrix.get("benchmark_lineage"),
                "split": matrix.get("split"),
                "setup_id": setup_id,
                "setup_short": _setup_short(setup),
                "setup_label": _setup_label(setup),
                "strategy": setup["strategy"],
                "skill_scope": setup["skill_scope"],
                "prompt_mode": setup["prompt_mode"],
                "candidate_policy": setup["candidate_policy"],
                "setup_fingerprint": candidate["setup_fingerprint"],
                "valid_csim_csynth_timing_resources": bool(
                    candidate.get("valid")
                ),
                "latency_cycles": latency,
                "latency_ns_at_3_33ns": latency * CLOCK_NS,
                "setup_rank": ranks[setup_id],
                "regret_vs_tournament_best": latency / best_cycles,
                "within_5pct_of_best": latency / best_cycles <= 1.05,
                "tournament_winner": setup_id == winner_setup_id,
                "best_skillless_cycles": best_skillless_cycles,
                "speedup_vs_best_skillless": (
                    best_skillless_cycles / latency
                ),
                "phase_b_cycles": _cycles(baseline_report),
                "speedup_vs_phase_b": (
                    _cycles(baseline_report) / latency
                    if _cycles(baseline_report)
                    else None
                ),
                "reference_csynth_cycles_reporting_only": reference_cycles,
                "speedup_vs_reference_reporting_only": (
                    reference_cycles / latency
                    if reference_cycles
                    else None
                ),
                "selected_step_name": _selected_step(detail),
                "estimated_clock_period_ns": final_report.get(
                    "estimated_clock_period_ns"
                ),
                "requested_clock_period_ns": final_report.get(
                    "requested_clock_period_ns"
                ),
                "slack_ns": final_report.get("slack_ns"),
                "bram": final_report.get("bram"),
                "dsp": final_report.get("dsp"),
                "ff": final_report.get("ff"),
                "lut": final_report.get("lut"),
                "uram": final_report.get("uram"),
                "phase_b_bottleneck_kinds": json.dumps(
                    bottleneck_kinds, separators=(",", ":")
                ),
                "llm_input_tokens": _llm_input_tokens(detail),
                "rendered_prompt_characters": telemetry[
                    "rendered_prompt_characters"
                ],
                "catalog_skill_count": telemetry["catalog_skill_count"],
                "routed_skill_count": telemetry["routed_skill_count"],
                "rendered_skill_count": telemetry["rendered_skill_count"],
                "declared_applied_skill_count": telemetry[
                    "declared_applied_skill_count"
                ],
                "verified_applied_skill_count": telemetry[
                    "verified_applied_skill_count"
                ],
                "synthesized_candidate_skill_count": telemetry[
                    "synthesized_candidate_skill_count"
                ],
                "routed_skill_ids": json.dumps(
                    telemetry["routed_skill_ids"], separators=(",", ":")
                ),
                "rendered_skill_ids": json.dumps(
                    telemetry["rendered_skill_ids"], separators=(",", ":")
                ),
                "declared_applied_skill_ids": json.dumps(
                    telemetry["declared_applied_skill_ids"],
                    separators=(",", ":"),
                ),
                "verified_applied_skill_ids": json.dumps(
                    telemetry["verified_applied_skill_ids"],
                    separators=(",", ":"),
                ),
                "synthesized_candidate_skill_ids": json.dumps(
                    telemetry["synthesized_candidate_skill_ids"],
                    separators=(",", ":"),
                ),
                "routing_reasons": json.dumps(
                    telemetry["routing_reasons"], separators=(",", ":")
                ),
                "code_sha256": candidate["code_sha256"],
                "result_path": candidate["result_path"],
            }
            long_rows.append(row)
            if row["tournament_winner"]:
                winner_rows.append(dict(row))

        reports.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": benchmark,
                "problem": matrix.get("problem"),
                "winner_setup_id": winner_setup_id,
                "winner_explanation": replay["winner_explanation"],
                "reference_blind_selection": True,
                "reference_metrics_reporting_only": True,
                "source_tournament_result": matrix.get("result_path"),
            }
        )
    frame = pd.DataFrame(long_rows)
    if len(frame) != len(matrix_records) * len(expected):
        raise ValueError("incomplete long-form setup table")
    return frame, reports, winner_rows


def _setup_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setup_id, group in frame.groupby("setup_id", sort=False):
        regrets = [float(value) for value in group["regret_vs_tournament_best"]]
        rows.append(
            {
                "setup_id": setup_id,
                "setup_short": group.iloc[0]["setup_short"],
                "setup_label": group.iloc[0]["setup_label"],
                "strategy": group.iloc[0]["strategy"],
                "skill_scope": group.iloc[0]["skill_scope"],
                "benchmarks": len(group),
                "valid": int(
                    group["valid_csim_csynth_timing_resources"].sum()
                ),
                "wins": int(group["tournament_winner"].sum()),
                "within_5pct": int(group["within_5pct_of_best"].sum()),
                "within_5pct_coverage": float(
                    group["within_5pct_of_best"].mean()
                ),
                "geomean_regret": geometric_mean(regrets),
                "median_regret": float(
                    group["regret_vs_tournament_best"].median()
                ),
                "geomean_speedup_vs_best_skillless": geometric_mean(
                    [
                        float(value)
                        for value in group["speedup_vs_best_skillless"]
                    ]
                ),
                "median_rendered_skill_count": float(
                    group["rendered_skill_count"].median()
                ),
                "median_verified_applied_skill_count": float(
                    group["verified_applied_skill_count"].median()
                ),
            }
        )
    order = {
        setup.setup_id: index
        for index, setup in enumerate(setup_registry(CORRECTED_VERSION))
    }
    return pd.DataFrame(rows).sort_values(
        "setup_id", key=lambda values: values.map(order)
    )


def _wide(frame: pd.DataFrame) -> pd.DataFrame:
    setup_order = [
        setup.setup_id for setup in setup_registry(CORRECTED_VERSION)
    ]
    wide = frame.pivot(
        index=["benchmark", "problem"],
        columns="setup_id",
        values="latency_cycles",
    ).reindex(columns=setup_order)
    return wide.reset_index()


def _charts(
    frame: pd.DataFrame,
    setup_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    setup_order = [
        setup.setup_id for setup in setup_registry(CORRECTED_VERSION)
    ]
    setup_records = {
        setup.setup_id: setup.to_record()
        for setup in setup_registry(CORRECTED_VERSION)
    }
    labels = [_setup_label(setup_records[item]) for item in setup_order]
    benchmarks = sorted(frame["benchmark"].unique())
    problems = [item.removeprefix("hlsfactory_") for item in benchmarks]
    outputs = []

    fig, axis = plt.subplots(figsize=(26, 12))
    positions = np.arange(len(benchmarks))
    width = 0.085
    for index, setup_id in enumerate(setup_order):
        selected = (
            frame[frame["setup_id"] == setup_id]
            .set_index("benchmark")
            .reindex(benchmarks)
        )
        axis.bar(
            positions + (index - 4.5) * width,
            selected["latency_cycles"],
            width,
            color=COLORS[setup_id],
            label=labels[index],
        )
    axis.set_yscale("log")
    axis.set_ylabel("Vitis CSynth latency (cycles, log scale)")
    axis.set_xticks(positions, problems, rotation=60, ha="right")
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(ncol=2, fontsize=9, frameon=False)
    axis.set_title("Corrected-v2 Sonnet 4.6 setup sweep: all 10 setups")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"corrected_v2_latency_grouped_log.{suffix}"
        fig.savefig(path, dpi=190)
        outputs.append(path)
    plt.close(fig)

    regret = frame.pivot(
        index="benchmark",
        columns="setup_id",
        values="regret_vs_tournament_best",
    ).reindex(index=benchmarks, columns=setup_order)
    heat = np.log2(regret.to_numpy(dtype=float))
    fig, axis = plt.subplots(figsize=(19, 18))
    image = axis.imshow(
        heat,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(1.0, float(np.nanpercentile(heat, 90))),
    )
    axis.set_xticks(np.arange(len(setup_order)), labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(benchmarks)), problems)
    for row in range(len(benchmarks)):
        for column in range(len(setup_order)):
            value = float(regret.iloc[row, column])
            text = "1.00" if value < 1.005 else f"{value:.1f}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color="black" if heat[row, column] < 2.5 else "white",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("log2 regret vs exhaustive winner")
    axis.set_title("Per-benchmark setup regret (cell text is linear regret)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"corrected_v2_setup_regret_heatmap.{suffix}"
        fig.savefig(path, dpi=190)
        outputs.append(path)
    plt.close(fig)

    winners = frame[frame["tournament_winner"]].copy()
    winners = winners.sort_values(
        "speedup_vs_best_skillless", ascending=True
    )
    fig, axis = plt.subplots(figsize=(14, 12))
    axis.barh(
        winners["problem"],
        winners["speedup_vs_best_skillless"],
        color=[COLORS[item] for item in winners["setup_id"]],
    )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Winner speedup over best skillless setup (log scale)")
    axis.grid(axis="x", which="both", alpha=0.2)
    axis.legend(
        handles=[
            Patch(color=COLORS[item], label=_setup_label(setup_records[item]))
            for item in setup_order
            if item in set(winners["setup_id"])
        ],
        ncol=2,
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    axis.set_title("Tournament winner benefit and winning setup")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"corrected_v2_winner_vs_skillless.{suffix}"
        fig.savefig(path, dpi=190)
        outputs.append(path)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 7))
    summary = setup_summary.set_index("setup_id").reindex(setup_order)
    axis.bar(
        np.arange(len(setup_order)),
        summary["wins"],
        color=[COLORS[item] for item in setup_order],
    )
    axis.set_xticks(
        np.arange(len(setup_order)),
        [setup_records[item]["strategy"][0].upper() + "-" + _setup_short(setup_records[item]).split("-", 1)[1] for item in setup_order],
    )
    axis.set_ylabel("Tournament wins")
    axis.set_xlabel("Setup (F=flash, M=multistep; see legend)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(
        handles=[
            Patch(color=COLORS[item], label=_setup_label(setup_records[item]))
            for item in setup_order
        ],
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    axis.set_title("Winner count by corrected-v2 setup")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"corrected_v2_setup_winner_counts.{suffix}"
        fig.savefig(path, dpi=190)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _markdown(
    frame: pd.DataFrame,
    setup_summary: pd.DataFrame,
) -> str:
    winners = frame[frame["tournament_winner"]].sort_values("benchmark")
    speedups = list(winners["speedup_vs_best_skillless"].astype(float))
    setup_counts = Counter(winners["setup_label"])
    lines = [
        "# Corrected-v2 Setup Tournament",
        "",
        f"- Benchmarks: {frame['benchmark'].nunique()}",
        f"- Setups per benchmark: {frame['setup_id'].nunique()}",
        f"- Measured candidates: {len(frame)}",
        "- Validation: executed CSim plus Vitis 2023.2 CSynth, U280, 3.33 ns; no COSIM.",
        "- Selection is reference-blind. Reference CSynth cycles appear only in reporting columns.",
        (
            "- Tournament vs best skillless: "
            f"{sum(value > 1.0000001 for value in speedups)} wins, "
            f"{sum(abs(value - 1.0) <= 1e-7 for value in speedups)} ties, "
            f"geomean {geometric_mean(speedups):.4f}x."
        ),
        "",
        "## Setup Legend",
        "",
        "| short | setup | setup id |",
        "|---|---|---|",
    ]
    for setup in setup_registry(CORRECTED_VERSION):
        record = setup.to_record()
        lines.append(
            f"| {_setup_short(record)} | "
            f"{_setup_label(record).replace(' | ', ' / ')} | "
            f"`{setup.setup_id}` |"
        )
    lines.extend(
        [
            "",
            "## Setup Aggregate",
            "",
            "| setup | wins | within 5% | geomean regret | median rendered | median verified |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in setup_summary.iterrows():
        lines.append(
            f"| {str(row['setup_label']).replace(' | ', ' / ')} | "
            f"{int(row['wins'])} | "
            f"{float(row['within_5pct_coverage']):.3f} | "
            f"{float(row['geomean_regret']):.3f} | "
            f"{float(row['median_rendered_skill_count']):.1f} | "
            f"{float(row['median_verified_applied_skill_count']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Benchmark Winners",
            "",
            "| benchmark | winner setup | cycles | best skillless | speedup | phase B | verified skills |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in winners.iterrows():
        verified = ", ".join(json.loads(row["verified_applied_skill_ids"]))
        lines.append(
            f"| {row['problem']} | "
            f"{str(row['setup_label']).replace(' | ', ' / ')} | "
            f"{int(row['latency_cycles']):,} | "
            f"{int(row['best_skillless_cycles']):,} | "
            f"{float(row['speedup_vs_best_skillless']):.4f}x | "
            f"{int(row['phase_b_cycles']):,} | {verified or 'none verified'} |"
        )
    lines.extend(
        [
            "",
            "## Winner Distribution",
            "",
        ]
    )
    for setup_label, count in sorted(
        setup_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- {setup_label.replace(' | ', ' / ')}: {count}")
    lines.extend(
        [
            "",
            "The mode-fit explanations are deterministic summaries of measured "
            "within-run evidence. They do not claim that a setup or an "
            "individual skill caused the observed improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def export(args: argparse.Namespace) -> dict[str, Any]:
    matrix_records = _load_jsonl(args.records)
    if not matrix_records:
        raise ValueError("matrix records are empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    frame, reports, winner_rows = _build(matrix_records)
    setup_summary = _setup_summary(frame)
    wide = _wide(frame)

    long_path = args.output_dir / "corrected_v2_setup_results_long.csv"
    frame.sort_values(["benchmark", "setup_id"]).to_csv(
        long_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    wide_path = args.output_dir / "corrected_v2_latency_cycles_wide.csv"
    wide.to_csv(wide_path, index=False)
    setup_path = args.output_dir / "corrected_v2_setup_summary.csv"
    setup_summary.to_csv(setup_path, index=False)
    winners_path = args.output_dir / "corrected_v2_winners.csv"
    pd.DataFrame(winner_rows).sort_values("benchmark").to_csv(
        winners_path, index=False
    )
    evidence_path = args.output_dir / "mode_fit_reports.jsonl"
    with evidence_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
    report_text = _markdown(frame, setup_summary)
    report_path = args.output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    chart_paths = _charts(frame, setup_summary, args.output_dir)

    analysis_stem = "corrected_v2_setup_tournament_20260728"
    analysis_md = args.analysis_dir / f"{analysis_stem}.md"
    analysis_json = args.analysis_dir / f"{analysis_stem}.json"
    analysis_md.write_text(report_text, encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_records": str(args.records.resolve()),
        "source_records_sha256": _sha256(args.records),
        "benchmarks": int(frame["benchmark"].nunique()),
        "setups": int(frame["setup_id"].nunique()),
        "candidate_measurements": len(frame),
        "valid_candidates": int(
            frame["valid_csim_csynth_timing_resources"].sum()
        ),
        "winner_setup_counts": dict(
            Counter(row["setup_id"] for row in winner_rows)
        ),
        "winner_vs_best_skillless": {
            "wins": int(
                sum(
                    float(row["speedup_vs_best_skillless"]) > 1.0000001
                    for row in winner_rows
                )
            ),
            "ties": int(
                sum(
                    abs(float(row["speedup_vs_best_skillless"]) - 1.0)
                    <= 1e-7
                    for row in winner_rows
                )
            ),
            "geomean_speedup": geometric_mean(
                [
                    float(row["speedup_vs_best_skillless"])
                    for row in winner_rows
                ]
            ),
        },
        "selection_contract": {
            "reference_blind": True,
            "gates": ["CSim", "CSynth", "timing", "U280 resource fit"],
            "vitis_version": "2023.2",
            "device": "xcu280-fsvh2892-2L-e",
            "clock_ns": CLOCK_NS,
            "cosim": False,
        },
    }
    analysis_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [
        long_path,
        wide_path,
        setup_path,
        winners_path,
        evidence_path,
        report_path,
        analysis_md,
        analysis_json,
        *chart_paths,
    ]
    manifest_path = args.output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **summary,
                "artifacts": {
                    str(path.relative_to(REPO_ROOT)): {
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifacts
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**summary, "output_dir": str(args.output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    if not args.records.is_file():
        parser.error(f"missing matrix records: {args.records}")
    return args


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), sort_keys=True))
