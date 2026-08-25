#!/usr/bin/env python3
"""Analyze the staged skill-cardinality and guard-policy ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCHEMA_VERSION = "c2hls.skill-dose-analysis.v1"


def _records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    telemetry = record.get("skill_telemetry") or {}
    return {
        "benchmark": record.get("benchmark"),
        "problem": record.get("problem"),
        "sample_index": record.get("sample_index"),
        "requested_positive_skill_count": record.get(
            "requested_positive_skill_count"
        ),
        "prompt_policy": record.get("prompt_policy"),
        "valid": record.get("valid") is True,
        "candidate_status": record.get("candidate_status"),
        "candidate_latency_cycles": record.get(
            "candidate_latency_cycles"
        ),
        "phase_b_latency_cycles": record.get("phase_b_latency_cycles"),
        "input_tokens": record.get("input_tokens", 0),
        "rendered_skill_count": telemetry.get("rendered_skill_count", 0),
        "declared_applied_skill_count": telemetry.get(
            "declared_applied_skill_count", 0
        ),
        "verified_applied_skill_count": telemetry.get(
            "verified_applied_skill_count", 0
        ),
        "synthesized_candidate_skill_count": telemetry.get(
            "synthesized_candidate_skill_count", 0
        ),
        "result_path": record.get("result_path"),
    }


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for keys, subset in frame.groupby(
        [
            "problem",
            "requested_positive_skill_count",
            "prompt_policy",
        ],
        dropna=False,
    ):
        problem, count, policy = keys
        valid = subset[subset["valid"]]
        cycles = [
            float(value)
            for value in valid["candidate_latency_cycles"]
            if pd.notna(value) and float(value) > 0
        ]
        groups.append(
            {
                "problem": problem,
                "requested_positive_skill_count": int(count),
                "prompt_policy": policy,
                "samples": len(subset),
                "valid_samples": len(valid),
                "validity": len(valid) / len(subset),
                "median_cycles": median(cycles) if cycles else None,
                "median_input_tokens": (
                    float(valid["input_tokens"].median())
                    if len(valid) else None
                ),
                "median_rendered_skill_count": (
                    float(valid["rendered_skill_count"].median())
                    if len(valid) else None
                ),
                "median_declared_applied_skill_count": (
                    float(
                        valid["declared_applied_skill_count"].median()
                    )
                    if len(valid) else None
                ),
                "median_verified_applied_skill_count": (
                    float(
                        valid["verified_applied_skill_count"].median()
                    )
                    if len(valid) else None
                ),
            }
        )
    aggregate = pd.DataFrame(groups)
    baseline = (
        aggregate[
            (aggregate["requested_positive_skill_count"] == 0)
            & (aggregate["prompt_policy"] == "action_only")
        ][["problem", "median_cycles"]]
        .rename(columns={"median_cycles": "skillless_median_cycles"})
    )
    aggregate = aggregate.merge(baseline, on="problem", how="left")
    aggregate["speedup_vs_skillless"] = (
        aggregate["skillless_median_cycles"]
        / aggregate["median_cycles"]
    )
    return aggregate


def _monotonicity(
    aggregate: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    main = aggregate[aggregate["prompt_policy"] == "action_only"].copy()
    per_kernel = []
    for problem, subset in main.groupby("problem"):
        subset = subset.dropna(subset=["median_cycles"]).sort_values(
            "requested_positive_skill_count"
        )
        if len(subset) < 3:
            continue
        rho, p_value = spearmanr(
            subset["requested_positive_skill_count"],
            subset["median_cycles"],
        )
        changes = np.diff(subset["median_cycles"].to_numpy(dtype=float))
        per_kernel.append(
            {
                "problem": problem,
                "spearman_count_vs_cycles": float(rho),
                "spearman_p_value": float(p_value),
                "agrees_more_skills_lower_cycles": bool(rho < 0),
                "strict_nonincreasing": bool(np.all(changes <= 0)),
                "decreasing_adjacent_fraction": float(
                    np.mean(changes < 0)
                ),
            }
        )

    usable = main.dropna(
        subset=["median_cycles", "skillless_median_cycles"]
    ).copy()
    usable["log_ratio"] = np.log(
        usable["median_cycles"]
        / usable["skillless_median_cycles"]
    )
    rng = np.random.default_rng(46)
    problems = sorted(usable["problem"].unique())
    slopes = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(problems, size=len(problems), replace=True)
        pieces = [
            usable[usable["problem"] == problem] for problem in sampled
        ]
        sample = pd.concat(pieces, ignore_index=True)
        x = np.log1p(
            sample["requested_positive_skill_count"].to_numpy(dtype=float)
        )
        y = sample["log_ratio"].to_numpy(dtype=float)
        if len(set(x)) >= 2:
            slopes.append(float(np.polyfit(x, y, 1)[0]))
    lower, upper = (
        (
            float(np.percentile(slopes, 2.5)),
            float(np.percentile(slopes, 97.5)),
        )
        if slopes
        else (None, None)
    )
    agreements = sum(
        item["agrees_more_skills_lower_cycles"] for item in per_kernel
    )
    includes_zero = bool(
        lower is None or upper is None or lower <= 0.0 <= upper
    )
    expand = includes_zero or agreements < 6
    return {
        "per_kernel": per_kernel,
        "direction_agreement_count": agreements,
        "pilot_kernel_count": len(per_kernel),
        "cluster_bootstrap_log_cycle_slope": {
            "samples": len(slopes),
            "median": float(np.median(slopes)) if slopes else None,
            "ci95": [lower, upper],
            "includes_zero": includes_zero,
        },
        "expansion_rule": (
            "expand when clustered bootstrap CI includes zero or fewer "
            "than six of eight kernels agree that cycles decrease"
        ),
        "expand_skill_dose_to_remaining_19": expand,
    }


def _plot(aggregate: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main = aggregate[
        aggregate["prompt_policy"] == "action_only"
    ].dropna(subset=["speedup_vs_skillless"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for problem, subset in main.groupby("problem"):
        subset = subset.sort_values("requested_positive_skill_count")
        ax.plot(
            subset["requested_positive_skill_count"],
            subset["speedup_vs_skillless"],
            marker="o",
            linewidth=1.4,
            label=problem,
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Rendered positive skills")
    ax.set_ylabel("Candidate cycle speedup vs skillless")
    ax.set_title("Skill cardinality does not imply additive speedup")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    speedup_path = output_dir / "skill_dose_speedup.png"
    fig.savefig(speedup_path, dpi=180)
    plt.close(fig)

    summary = (
        aggregate.groupby(
            ["requested_positive_skill_count", "prompt_policy"]
        )
        .agg(
            validity=("validity", "mean"),
            median_tokens=("median_input_tokens", "median"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for policy, subset in summary.groupby("prompt_policy"):
        label = str(policy).replace("_", " ")
        axes[0].plot(
            subset["requested_positive_skill_count"],
            subset["validity"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            subset["requested_positive_skill_count"],
            subset["median_tokens"],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Mean candidate validity")
    axes[1].set_ylabel("Median input tokens")
    for ax in axes:
        ax.set_xlabel("Rendered positive skills")
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    operations_path = output_dir / "skill_dose_validity_tokens.png"
    fig.savefig(operations_path, dpi=180)
    plt.close(fig)
    return [str(speedup_path), str(operations_path)]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    records = _records(args.records)
    frame = pd.DataFrame(_flatten(record) for record in records)
    aggregate = _aggregate(frame)
    monotonicity = _monotonicity(
        aggregate,
        bootstrap_samples=args.bootstrap_samples,
    )
    preconditions = aggregate[
        aggregate["requested_positive_skill_count"].isin([3, 42])
    ].copy()
    guard_rows = []
    for keys, subset in preconditions.groupby(
        ["problem", "requested_positive_skill_count"]
    ):
        problem, count = keys
        by_policy = {
            row["prompt_policy"]: row
            for _, row in subset.iterrows()
        }
        action = by_policy.get("action_only")
        neutral = by_policy.get("positive_with_preconditions")
        guard_rows.append(
            {
                "problem": problem,
                "count": int(count),
                "action_only_cycles": (
                    action["median_cycles"] if action is not None else None
                ),
                "positive_preconditions_cycles": (
                    neutral["median_cycles"]
                    if neutral is not None else None
                ),
                "preconditions_speedup_vs_action_only": (
                    action["median_cycles"] / neutral["median_cycles"]
                    if action is not None
                    and neutral is not None
                    and pd.notna(action["median_cycles"])
                    and pd.notna(neutral["median_cycles"])
                    else None
                ),
            }
        )
    mismatches = frame[
        frame["rendered_skill_count"]
        != frame["requested_positive_skill_count"]
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "benchmarks": sorted(frame["problem"].dropna().unique()),
        "validity": float(frame["valid"].mean()) if len(frame) else None,
        "failure_classes": dict(Counter(frame["candidate_status"])),
        "exact_rendered_count_mismatches": len(mismatches),
        "monotonicity": monotonicity,
        "guard_policy_ablation": guard_rows,
        "telemetry": {
            "median_requested_skills": float(
                frame["requested_positive_skill_count"].median()
            ),
            "median_rendered_skills": float(
                frame["rendered_skill_count"].median()
            ),
            "median_declared_applied_skills": float(
                frame["declared_applied_skill_count"].median()
            ),
            "median_verified_applied_skills": float(
                frame["verified_applied_skill_count"].median()
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "skill_dose_records.csv"
    frame.to_csv(raw_csv, index=False)
    aggregate_csv = args.output_dir / "skill_dose_aggregate.csv"
    aggregate.to_csv(aggregate_csv, index=False)
    charts = _plot(aggregate, args.output_dir / "figures")
    output["artifacts"] = {
        "records_csv": str(raw_csv),
        "aggregate_csv": str(aggregate_csv),
        "charts": charts,
    }
    json_path = args.output_dir / "skill_dose_analysis.json"
    json_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Skill-Dose Pilot",
        "",
        f"- Records: {len(records)}",
        f"- Candidate validity: {output['validity']:.3f}",
        f"- Exact rendered-count mismatches: {len(mismatches)}",
        (
            "- Direction agreement: "
            f"{monotonicity['direction_agreement_count']}/"
            f"{monotonicity['pilot_kernel_count']}"
        ),
        (
            "- Cluster-bootstrap slope 95% CI: "
            f"{monotonicity['cluster_bootstrap_log_cycle_slope']['ci95']}"
        ),
        (
            "- Expand cardinality sweep to remaining 19 kernels: "
            f"{monotonicity['expand_skill_dose_to_remaining_19']}"
        ),
        "",
        "Skills are treated as candidate transformations. Exposure, declared "
        "use, verified static evidence, and synthesis are reported separately.",
    ]
    (args.output_dir / "skill_dose_analysis.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    result = analyze(parse_args())
    print(
        json.dumps(
            {
                "records": result["records"],
                "validity": result["validity"],
                "expand": result["monotonicity"][
                    "expand_skill_dose_to_remaining_19"
                ],
            },
            sort_keys=True,
        )
    )
