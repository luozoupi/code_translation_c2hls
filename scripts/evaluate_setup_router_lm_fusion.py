#!/usr/bin/env python3
"""Evaluate an exploratory ExtraTrees plus small-LM Top-5 fusion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_setup_router_topk_architectures import (
    _wilson_interval,
    _write_csv,
)


SCHEMA_VERSION = "c2hls.setup-router-lm-fusion.v1"
DEFAULT_CLASSICAL = (
    REPO_ROOT
    / "artifacts"
    / "setup_router"
    / "topk_architecture_comparison_corrected_v2_20260728"
    / "oof_setup_rankings.csv"
)
DEFAULT_LM = (
    REPO_ROOT
    / "artifacts"
    / "setup_router"
    / "qwen3_06b_lora_oof_corrected_v2_20260728"
    / "oof_setup_rankings.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "setup_router"
    / "qwen3_06b_lora_oof_corrected_v2_20260728"
    / "fusion"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _router_rows(
    rows: list[dict[str, str]],
    router: str,
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["router"] != router:
            continue
        output.setdefault(row["problem"], []).append(row)
    for values in output.values():
        values.sort(
            key=lambda row: (
                int(row["predicted_rank"]),
                row["setup_id"],
            )
        )
    return output


def _selection_result(
    selected: list[dict[str, str]],
    *,
    policy: str,
) -> dict[str, Any]:
    valid = [row for row in selected if row["actual_valid"] == "True"]
    winner = (
        min(
            valid,
            key=lambda row: (
                float(row["actual_cycles"]),
                row["setup_id"],
            ),
        )
        if valid
        else None
    )
    actual_best = selected[0]["actual_best_setup_id"]
    regret = float(winner["actual_regret"]) if winner else 1000.0
    return {
        "problem": selected[0]["problem"],
        "benchmark_lineage": selected[0]["benchmark_lineage"],
        "policy": policy,
        "selected_setup_ids": json.dumps(
            [row["setup_id"] for row in selected]
        ),
        "selected_winner_setup_id": (
            winner["setup_id"] if winner else ""
        ),
        "selected_winner_regret": regret,
        "actual_best_setup_id": actual_best,
        "exact_best_in_top_5": any(
            row["setup_id"] == actual_best for row in selected
        ),
        "within_5pct_of_best": regret <= 1.05,
        "at_least_one_valid": bool(valid),
        "candidate_validity": len(valid) / len(selected),
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    count = len(rows)
    exact = sum(bool(row["exact_best_in_top_5"]) for row in rows)
    within = sum(bool(row["within_5pct_of_best"]) for row in rows)
    exact_low, exact_high = _wilson_interval(exact, count)
    within_low, within_high = _wilson_interval(within, count)
    regrets = [float(row["selected_winner_regret"]) for row in rows]
    return {
        "policy": policy,
        "benchmark_count": count,
        "exact_best_count": exact,
        "top_5_exact_accuracy": exact / count,
        "top_5_exact_wilson_low": exact_low,
        "top_5_exact_wilson_high": exact_high,
        "within_5pct_count": within,
        "within_5pct_coverage": within / count,
        "within_5pct_wilson_low": within_low,
        "within_5pct_wilson_high": within_high,
        "geomean_regret": geometric_mean(regrets),
        "p95_regret": float(np.percentile(regrets, 95)),
        "selection_validity": sum(
            bool(row["at_least_one_valid"]) for row in rows
        )
        / count,
        "candidate_savings_vs_10_setups": 0.5,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    classical = _read_csv(args.classical_rankings)
    lm = _read_csv(args.lm_rankings)
    sources = {
        "ExtraTrees pairwise": _router_rows(
            classical,
            "extratrees_pairwise_hybrid32",
        ),
        "ExtraTrees rank ensemble v2": _router_rows(
            classical,
            "extratrees_rank_ensemble_v2",
        ),
        "Qwen3-0.6B prompted base": _router_rows(
            lm,
            "qwen3_06b_prompted_base_oof",
        ),
    }
    problems = sorted(
        set.intersection(
            *(set(values) for values in sources.values())
        )
    )
    selection_rows = []
    for label, values in sources.items():
        for problem in problems:
            selection_rows.append(
                _selection_result(
                    values[problem][:5],
                    policy=label,
                )
            )

    tree = sources["ExtraTrees rank ensemble v2"]
    language_model = sources["Qwen3-0.6B prompted base"]
    fusion_label = "ExtraTrees rank ensemble Top-4 + Qwen unique Top-1"
    for problem in problems:
        selected = list(tree[problem][:4])
        selected_ids = {row["setup_id"] for row in selected}
        selected.extend(
            row
            for row in language_model[problem]
            if row["setup_id"] not in selected_ids
        )
        selection_rows.append(
            _selection_result(
                selected[:5],
                policy=fusion_label,
            )
        )

    policy_order = [
        "ExtraTrees pairwise",
        "ExtraTrees rank ensemble v2",
        "Qwen3-0.6B prompted base",
        fusion_label,
    ]
    metrics = [
        _aggregate(
            [
                row
                for row in selection_rows
                if row["policy"] == policy
            ],
            policy=policy,
        )
        for policy in policy_order
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "top5_fusion_metrics.csv", metrics)
    _write_csv(
        args.output_dir / "top5_fusion_per_benchmark.csv",
        selection_rows,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "protocol": "five-fold benchmark-grouped OOF",
            "development_lineages": len(problems),
            "fusion_allocation": "four tree-ranked plus one unique LM-ranked",
            "allocation_status": (
                "exploratory; selected after development OOF inspection"
            ),
            "fixed_test_confirmation_required": True,
        },
        "inputs": {
            "classical_rankings": str(args.classical_rankings.resolve()),
            "lm_rankings": str(args.lm_rankings.resolve()),
        },
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Top-5 Classical and Small-LM Router Fusion",
        "",
        "| policy | exact best | within 5% | geomean regret | p95 regret | savings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric['policy']} | "
            f"{metric['top_5_exact_accuracy']:.3f} "
            f"({metric['exact_best_count']}/"
            f"{metric['benchmark_count']}) | "
            f"{metric['within_5pct_coverage']:.3f} "
            f"({metric['within_5pct_count']}/"
            f"{metric['benchmark_count']}) | "
            f"{metric['geomean_regret']:.3f} | "
            f"{metric['p95_regret']:.3f} | 50% |"
        )
    lines.extend(
        [
            "",
            "The 4+1 fusion is an exploratory development result because its allocation was inspected on these OOF predictions. Freeze it before evaluating the untouched test lineages.",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical-rankings",
        type=Path,
        default=DEFAULT_CLASSICAL,
    )
    parser.add_argument("--lm-rankings", type=Path, default=DEFAULT_LM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
