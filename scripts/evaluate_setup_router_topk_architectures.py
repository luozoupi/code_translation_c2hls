#!/usr/bin/env python3
"""Compare setup-router architectures at fixed top-1, top-3, and top-5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import matplotlib
import numpy as np
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_neural_setup_router import (
    VARIANTS,
    _canonical_hybrid_records,
    _fit_variant,
)
from scripts.train_setup_router import MANDATORY_BASELINE_SUFFIX, _load, _sha256
from scripts.train_strengthened_setup_router import (
    INVALID_REGRET_PENALTY,
    _canonicalize_phase_b,
    _diverse_voted_setups,
    _fit_direct,
    _fit_pairwise,
    _fit_within5,
    _outer_fold_records,
    _outcome_groups,
    _rank_consensus,
    _record_id,
    _retrieval_scores,
)


SCHEMA_VERSION = "c2hls.setup-router-topk-architecture-comparison.v1"
TOP_K_VALUES = (1, 3, 5)
PROTOCOLS = ("raw_predicted", "mandatory_skillless_fallback")
DISPLAY_NAMES = {
    "adaptive_extratrees_committee_v2": (
        "Current adaptive ExtraTrees committee v2"
    ),
    "extratrees_rank_ensemble_v2": "ExtraTrees rank ensemble v2",
    "extratrees_log_regret_hybrid32": "ExtraTrees log-regret",
    "extratrees_within5_hybrid32": "ExtraTrees within-5%",
    "extratrees_pairwise_hybrid32": "ExtraTrees pairwise",
    "extratrees_relative_hybrid32": "ExtraTrees relative cycles",
    "extratrees_absolute_structured": "ExtraTrees absolute",
    "embedding_retrieval_source_k1": "Embedding retrieval",
    "ranknet_hybrid32": "RankNet",
    "mlp_pairwise_hybrid32": "MLP pairwise",
    "mlp_deep_hybrid32": "MLP deep",
    "mlp_shallow_hybrid32": "MLP shallow",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _selected_candidates(
    predicted: list[dict[str, Any]],
    *,
    top_k: int,
    protocol: str,
) -> list[dict[str, Any]]:
    if protocol == "raw_predicted":
        return predicted[:top_k]
    if protocol != "mandatory_skillless_fallback":
        raise ValueError(f"unknown top-k protocol: {protocol}")
    mandatory = next(
        (
            record
            for record in predicted
            if str(record["setup"]["setup_id"]).endswith(
                MANDATORY_BASELINE_SUFFIX
            )
        ),
        None,
    )
    if mandatory is None:
        raise ValueError("mandatory multistep skillless setup is missing")
    alternatives = [
        record
        for record in predicted
        if record["setup"]["setup_id"] != mandatory["setup"]["setup_id"]
    ]
    return [mandatory, *alternatives[: max(top_k - 1, 0)]]


def evaluate_topk_scores(
    records: list[dict[str, Any]],
    scores: dict[str, float],
    *,
    split: str,
    router: str,
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-problem top-k selections and full predicted rankings."""

    selection_rows = []
    ranking_rows = []
    for problem, outcomes in sorted(
        _outcome_groups(records, split=split).items()
    ):
        predicted = sorted(
            outcomes,
            key=lambda record: (
                scores[_record_id(record)],
                record["setup"]["setup_fingerprint"],
            ),
        )
        valid = [record for record in outcomes if record["labels"]["valid"]]
        if not valid:
            raise ValueError(f"{problem}: no valid setup outcome")
        actual_best = min(
            valid,
            key=lambda record: (
                record["labels"]["latency_cycles"],
                record["setup"]["setup_fingerprint"],
            ),
        )
        actual_best_id = str(actual_best["setup"]["setup_id"])
        for predicted_rank, record in enumerate(predicted, start=1):
            ranking_rows.append(
                {
                    "fold": fold,
                    "router": router,
                    "router_label": DISPLAY_NAMES.get(router, router),
                    "problem": problem,
                    "benchmark_lineage": record["benchmark_lineage"],
                    "setup_id": record["setup"]["setup_id"],
                    "strategy": record["setup"]["strategy"],
                    "skill_scope": record["setup"]["skill_scope"],
                    "predicted_rank": predicted_rank,
                    "score": float(scores[_record_id(record)]),
                    "actual_valid": bool(record["labels"]["valid"]),
                    "actual_cycles": record["labels"]["latency_cycles"],
                    "actual_regret": record["labels"]["regret"],
                    "actual_best_setup_id": actual_best_id,
                }
            )
        for protocol in PROTOCOLS:
            for top_k in TOP_K_VALUES:
                selected = _selected_candidates(
                    predicted,
                    top_k=top_k,
                    protocol=protocol,
                )
                selected_ids = [
                    str(record["setup"]["setup_id"]) for record in selected
                ]
                valid_selected = [
                    record
                    for record in selected
                    if record["labels"]["valid"]
                ]
                selected_winner = (
                    min(
                        valid_selected,
                        key=lambda record: (
                            record["labels"]["latency_cycles"],
                            record["setup"]["setup_fingerprint"],
                        ),
                    )
                    if valid_selected
                    else None
                )
                regret = (
                    float(selected_winner["labels"]["regret"])
                    if selected_winner is not None
                    else INVALID_REGRET_PENALTY
                )
                selection_rows.append(
                    {
                        "fold": fold,
                        "router": router,
                        "router_label": DISPLAY_NAMES.get(router, router),
                        "problem": problem,
                        "benchmark_lineage": actual_best[
                            "benchmark_lineage"
                        ],
                        "protocol": protocol,
                        "top_k": top_k,
                        "selected_setup_ids": json.dumps(selected_ids),
                        "selected_winner_setup_id": (
                            selected_winner["setup"]["setup_id"]
                            if selected_winner is not None
                            else ""
                        ),
                        "selected_winner_cycles": (
                            selected_winner["labels"]["latency_cycles"]
                            if selected_winner is not None
                            else None
                        ),
                        "selected_winner_regret": regret,
                        "exact_best_in_top_k": actual_best_id in selected_ids,
                        "within_5pct_of_best": regret <= 1.05,
                        "at_least_one_valid": bool(valid_selected),
                        "selected_candidate_validity": (
                            sum(
                                bool(record["labels"]["valid"])
                                for record in selected
                            )
                            / len(selected)
                        ),
                        "actual_best_setup_id": actual_best_id,
                        "actual_best_cycles": actual_best["labels"][
                            "latency_cycles"
                        ],
                    }
                )
    return selection_rows, ranking_rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(row["router"], row["protocol"], row["top_k"])].append(row)
    output = []
    for (router, protocol, top_k), values in sorted(grouped.items()):
        count = len(values)
        exact_count = sum(bool(row["exact_best_in_top_k"]) for row in values)
        within_count = sum(bool(row["within_5pct_of_best"]) for row in values)
        exact_low, exact_high = _wilson_interval(exact_count, count)
        within_low, within_high = _wilson_interval(within_count, count)
        regrets = [
            float(row["selected_winner_regret"]) for row in values
        ]
        output.append(
            {
                "router": router,
                "router_label": DISPLAY_NAMES.get(router, router),
                "protocol": protocol,
                "top_k": top_k,
                "benchmark_count": count,
                "exact_best_count": exact_count,
                "top_k_exact_accuracy": exact_count / count,
                "top_k_exact_wilson_low": exact_low,
                "top_k_exact_wilson_high": exact_high,
                "within_5pct_count": within_count,
                "within_5pct_coverage": within_count / count,
                "within_5pct_wilson_low": within_low,
                "within_5pct_wilson_high": within_high,
                "selection_validity": sum(
                    bool(row["at_least_one_valid"]) for row in values
                )
                / count,
                "candidate_validity": float(
                    np.mean(
                        [
                            row["selected_candidate_validity"]
                            for row in values
                        ]
                    )
                ),
                "geomean_regret": geometric_mean(regrets),
                "p95_regret": float(np.percentile(regrets, 95)),
                "candidate_savings_vs_exhaustive": 1.0 - top_k / 10.0,
            }
        )
    return output


def _adaptive_committee_policy_scores(
    records: list[dict[str, Any]],
    committee_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Encode the current mandatory-baseline, diverse committee order."""

    output = {}
    for outcomes in _outcome_groups(records).values():
        mandatory = next(
            record
            for record in outcomes
            if str(record["setup"]["setup_id"]).endswith(
                MANDATORY_BASELINE_SUFFIX
            )
        )
        mandatory_id = str(mandatory["setup"]["setup_id"])
        alternatives_by_id = {
            str(record["setup"]["setup_id"]): record
            for record in outcomes
            if str(record["setup"]["setup_id"]) != mandatory_id
        }
        votes: dict[str, float] = defaultdict(float)
        for score_map in committee_scores.values():
            ordered = sorted(
                alternatives_by_id.values(),
                key=lambda record: (
                    score_map[_record_id(record)],
                    record["setup"]["setup_fingerprint"],
                ),
            )
            for rank, record in enumerate(ordered):
                votes[str(record["setup"]["setup_id"])] += 1.0 / (
                    rank + 1.0
                )
        voted = sorted(
            alternatives_by_id,
            key=lambda setup_id: (
                -votes[setup_id],
                alternatives_by_id[setup_id]["setup"][
                    "setup_fingerprint"
                ],
            ),
        )
        diverse = _diverse_voted_setups(
            alternatives_by_id=alternatives_by_id,
            voted=voted,
            count=len(voted),
        )
        policy_order = [mandatory_id, *diverse]
        for rank, setup_id in enumerate(policy_order):
            output[_record_id(
                mandatory
                if setup_id == mandatory_id
                else alternatives_by_id[setup_id]
            )] = float(rank)
    return output


def _score_fold(
    fold_structured: list[dict[str, Any]],
    fold_hybrid: list[dict[str, Any]],
    *,
    trees: int,
    jobs: int,
    ranknet_epochs: int,
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    scores["extratrees_absolute_structured"], _, _ = _fit_direct(
        fold_structured,
        target="absolute_log_cycles",
        trees=trees,
        jobs=jobs,
        compute_cv=False,
    )
    scores["extratrees_relative_hybrid32"], _, _ = _fit_direct(
        fold_hybrid,
        target="relative_log_cycles",
        trees=trees,
        jobs=jobs,
        compute_cv=False,
    )
    scores["extratrees_log_regret_hybrid32"], _, _ = _fit_direct(
        fold_hybrid,
        target="log_regret",
        trees=trees,
        jobs=jobs,
        compute_cv=False,
    )
    scores["extratrees_within5_hybrid32"], _, _ = _fit_within5(
        fold_hybrid,
        trees=trees,
        jobs=jobs,
        compute_cv=False,
    )
    scores["extratrees_pairwise_hybrid32"], _, _ = _fit_pairwise(
        fold_hybrid,
        trees=trees,
        jobs=jobs,
        compute_cv=False,
    )
    scores["embedding_retrieval_source_k1"], _ = _retrieval_scores(
        fold_hybrid,
        neighbors=1,
        include_phase_b=False,
    )
    scores["extratrees_rank_ensemble_v2"] = _rank_consensus(
        fold_hybrid,
        [
            scores["extratrees_log_regret_hybrid32"],
            scores["extratrees_within5_hybrid32"],
            scores["extratrees_pairwise_hybrid32"],
            scores["embedding_retrieval_source_k1"],
        ],
    )
    scores["adaptive_extratrees_committee_v2"] = (
        _adaptive_committee_policy_scores(
            fold_hybrid,
            {
                "log_regret": scores[
                    "extratrees_log_regret_hybrid32"
                ],
                "within5": scores["extratrees_within5_hybrid32"],
                "pairwise": scores["extratrees_pairwise_hybrid32"],
                "retrieval": scores[
                    "embedding_retrieval_source_k1"
                ],
            },
        )
    )
    for variant in VARIANTS:
        variant_scores, _ = _fit_variant(
            fold_hybrid,
            variant,
            ranknet_epochs=ranknet_epochs,
        )
        scores[variant.name] = variant_scores
    return scores


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_matrix(
    aggregate_rows: list[dict[str, Any]],
    *,
    protocol: str,
    metric: str,
    router_order: list[str],
) -> np.ndarray:
    lookup = {
        (row["router"], row["top_k"]): float(row[metric])
        for row in aggregate_rows
        if row["protocol"] == protocol
    }
    return np.asarray(
        [
            [lookup[(router, top_k)] for top_k in TOP_K_VALUES]
            for router in router_order
        ],
        dtype=float,
    )


def _heatmap(
    matrix: np.ndarray,
    *,
    router_order: list[str],
    title: str,
    colorbar_label: str,
    destination: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig, axis = plt.subplots(
        figsize=(10, max(6.5, len(router_order) * 0.58))
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="YlGn",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xticks(
        np.arange(len(TOP_K_VALUES)),
        [f"Top-{value}" for value in TOP_K_VALUES],
    )
    axis.set_yticks(
        np.arange(len(router_order)),
        [DISPLAY_NAMES.get(router, router) for router in router_order],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    axis.set_title(title)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=190)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def _regret_chart(
    aggregate_rows: list[dict[str, Any]],
    *,
    router_order: list[str],
    destination: Path,
) -> None:
    matrix = _metric_matrix(
        aggregate_rows,
        protocol="raw_predicted",
        metric="geomean_regret",
        router_order=router_order,
    )
    positions = np.arange(len(router_order))
    width = 0.23
    colors = ("#4C78A8", "#59A14F", "#E15759")
    fig, axis = plt.subplots(figsize=(15, 7))
    for index, top_k in enumerate(TOP_K_VALUES):
        axis.bar(
            positions + (index - 1) * width,
            matrix[:, index],
            width,
            label=f"Top-{top_k}",
            color=colors[index],
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_yscale("log")
    axis.set_ylabel("Geomean regret versus exhaustive best (log scale)")
    axis.set_xticks(
        positions,
        [DISPLAY_NAMES.get(router, router) for router in router_order],
        rotation=28,
        ha="right",
    )
    axis.set_title("Out-of-fold router regret by candidate budget")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=190)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def _report(
    aggregate_rows: list[dict[str, Any]],
    *,
    router_order: list[str],
) -> str:
    lookup = {
        (row["router"], row["protocol"], row["top_k"]): row
        for row in aggregate_rows
    }
    raw_lines = []
    fallback_lines = []
    best_lines = []
    raw_by_k = {
        top_k: [
            row
            for row in aggregate_rows
            if row["protocol"] == "raw_predicted"
            and row["top_k"] == top_k
        ]
        for top_k in TOP_K_VALUES
    }
    for top_k in TOP_K_VALUES:
        best_exact = min(
            raw_by_k[top_k],
            key=lambda row: (
                -row["top_k_exact_accuracy"],
                row["geomean_regret"],
                row["router"],
            ),
        )
        best_within = min(
            raw_by_k[top_k],
            key=lambda row: (
                -row["within_5pct_coverage"],
                row["geomean_regret"],
                row["router"],
            ),
        )
        best_lines.append(
            "| "
            f"Top-{top_k} | "
            f"{best_exact['router_label']} | "
            f"{best_exact['top_k_exact_accuracy']:.3f} "
            f"[{best_exact['top_k_exact_wilson_low']:.3f}, "
            f"{best_exact['top_k_exact_wilson_high']:.3f}] | "
            f"{best_within['router_label']} | "
            f"{best_within['within_5pct_coverage']:.3f} "
            f"[{best_within['within_5pct_wilson_low']:.3f}, "
            f"{best_within['within_5pct_wilson_high']:.3f}] | "
            f"{best_within['geomean_regret']:.3f} |"
        )
    for router in router_order:
        raw = [
            lookup[(router, "raw_predicted", top_k)]
            for top_k in TOP_K_VALUES
        ]
        fallback = [
            lookup[(router, "mandatory_skillless_fallback", top_k)]
            for top_k in TOP_K_VALUES
        ]
        raw_lines.append(
            "| "
            + DISPLAY_NAMES.get(router, router)
            + " | "
            + " | ".join(
                f"{row['top_k_exact_accuracy']:.3f}" for row in raw
            )
            + " | "
            + " | ".join(
                f"{row['within_5pct_coverage']:.3f}" for row in raw
            )
            + " |"
        )
        fallback_lines.append(
            "| "
            + DISPLAY_NAMES.get(router, router)
            + " | "
            + " | ".join(
                f"{row['within_5pct_coverage']:.3f}"
                for row in fallback
            )
            + " | "
            + " | ".join(
                f"{row['geomean_regret']:.3f}" for row in fallback
            )
            + " |"
        )
    return (
        "\n".join(
            [
                "# Corrected-v2 Router Top-K Architecture Comparison",
                "",
                "- Protocol: five-fold benchmark-grouped out-of-fold evaluation.",
                "- Development lineages: 19; each benchmark is predicted only by a model that did not train on it.",
                "- Inputs: reference-blind source, canonical Phase-B features, setup fingerprint, and frozen Qwen3-Embedding-0.6B vectors where applicable.",
                "- Top-k exact accuracy means the exhaustive best setup appears in the predicted set.",
                "- Within-5% coverage means synthesizing the selected set and retaining its best valid candidate lands within 5% of exhaustive.",
                "",
                "## Best Architecture By Budget",
                "",
                "| budget | best exact architecture | exact accuracy (95% CI) | best within-5% architecture | within-5% coverage (95% CI) | geomean regret |",
                "|---|---|---:|---|---:|---:|",
                *best_lines,
                "",
                "Confidence intervals are Wilson intervals over 19 out-of-fold development benchmarks.",
                "",
                "## Raw Predicted Top-K",
                "",
                "| router | exact@1 | exact@3 | exact@5 | within5@1 | within5@3 | within5@5 |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *raw_lines,
                "",
                "## Mandatory Skillless Fallback",
                "",
                "This deployment-shaped policy always spends one candidate on corrected-v2 multistep skillless.",
                "",
                "| router | within5@1 | within5@3 | within5@5 | regret@1 | regret@3 | regret@5 |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *fallback_lines,
                "",
                "## Interpretation",
                "",
                "Use raw top-k for architecture accuracy comparisons and the fallback table for deployable tournament cost. Top-1 fallback is the same skillless baseline for every router, so it is not evidence about model quality.",
                "",
                "The current adaptive committee row follows its actual mandatory-skillless and diverse-voting order. The rank-ensemble row uses the same four members without that deployment constraint and is the appropriate row for unconstrained ranking accuracy.",
                "",
                "The Qwen3-0.6B LoRA router is excluded from the primary table because no five-fold corrected-v2 LoRA retraining exists. Its four-kernel historical result is not directly comparable to these out-of-fold measurements.",
            ]
        )
        + "\n"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_records = _load(args.corpus)
    canonical_structured = _canonicalize_phase_b(base_records)
    canonical_hybrid, embedding_manifest = _canonical_hybrid_records(
        args.corpus,
        args.embedding_dir,
    )
    train_problems = sorted(
        {
            str(record["problem"])
            for record in canonical_structured
            if record["split"] == "train"
        }
    )
    splitter = GroupKFold(n_splits=5)
    held_out_folds = [
        {train_problems[index] for index in held_out_indices}
        for _, held_out_indices in splitter.split(
            np.zeros(len(train_problems)),
            groups=np.asarray(train_problems),
        )
    ]
    selection_rows = []
    ranking_rows = []
    fold_manifest = []
    for fold, held_out in enumerate(held_out_folds):
        fold_structured = _outer_fold_records(
            canonical_structured,
            held_out,
        )
        fold_hybrid = _outer_fold_records(canonical_hybrid, held_out)
        score_maps = _score_fold(
            fold_structured,
            fold_hybrid,
            trees=args.trees,
            jobs=args.jobs,
            ranknet_epochs=args.ranknet_epochs,
        )
        fold_manifest.append(
            {
                "fold": fold,
                "held_out_problems": sorted(held_out),
            }
        )
        for router, scores in sorted(score_maps.items()):
            selected, rankings = evaluate_topk_scores(
                fold_structured,
                scores,
                split="validation",
                router=router,
                fold=fold,
            )
            selection_rows.extend(selected)
            ranking_rows.extend(rankings)

    expected_problem_router_pairs = len(train_problems) * len(DISPLAY_NAMES)
    actual_problem_router_pairs = len(
        {
            (row["problem"], row["router"])
            for row in ranking_rows
        }
    )
    if actual_problem_router_pairs != expected_problem_router_pairs:
        raise ValueError(
            "incomplete out-of-fold predictions: "
            f"{actual_problem_router_pairs} != "
            f"{expected_problem_router_pairs}"
        )
    aggregate_rows = _aggregate(selection_rows)
    raw_top5 = {
        row["router"]: row
        for row in aggregate_rows
        if row["protocol"] == "raw_predicted" and row["top_k"] == 5
    }
    router_order = sorted(
        raw_top5,
        key=lambda router: (
            -raw_top5[router]["within_5pct_coverage"],
            -raw_top5[router]["top_k_exact_accuracy"],
            raw_top5[router]["geomean_regret"],
            router,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / "topk_metrics.csv"
    selections_csv = args.output_dir / "per_benchmark_topk.csv"
    rankings_csv = args.output_dir / "oof_setup_rankings.csv"
    _write_csv(metrics_csv, aggregate_rows)
    _write_csv(selections_csv, selection_rows)
    _write_csv(rankings_csv, ranking_rows)
    report_path = args.output_dir / "report.md"
    report_path.write_text(
        _report(aggregate_rows, router_order=router_order),
        encoding="utf-8",
    )
    exact_base = args.output_dir / "raw_topk_exact_accuracy"
    within_base = args.output_dir / "raw_topk_within5_coverage"
    fallback_base = args.output_dir / "fallback_topk_within5_coverage"
    regret_base = args.output_dir / "raw_topk_geomean_regret"
    _heatmap(
        _metric_matrix(
            aggregate_rows,
            protocol="raw_predicted",
            metric="top_k_exact_accuracy",
            router_order=router_order,
        ),
        router_order=router_order,
        title="Out-of-fold exact-best setup coverage",
        colorbar_label="Exact top-k accuracy",
        destination=exact_base,
        vmin=0.0,
        vmax=1.0,
    )
    _heatmap(
        _metric_matrix(
            aggregate_rows,
            protocol="raw_predicted",
            metric="within_5pct_coverage",
            router_order=router_order,
        ),
        router_order=router_order,
        title="Out-of-fold within-5% setup coverage",
        colorbar_label="Within-5% coverage",
        destination=within_base,
        vmin=0.0,
        vmax=1.0,
    )
    _heatmap(
        _metric_matrix(
            aggregate_rows,
            protocol="mandatory_skillless_fallback",
            metric="within_5pct_coverage",
            router_order=router_order,
        ),
        router_order=router_order,
        title="Within-5% coverage with mandatory skillless fallback",
        colorbar_label="Within-5% coverage",
        destination=fallback_base,
        vmin=0.0,
        vmax=1.0,
    )
    _regret_chart(
        aggregate_rows,
        router_order=router_order,
        destination=regret_base,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "methodology": {
            "protocol": "five-fold benchmark-grouped out-of-fold",
            "development_lineages": len(train_problems),
            "top_k_values": list(TOP_K_VALUES),
            "candidate_set_protocols": list(PROTOCOLS),
            "reference_metrics_as_inputs": False,
            "post_candidate_features_as_inputs": False,
            "canonical_phase_b_context": True,
            "small_lm_exclusion": (
                "no five-fold corrected-v2 LoRA retraining"
            ),
        },
        "corpus": {
            "path": str(args.corpus.resolve()),
            "sha256": _sha256(args.corpus),
        },
        "embedding": embedding_manifest,
        "training": {
            "trees_per_fold_model": args.trees,
            "ranknet_epochs": args.ranknet_epochs,
            "jobs": args.jobs,
        },
        "fold_manifest": fold_manifest,
        "router_order": router_order,
        "metrics": aggregate_rows,
    }
    metrics_json = args.output_dir / "metrics.json"
    metrics_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [
        metrics_csv,
        selections_csv,
        rankings_csv,
        report_path,
        metrics_json,
        *[
            base.with_suffix(suffix)
            for base in (exact_base, within_base, fallback_base, regret_base)
            for suffix in (".png", ".pdf")
        ],
    ]
    manifest_path = args.output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "c2hls.setup-router-topk-artifact-manifest.v1"
                ),
                "created_at": _utc_now(),
                "artifacts": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
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
    return {
        "output_dir": str(args.output_dir),
        "development_lineages": len(train_problems),
        "routers": len(router_order),
        "top_k_values": list(TOP_K_VALUES),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--ranknet-epochs", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
