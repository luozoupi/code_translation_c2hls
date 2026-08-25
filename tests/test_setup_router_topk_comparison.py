from __future__ import annotations

from scripts.evaluate_setup_router_topk_architectures import (
    _adaptive_committee_policy_scores,
    _aggregate,
    evaluate_topk_scores,
)


def _record(
    setup_id: str,
    cycles: int,
    regret: float,
    fingerprint: str,
) -> dict:
    return {
        "record_kind": "setup_outcome",
        "split": "validation",
        "problem": "toy",
        "benchmark_lineage": "toy:one",
        "setup": {
            "setup_id": setup_id,
            "setup_fingerprint": fingerprint,
            "strategy": setup_id.split(":")[-2],
            "skill_scope": setup_id.split(":")[-1],
            "behavior_version": "corrected_v2",
        },
        "labels": {
            "valid": True,
            "latency_cycles": cycles,
            "regret": regret,
        },
        "features": {"setup_behavior_version": "corrected_v2"},
        "eligibility": {"ranking_model": True},
        "provenance": {"dedup_key_sha256": fingerprint},
    }


def test_topk_evaluator_separates_raw_and_skillless_fallback() -> None:
    records = [
        _record(
            "corrected_v2:multistep:skillless",
            200,
            2.0,
            "a",
        ),
        _record(
            "corrected_v2:flash:matched_positive",
            100,
            1.0,
            "b",
        ),
        _record(
            "corrected_v2:multistep:smart_best_fit_v2",
            104,
            1.04,
            "c",
        ),
    ]
    scores = {"a": 0.3, "b": 0.2, "c": 0.1}
    selections, rankings = evaluate_topk_scores(
        records,
        scores,
        split="validation",
        router="toy_router",
        fold=0,
    )

    assert len(rankings) == 3
    raw_top1 = next(
        row
        for row in selections
        if row["protocol"] == "raw_predicted" and row["top_k"] == 1
    )
    fallback_top1 = next(
        row
        for row in selections
        if row["protocol"] == "mandatory_skillless_fallback"
        and row["top_k"] == 1
    )
    raw_top3 = next(
        row
        for row in selections
        if row["protocol"] == "raw_predicted" and row["top_k"] == 3
    )
    assert raw_top1["exact_best_in_top_k"] is False
    assert raw_top1["within_5pct_of_best"] is True
    assert fallback_top1["selected_winner_setup_id"].endswith(
        ":skillless"
    )
    assert fallback_top1["within_5pct_of_best"] is False
    assert raw_top3["exact_best_in_top_k"] is True


def test_topk_aggregate_reports_exact_and_within5_separately() -> None:
    rows = [
        {
            "router": "toy",
            "router_label": "Toy",
            "protocol": "raw_predicted",
            "top_k": 1,
            "exact_best_in_top_k": False,
            "within_5pct_of_best": True,
            "at_least_one_valid": True,
            "selected_candidate_validity": 1.0,
            "selected_winner_regret": 1.04,
        },
        {
            "router": "toy",
            "router_label": "Toy",
            "protocol": "raw_predicted",
            "top_k": 1,
            "exact_best_in_top_k": True,
            "within_5pct_of_best": True,
            "at_least_one_valid": True,
            "selected_candidate_validity": 1.0,
            "selected_winner_regret": 1.0,
        },
    ]
    [metrics] = _aggregate(rows)
    assert metrics["top_k_exact_accuracy"] == 0.5
    assert metrics["within_5pct_coverage"] == 1.0
    assert metrics["geomean_regret"] > 1.0


def test_adaptive_committee_order_starts_with_mandatory_skillless() -> None:
    records = [
        _record(
            "corrected_v2:multistep:skillless",
            200,
            2.0,
            "a",
        ),
        _record(
            "corrected_v2:flash:matched_positive",
            100,
            1.0,
            "b",
        ),
        _record(
            "corrected_v2:multistep:smart_best_fit_v2",
            104,
            1.04,
            "c",
        ),
    ]
    scores = _adaptive_committee_policy_scores(
        records,
        {
            "first": {"a": 0.9, "b": 0.1, "c": 0.2},
            "second": {"a": 0.8, "b": 0.2, "c": 0.1},
        },
    )
    assert scores["a"] == 0.0
    assert sorted(scores, key=scores.get)[0] == "a"
