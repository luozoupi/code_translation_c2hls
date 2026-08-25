from __future__ import annotations

import pytest

from scripts.evaluate_setup_router_topk_architectures import (
    _aggregate,
    _selected_candidates,
    evaluate_topk_scores,
)
from scripts.train_strengthened_setup_router import _record_id


def _outcome(
    setup_id: str,
    *,
    cycles: int,
    regret: float,
    valid: bool = True,
) -> dict:
    return {
        "record_kind": "setup_outcome",
        "split": "validation",
        "problem": "synthetic",
        "benchmark_lineage": "synthetic:held-out",
        "features": {"setup_behavior_version": "corrected_v2"},
        "eligibility": {"ranking_model": True},
        "provenance": {"dedup_key_sha256": setup_id},
        "setup": {
            "setup_id": setup_id,
            "setup_fingerprint": setup_id,
            "strategy": setup_id.split(":")[1],
            "skill_scope": setup_id.split(":")[2],
        },
        "labels": {
            "valid": valid,
            "latency_cycles": cycles,
            "regret": regret,
        },
    }


def test_fixed_topk_reports_exact_best_and_near_best_separately() -> None:
    skillless = _outcome(
        "corrected_v2:multistep:skillless",
        cycles=200,
        regret=2.0,
    )
    near_best = _outcome(
        "corrected_v2:flash:matched_positive",
        cycles=104,
        regret=1.04,
    )
    best = _outcome(
        "corrected_v2:multistep:smart_best_fit_v2",
        cycles=100,
        regret=1.0,
    )
    invalid = _outcome(
        "corrected_v2:flash:smart_exhaustive_v2",
        cycles=90,
        regret=1.0,
        valid=False,
    )
    records = [skillless, near_best, best, invalid]
    scores = {
        _record_id(record): score
        for record, score in zip(records, (0.0, 1.0, 2.0, 3.0))
    }

    rows, rankings = evaluate_topk_scores(
        records,
        scores,
        split="validation",
        router="synthetic_router",
        fold=0,
    )

    assert len(rankings) == 4
    raw = {
        row["top_k"]: row
        for row in rows
        if row["protocol"] == "raw_predicted"
    }
    assert raw[1]["exact_best_in_top_k"] is False
    assert raw[1]["within_5pct_of_best"] is False
    assert raw[3]["exact_best_in_top_k"] is True
    assert raw[3]["within_5pct_of_best"] is True
    assert raw[3]["selected_winner_setup_id"] == best["setup"]["setup_id"]


def test_mandatory_fallback_spends_one_slot_on_skillless() -> None:
    predicted = [
        _outcome(
            "corrected_v2:flash:matched_positive",
            cycles=100,
            regret=1.0,
        ),
        _outcome(
            "corrected_v2:flash:smart_best_fit_v2",
            cycles=110,
            regret=1.1,
        ),
        _outcome(
            "corrected_v2:multistep:skillless",
            cycles=200,
            regret=2.0,
        ),
    ]

    selected = _selected_candidates(
        predicted,
        top_k=3,
        protocol="mandatory_skillless_fallback",
    )

    assert len(selected) == 3
    assert selected[0]["setup"]["setup_id"].endswith(
        ":multistep:skillless"
    )


def test_aggregate_reports_candidate_savings_and_geomean_regret() -> None:
    rows = [
        {
            "router": "router",
            "router_label": "Router",
            "protocol": "raw_predicted",
            "top_k": 3,
            "exact_best_in_top_k": True,
            "within_5pct_of_best": True,
            "at_least_one_valid": True,
            "selected_candidate_validity": 1.0,
            "selected_winner_regret": 1.0,
        },
        {
            "router": "router",
            "router_label": "Router",
            "protocol": "raw_predicted",
            "top_k": 3,
            "exact_best_in_top_k": False,
            "within_5pct_of_best": False,
            "at_least_one_valid": True,
            "selected_candidate_validity": 2.0 / 3.0,
            "selected_winner_regret": 1.21,
        },
    ]

    [summary] = _aggregate(rows)

    assert summary["top_k_exact_accuracy"] == 0.5
    assert summary["within_5pct_coverage"] == 0.5
    assert summary["geomean_regret"] == pytest.approx(1.1)
    assert summary["candidate_savings_vs_exhaustive"] == 0.7
