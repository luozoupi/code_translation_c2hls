from __future__ import annotations

from scripts.train_small_lm_setup_router_oof import (
    _scores_from_pair_margins,
)


def _outcome(setup_id: str, record_id: str) -> dict:
    return {
        "problem": "example",
        "split": "validation",
        "setup": {"setup_id": setup_id},
        "provenance": {"dedup_key_sha256": record_id},
    }


def test_pairwise_margins_produce_complete_lower_is_better_scores() -> None:
    outcomes = [
        _outcome("setup_a", "record_a"),
        _outcome("setup_b", "record_b"),
        _outcome("setup_c", "record_c"),
    ]
    predictions = [
        {
            "problem": "example",
            "setup_a": "setup_a",
            "setup_b": "setup_b",
            "margin_a_minus_b": 2.0,
        },
        {
            "problem": "example",
            "setup_a": "setup_a",
            "setup_b": "setup_c",
            "margin_a_minus_b": 1.0,
        },
        {
            "problem": "example",
            "setup_a": "setup_b",
            "setup_b": "setup_c",
            "margin_a_minus_b": 0.5,
        },
    ]

    scores = _scores_from_pair_margins(predictions, outcomes)

    assert set(scores) == {"record_a", "record_b", "record_c"}
    assert scores["record_a"] < scores["record_b"] < scores["record_c"]
