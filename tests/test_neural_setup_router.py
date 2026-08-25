from __future__ import annotations

import torch

from scripts.train_neural_setup_router import RankNet, _ranknet_pairs


def _record(
    *,
    problem: str,
    setup_id: str,
    cycles: float | None,
) -> dict:
    return {
        "problem": problem,
        "setup": {"setup_id": setup_id},
        "labels": {
            "valid": cycles is not None,
            "latency_cycles": cycles,
        },
        "provenance": {
            "dedup_key_sha256": f"{problem}:{setup_id}",
        },
    }


def test_ranknet_pairs_are_group_local_and_validity_first() -> None:
    records = [
        _record(problem="first", setup_id="fast", cycles=10),
        _record(problem="first", setup_id="slow", cycles=20),
        _record(problem="first", setup_id="invalid", cycles=None),
        _record(problem="second", setup_id="a", cycles=5),
        _record(problem="second", setup_id="b", cycles=5),
    ]
    ordered, pairs = _ranknet_pairs(records)
    by_id = {
        record["provenance"]["dedup_key_sha256"]: index
        for index, record in enumerate(ordered)
    }
    assert (by_id["first:fast"], by_id["first:slow"]) in pairs
    assert (by_id["first:fast"], by_id["first:invalid"]) in pairs
    assert (by_id["first:slow"], by_id["first:invalid"]) in pairs
    assert all(
        ordered[winner]["problem"] == ordered[loser]["problem"]
        for winner, loser in pairs
    )
    assert not any(
        {
            ordered[winner]["setup"]["setup_id"],
            ordered[loser]["setup"]["setup_id"],
        }
        == {"a", "b"}
        for winner, loser in pairs
    )


def test_ranknet_returns_one_score_per_candidate() -> None:
    model = RankNet(7, (8, 4))
    scores = model(torch.zeros((3, 7), dtype=torch.float32))
    assert tuple(scores.shape) == (3,)
