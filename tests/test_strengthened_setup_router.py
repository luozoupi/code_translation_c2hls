from __future__ import annotations

import pytest

from scripts.train_strengthened_setup_router import (
    INFERENCE_BUNDLE_SCHEMA_VERSION,
    _build_retrieval_index,
    _canonicalize_phase_b,
    _committee_candidate_budget,
    _diverse_voted_setups,
    _first_wins,
    _pair_training_records,
    _selection_key,
    _validate_inference_bundle,
    _within5_training_weights,
)


def _outcome(
    *,
    problem: str,
    setup_id: str,
    strategy: str,
    cycles: float | None,
    phase_b_cycles: float,
) -> dict:
    valid = cycles is not None
    return {
        "record_kind": "setup_outcome",
        "problem": problem,
        "benchmark": f"hlsfactory_{problem}",
        "benchmark_lineage": f"polybench:{problem}",
        "split": "train",
        "setup": {
            "setup_id": setup_id,
            "setup_fingerprint": setup_id,
            "behavior_version": "legacy_v1",
            "strategy": strategy,
            "skill_scope": "skillless",
            "router_version": 0,
        },
        "features": {
            "model_id": "teacher",
            "setup_strategy": strategy,
            "setup_skill_scope": "skillless",
            "setup_behavior_version": "legacy_v1",
            "setup_router_version": 0.0,
            "phase_b_latency_cycles": phase_b_cycles,
            "phase_b_log_latency_cycles": 1.0,
            "source_loop_count": 3.0,
        },
        "labels": {
            "valid": valid,
            "latency_cycles": cycles,
            "log_latency_cycles": 2.0 if valid else None,
            "regret": 1.0 if valid else None,
        },
        "eligibility": {
            "ranking_model": True,
            "feasibility_model": True,
        },
        "provenance": {
            "dedup_key_sha256": f"{problem}:{setup_id}",
        },
    }


def test_canonicalization_uses_mandatory_phase_b_for_every_setup() -> None:
    flash = _outcome(
        problem="demo",
        setup_id="legacy_v1:flash:skillless",
        strategy="flash",
        cycles=80.0,
        phase_b_cycles=900.0,
    )
    mandatory = _outcome(
        problem="demo",
        setup_id="legacy_v1:multistep:skillless",
        strategy="multistep",
        cycles=100.0,
        phase_b_cycles=1000.0,
    )
    canonical = _canonicalize_phase_b([flash, mandatory])
    by_setup = {record["setup"]["setup_id"]: record for record in canonical}
    assert (
        by_setup["legacy_v1:flash:skillless"]["features"][
            "phase_b_latency_cycles"
        ]
        == 1000.0
    )
    assert (
        by_setup["legacy_v1:flash:skillless"]["features"][
            "setup_strategy"
        ]
        == "flash"
    )


def test_pairwise_label_is_antisymmetric_and_validity_first() -> None:
    fast = _outcome(
        problem="demo",
        setup_id="legacy_v1:flash:skillless",
        strategy="flash",
        cycles=80.0,
        phase_b_cycles=1000.0,
    )
    slow = _outcome(
        problem="demo",
        setup_id="legacy_v1:multistep:skillless",
        strategy="multistep",
        cycles=100.0,
        phase_b_cycles=1000.0,
    )
    invalid = _outcome(
        problem="demo",
        setup_id="legacy_v1:flash:matched",
        strategy="flash",
        cycles=None,
        phase_b_cycles=1000.0,
    )
    assert _first_wins(fast, slow) is True
    assert _first_wins(slow, fast) is False
    assert _first_wins(slow, invalid) is True
    assert _first_wins(invalid, slow) is False
    assert _first_wins(invalid, invalid) is None


def test_pair_training_adds_both_orientations() -> None:
    fast = _outcome(
        problem="demo",
        setup_id="legacy_v1:flash:skillless",
        strategy="flash",
        cycles=80.0,
        phase_b_cycles=1000.0,
    )
    slow = _outcome(
        problem="demo",
        setup_id="legacy_v1:multistep:skillless",
        strategy="multistep",
        cycles=100.0,
        phase_b_cycles=1000.0,
    )
    pairs = _pair_training_records([fast, slow])
    assert len(pairs) == 2
    assert sorted(pair["label"] for pair in pairs) == [False, True]


def test_selection_prefers_candidate_savings_after_equal_quality() -> None:
    def metrics(savings: float) -> dict:
        return {
            "top_1_accuracy": 0.25,
            "raw_top_3_oracle_coverage": 0.75,
            "learned_top_k": {
                "validity": 1.0,
                "within_5pct_coverage": 0.75,
                "oracle_coverage": 0.75,
                "geomean_regret": 1.1,
                "candidate_savings_vs_exhaustive": savings,
            },
        }

    assert _selection_key(
        metrics(0.55),
        "adaptive",
    ) < _selection_key(metrics(0.5), "fixed-five")


def test_retrieval_index_contains_training_vectors_and_setup_costs() -> None:
    records = []
    for problem, embedding in (
        ("first", (1.0, 0.0)),
        ("second", (0.0, 1.0)),
    ):
        for strategy, cycles in (("flash", 80.0), ("multistep", 100.0)):
            record = _outcome(
                problem=problem,
                setup_id=f"legacy_v1:{strategy}:skillless",
                strategy=strategy,
                cycles=cycles,
                phase_b_cycles=1000.0,
            )
            record["features"].update(
                {
                    "transformer_source_000": embedding[0],
                    "transformer_source_001": embedding[1],
                }
            )
            records.append(record)

    index = _build_retrieval_index(
        records,
        include_phase_b=False,
    )
    assert index["train_lineages"] == ["first", "second"]
    assert index["feature_names"] == [
        "transformer_source_000",
        "transformer_source_001",
    ]
    assert sorted(index["setup_ids"]) == [
        "legacy_v1:flash:skillless",
        "legacy_v1:multistep:skillless",
    ]
    assert all(len(vector) == 2 for vector in index["train_vectors"].values())


def test_inference_bundle_requires_every_selected_committee_member() -> None:
    bundle = {
        "schema_version": INFERENCE_BUNDLE_SCHEMA_VERSION,
        "selected_strengthened": "committee_disagreement_adaptive",
        "models": {
            "relative_hybrid32_canonical": {"kind": "direct"},
            "pairwise_hybrid32_canonical": {"kind": "pairwise"},
        },
        "retrieval_indexes": {
            "retrieval_source_k1": {
                "feature_names": ["transformer_source_000"],
                "train_vectors": {"train": [1.0]},
                "setup_costs": {"train": {"setup": 0.0}},
                "setup_priors": {"setup": 0.0},
            }
        },
        "committee_policy": {
            "members": [
                "pairwise_hybrid32_canonical",
                "relative_hybrid32_canonical",
                "retrieval_source_k1",
            ],
            "mandatory_setup": "multistep skillless",
            "maximum_candidate_budget": 8,
        },
    }
    _validate_inference_bundle(bundle)

    del bundle["models"]["pairwise_hybrid32_canonical"]
    with pytest.raises(ValueError, match="missing policy members"):
        _validate_inference_bundle(bundle)


def test_uncertain_committee_uses_three_five_or_eight_candidates() -> None:
    assert _committee_candidate_budget(
        ["a", "a", "a"],
        adaptive=True,
        maximum_budget=8,
    ) == 3
    assert _committee_candidate_budget(
        ["a", "a", "b"],
        adaptive=True,
        maximum_budget=8,
    ) == 5
    assert _committee_candidate_budget(
        ["a", "b", "c"],
        adaptive=True,
        maximum_budget=8,
    ) == 8


def test_diverse_selection_retains_top_two_then_expands_modes() -> None:
    def record(strategy: str, scope: str) -> dict:
        return {
            "setup": {
                "strategy": strategy,
                "skill_scope": scope,
                "setup_fingerprint": f"{strategy}:{scope}",
            }
        }

    alternatives = {
        "a": record("flash", "matched"),
        "b": record("flash", "best"),
        "c": record("flash", "all"),
        "d": record("multistep", "exhaustive"),
    }
    selected = _diverse_voted_setups(
        alternatives_by_id=alternatives,
        voted=["a", "b", "c", "d"],
        count=3,
    )
    assert selected[:2] == ["a", "b"]
    assert selected[2] == "d"


def test_within5_weights_emphasize_rare_winners_and_hard_negatives() -> None:
    common = _outcome(
        problem="a",
        setup_id="legacy_v1:flash:skillless",
        strategy="flash",
        cycles=10.0,
        phase_b_cycles=100.0,
    )
    common["labels"].update(
        {"within_5pct_of_best": True, "regret": 1.0}
    )
    common_two = {
        **common,
        "problem": "b",
        "provenance": {"dedup_key_sha256": "b:common"},
    }
    rare = _outcome(
        problem="c",
        setup_id="legacy_v1:multistep:skillless",
        strategy="multistep",
        cycles=10.0,
        phase_b_cycles=100.0,
    )
    rare["labels"].update({"within_5pct_of_best": True, "regret": 1.0})
    hard = _outcome(
        problem="d",
        setup_id="legacy_v1:flash:matched",
        strategy="flash",
        cycles=11.0,
        phase_b_cycles=100.0,
    )
    hard["labels"].update(
        {"within_5pct_of_best": False, "regret": 1.1}
    )

    weights = _within5_training_weights(
        [common, common_two, rare, hard]
    )
    assert weights[2] > weights[0]
    assert weights[3] == 2.0
