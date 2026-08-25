from setup_router import (
    CORRECTED_VERSION,
    LEGACY_VERSION,
    adaptive_candidate_budget,
    resolve_policy_setups,
    select_tournament_winner,
    setup_registry,
)


def _candidate(setup_id: str, fingerprint: str, cycles: int) -> dict:
    return {
        "setup_id": setup_id,
        "setup_fingerprint": fingerprint,
        "csim": {"ran": True, "passed": True},
        "final_report": {
            "latency_cycles": cycles,
            "slack_ns": 0.2,
            "bram": 1,
            "dsp": 1,
            "ff": 100,
            "lut": 100,
            "uram": 0,
        },
        "hls_code": f"void workload() {{ /* {setup_id} */ }}",
    }


def test_registries_have_ten_distinct_versioned_setups() -> None:
    legacy = setup_registry(LEGACY_VERSION)
    corrected = setup_registry(CORRECTED_VERSION)

    assert len(legacy) == 10
    assert len(corrected) == 10
    assert {item.fingerprint for item in legacy}.isdisjoint(
        {item.fingerprint for item in corrected}
    )


def test_learned_top_k_always_includes_multistep_skillless() -> None:
    predicted = [
        f"{CORRECTED_VERSION}:flash:smart_best_fit_v2",
        f"{CORRECTED_VERSION}:flash:all_positive_preconditions",
        f"{CORRECTED_VERSION}:multistep:matched_positive",
    ]
    selected = resolve_policy_setups(
        policy="learned_top_k",
        predicted_setup_ids=predicted,
    )

    assert len(selected) == 3
    assert selected[0].setup_id == (
        f"{CORRECTED_VERSION}:multistep:skillless"
    )


def test_winner_selection_is_feasible_and_deterministic() -> None:
    candidates = [
        _candidate("slow", "b", 200),
        _candidate("fast-z", "z", 100),
        _candidate("fast-a", "a", 100),
        {
            **_candidate("invalid", "0", 1),
            "csim": {"ran": True, "passed": False},
        },
    ]
    first = select_tournament_winner(candidates)
    second = select_tournament_winner(reversed(candidates))

    assert first["winner"]["setup_id"] == "fast-a"
    assert second["winner"]["setup_id"] == "fast-a"
    assert first["winner_explanation"]["winner_latency_cycles"] == 100


def test_adaptive_policy_expands_uncertain_predictions_with_diversity() -> None:
    predicted = [
        f"{CORRECTED_VERSION}:flash:matched_positive",
        f"{CORRECTED_VERSION}:flash:smart_best_fit_v2",
        f"{CORRECTED_VERSION}:flash:all_positive_preconditions",
        f"{CORRECTED_VERSION}:multistep:smart_exhaustive_v2",
        f"{CORRECTED_VERSION}:multistep:matched_positive",
    ]
    selected = resolve_policy_setups(
        policy="adaptive_diverse_top_k",
        predicted_setup_ids=predicted,
        prediction_metadata={"recommended_candidate_budget": 5},
    )

    assert len(selected) == 5
    assert selected[0].setup_id.endswith(":multistep:skillless")
    assert {item.strategy for item in selected} == {"flash", "multistep"}
    assert len({item.skill_scope for item in selected}) >= 4
    assert adaptive_candidate_budget(None) == 8


def test_winner_explanation_reports_skillless_and_phase_b_evidence() -> None:
    skillless = _candidate(
        f"{CORRECTED_VERSION}:multistep:skillless",
        "skillless",
        200,
    )
    skillless["setup"] = {
        "setup_id": skillless["setup_id"],
        "strategy": "multistep",
        "skill_scope": "skillless",
    }
    skilled = _candidate(
        f"{CORRECTED_VERSION}:multistep:matched_positive",
        "skilled",
        100,
    )
    skilled["setup"] = {
        "setup_id": skilled["setup_id"],
        "strategy": "multistep",
        "skill_scope": "matched_positive",
    }
    skilled["baseline_report"] = {
        **skilled["final_report"],
        "latency_cycles": 1000,
        "feedback": {
            "bottlenecks": [{"kind": "ii_target_miss"}],
        },
    }
    skilled["best_so_far_history"] = [
        {"step_name": "baseline", "report": {"latency_cycles": 1000}},
        {"step_name": "pipeline", "report": {"latency_cycles": 100}},
    ]
    skilled["generated_step_history"] = [
        {
            "step_name": "pipeline",
            "skill_prompt": {
                "routed_skill_ids": ["pipeline"],
                "rendered_skill_ids": ["pipeline"],
                "declared_applied_skill_ids": ["pipeline"],
                "verified_applied_skill_ids": ["pipeline"],
                "synthesized_candidate_skill_ids": ["pipeline"],
            },
        }
    ]

    result = select_tournament_winner([skillless, skilled])
    evidence = result["winner_explanation"]["mode_fit_evidence"]
    assert evidence["best_skillless"]["winner_speedup"] == 2.0
    assert evidence["phase_b"]["latency_cycles"] == 1000
    assert evidence["skill_evidence"]["verified_applied_skill_ids"] == [
        "pipeline"
    ]
