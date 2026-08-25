from __future__ import annotations

import json
import os
import re
from pathlib import Path

import c2hls
from run_agentic_sweep import (
    _apply_post_profile_cosim_policy,
    _selected_skill_modes,
)
from skill_library import (
    TIER_AVOID,
    Skill,
    SkillLibrary,
    load_frozen_library,
    render_skill_set_for_prompt,
)
from smart_skill_router import render_empty_skill_source, route_smart_skills


REPO = Path(__file__).resolve().parents[1]


def _library(tmp_path: Path) -> SkillLibrary:
    library = SkillLibrary(tmp_path / "mutable.json")
    library.add(
        Skill(
            id="pipeline-ii-repair",
            pattern="pipeline loop misses its initiation interval",
            strategy="pipeline the loop and resolve recurrence or memory ports",
            confidence="high",
            bottleneck_kinds=["ii_target_miss"],
            tags=["pipeline", "ii"],
        )
    )
    library.add(
        Skill(
            id="tile-capacity-repair",
            pattern="working set exceeds local memory capacity",
            strategy="tile the array working set",
            confidence="medium",
            bottleneck_kinds=["resource_pressure"],
            tags=["tiling", "memory"],
        )
    )
    library.add(
        Skill(
            id="avoid-forced-ii",
            pattern="forcing II one is harmful",
            strategy="retain the achieved interval",
            confidence="avoid",
            bottleneck_kinds=["ii_target_miss"],
            tags=["pipeline", "ii"],
        )
    )
    return library


def test_focused_router_selects_supported_positive_skill(tmp_path: Path) -> None:
    route = route_smart_skills(
        _library(tmp_path),
        scope="smart_best_fit",
        step_name="pipeline",
        current_code="for (int i = 0; i < n; ++i) out[i] = in[i];",
        synth_report={
            "feedback": {
                "bottlenecks": [{"kind": "ii_target_miss"}],
            }
        },
        max_skills=3,
        min_score=4.0,
    )

    assert route.audit["search_policy"] == "focused_candidate_pool"
    assert route.audit["selected_skill_ids"][0] == "pipeline-ii-repair"
    assert "avoid-forced-ii" not in route.audit["selected_skill_ids"]
    assert all(skill.confidence != "avoid" for skill in route.selected)


def test_exhaustive_router_can_return_no_skill(tmp_path: Path) -> None:
    route = route_smart_skills(
        _library(tmp_path),
        scope="smart_exhaustive",
        step_name="opaque_transform",
        current_code="unsigned checksum = 7;",
        synth_report={},
        max_skills=3,
        min_score=4.0,
    )

    assert route.audit["search_policy"] == "all_applicable_positive_skills"
    assert route.audit["catalog_positive_applicable_count"] == 2
    assert route.audit["candidate_count"] == 2
    assert route.audit["no_match"] is True
    assert route.selected == []


def test_v2_router_requires_code_or_report_evidence(
    tmp_path: Path,
) -> None:
    route = route_smart_skills(
        _library(tmp_path),
        scope="smart_best_fit_v2",
        step_name="pipeline",
        current_code="unsigned checksum = 7;",
        synth_report={},
        requested_skill_id="pipeline-ii-repair",
        max_skills=3,
        min_score=4.0,
    )

    assert route.selected == []
    assert route.audit["strict_evidence"] is True
    assert route.audit["no_match"] is True


def test_v2_double_buffer_route_accepts_load_compute_serialization(
    tmp_path: Path,
) -> None:
    library = _library(tmp_path)
    library.add(
        Skill(
            id="doublebuffer-load-compute",
            pattern="load and compute phases serialize",
            strategy="use ping-pong local buffers with dataflow",
            confidence="high",
            bottleneck_kinds=["load_compute_serialize"],
            tags=["doublebuffer", "ping-pong-buffer", "dataflow"],
        )
    )
    route = route_smart_skills(
        library,
        scope="smart_exhaustive_v2",
        step_name="flash",
        current_code="""
        void kernel(float *a, float *b) {
          for (int tile = 0; tile < 8; ++tile) {
            for (int i = 0; i < 64; ++i) b[i] = a[i];
            for (int i = 0; i < 64; ++i) b[i] += 1.0f;
          }
        }
        """,
        synth_report={
            "feedback": {
                "bottlenecks": [{"kind": "load_compute_serialize"}],
            }
        },
        max_skills=5,
        min_score=4.0,
    )

    assert "doublebuffer-load-compute" in route.audit["selected_skill_ids"]


def test_specialized_skills_require_task_evidence(tmp_path: Path) -> None:
    library = load_frozen_library(REPO / "skill_v2" / "skills.json")
    route = route_smart_skills(
        library,
        scope="smart_exhaustive",
        step_name="flash",
        current_code="""
        void kernel(double a[40 + 0][50 + 0]) {
          for (int i = 0; i < 40; ++i)
            for (int j = 0; j < 50; ++j)
              a[i][j] += 1.0;
        }
        """,
        synth_report={
            "feedback": {
                "bottlenecks": [
                    {"kind": "ii_target_miss"},
                    {"kind": "pipeline_blocked"},
                ]
            }
        },
        max_skills=3,
        min_score=4.0,
    )

    selected = set(route.audit["selected_skill_ids"])
    assert "hls-inplace-stencil-true-dependence" not in selected
    assert "dependence-inter-false-on-accum" not in selected
    assert "axi-burst-widening-512" not in selected
    stencil = next(
        score
        for score in route.audit["scores"]
        if score["skill_id"] == "hls-inplace-stencil-true-dependence"
    )
    assert stencil["specialty_compatible"] is False


def test_skillless_prompt_is_literal_empty_source() -> None:
    block = render_empty_skill_source(
        "skillless",
        reason="explicit_empty_skill_list",
    )

    assert "SKILL SOURCE: skillless" in block
    assert "AVAILABLE SKILLS: []" in block
    assert "[skill " not in block


def test_skillless_run_loads_no_library(monkeypatch) -> None:
    orchestrator = object.__new__(c2hls.C2HLSOrchestrator)
    orchestrator.skill_library = object()
    orchestrator.skill_library_provenance = {}
    monkeypatch.setenv("C2HLS_SKILL_MODE", "skillless")
    monkeypatch.setenv("C2HLS_SKILL_PROMPT_SCOPE", "skillless")

    orchestrator._prepare_skill_library_for_run()

    assert orchestrator.skill_library is None
    assert orchestrator.skill_library_provenance["source_mode"] == "skillless"
    assert (
        orchestrator.skill_library_provenance["store"]["skill_count"] == 0
    )
    assert orchestrator.skill_library_provenance["online_updates_enabled"] is False


def test_sweep_all_expands_to_five_stable_skill_conditions(
    monkeypatch,
) -> None:
    monkeypatch.setenv("C2HLS_SWEEP_SKILL_MODES", "all")

    assert _selected_skill_modes() == [
        ("skillless", "skillless"),
        ("matched", "matched"),
        ("smart_best_fit", "smart_best_fit"),
        ("smart_exhaustive", "smart_exhaustive"),
        ("all_positive", "all_positive"),
    ]


def test_post_profile_no_cosim_policy_disables_every_entry_point(
    monkeypatch,
) -> None:
    monkeypatch.setenv("C2HLS_SWEEP_DISABLE_ALL_COSIM", "1")
    monkeypatch.setenv("C2HLS_SWEEP_REFERENCE_AUDIT_ADVISORY", "1")
    for name in (
        "C2HLS_COSIM_REQUIRED",
        "C2HLS_COSIM_SELECTED_ONLY",
        "C2HLS_FORCE_SELECTED_COSIM",
        "C2HLS_REFERENCE_COSIM",
        "C2HLS_REFERENCE_COSIM_SELECTED_ONLY",
        "C2HLS_REFERENCE_COSIM_BASELINE",
        "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM",
    ):
        monkeypatch.setenv(name, "1")
    profile = {"name": "test"}

    _apply_post_profile_cosim_policy(profile)

    assert all(
        os.environ[name] == "0"
        for name in (
            "C2HLS_COSIM_REQUIRED",
            "C2HLS_COSIM_SELECTED_ONLY",
            "C2HLS_FORCE_SELECTED_COSIM",
            "C2HLS_REFERENCE_COSIM",
            "C2HLS_REFERENCE_COSIM_SELECTED_ONLY",
            "C2HLS_REFERENCE_COSIM_BASELINE",
            "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM",
        )
    )
    assert "csim_csynth_only" in profile["post_profile_overrides"]
    assert os.environ["C2HLS_REFERENCE_BLIND_FAIL_ON_LEAK"] == "0"
    assert (
        profile["post_profile_overrides"]["reference_isolation_audit"]
        == "advisory"
    )


def test_skill_v2_is_exact_frozen_superset() -> None:
    snapshot = REPO / "skill_v2" / "skills.json"
    manifest = json.loads(
        (REPO / "skill_v2" / "manifest.json").read_text(encoding="utf-8")
    )
    library = load_frozen_library(snapshot)

    assert len(library.all()) == 56
    assert manifest["primary"]["skill_count"] == 55
    assert manifest["supplement_added_ids"] == [
        "hls-inplace-stencil-true-dependence"
    ]
    assert library.get("hls-inplace-stencil-true-dependence") is not None
    assert all(skill.occurrences == 0 for skill in library.all())


def test_all_positive_v2_render_has_no_guard_or_avoid_sections() -> None:
    library = load_frozen_library(REPO / "skill_v2" / "skills.json")
    positive = [
        skill for skill in library.all()
        if skill.confidence != TIER_AVOID
    ]
    rendered = render_skill_set_for_prompt(
        positive,
        max_skills=len(positive),
        prompt_mode="action_only",
    )

    assert len(positive) == 42
    assert rendered.count("[skill ") == 42
    assert "  guards:" not in rendered
    assert not re.search(
        r"^\s+-\s+(?:do not|don't|avoid|never|must not)\b",
        rendered,
        re.IGNORECASE | re.MULTILINE,
    )


def test_positive_precondition_render_is_neutral_and_exact() -> None:
    library = load_frozen_library(REPO / "skill_v2" / "skills.json")
    positive = [
        skill for skill in library.all()
        if skill.confidence != TIER_AVOID
    ][:8]
    rendered = render_skill_set_for_prompt(
        positive,
        max_skills=5,
        prompt_mode="positive_with_preconditions",
    )

    assert rendered.count("[skill ") == 5
    assert rendered.count("positive applicability preconditions:") == 5
    assert "  guards:" not in rendered
    assert not re.search(
        r"^\s+-\s+(?:do not|don't|avoid|never|must not)\b",
        rendered,
        re.IGNORECASE | re.MULTILINE,
    )
