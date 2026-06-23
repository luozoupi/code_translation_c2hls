"""Smoke tests for LLM skill curation helpers (offline, no LLM/Vitis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hls_feedback import render_diagnostic_for_prompt  # noqa: E402
from skill_library import (  # noqa: E402
    Skill,
    SkillLibrary,
    TIER_AVOID,
    TIER_HIGH,
    build_curated_skill_prompt_block,
    parse_skill_curation_response,
    validate_and_resolve_curation,
)


def _tiny_library() -> SkillLibrary:
    lib = SkillLibrary(store_path=Path("/tmp/nonexistent_skill_store_curation_test.json"))
    lib.add(
        Skill(
            id="pipeline-inner-ii1",
            pattern="hot loop not pipelined",
            strategy="add PIPELINE II=1",
            confidence=TIER_HIGH,
            bottleneck_kinds=["non_pipelined_hot_loop"],
        ),
        overwrite=True,
    )
    lib.add(
        Skill(
            id="avoid-full-unroll",
            pattern="full unroll on large loop",
            strategy="do not fully unroll",
            confidence=TIER_AVOID,
            bottleneck_kinds=["ii_target_miss"],
        ),
        overwrite=True,
    )
    return lib


def test_render_diagnostic_for_prompt():
    feedback = {
        "static_extras": {
            "diagnostic": {
                "warnings": 2,
                "errors": 1,
                "rejected_pragmas": [
                    {
                        "id": "PRAGMA_INVALID",
                        "loc": "kernel.cpp:10",
                        "body": "unsupported PIPELINE on this loop",
                    }
                ],
                "examples": [
                    {
                        "severity": "WARNING",
                        "id": "W123",
                        "loc": "kernel.cpp:20",
                        "body": "loop cannot be pipelined",
                    }
                ],
            }
        }
    }
    text = render_diagnostic_for_prompt(feedback)
    assert "PRAGMA_INVALID" in text
    assert "W123" in text
    assert "2 warnings" in text


def test_parse_skill_curation_response_json():
    raw = json.dumps({
        "analysis": {"primary_bottlenecks": ["ii_target_miss"], "key_warnings": [], "lcs_notes": ""},
        "selected_skill_ids": ["pipeline-inner-ii1"],
        "avoid_skill_ids": [],
        "curated_guidance": [],
    })
    parsed = parse_skill_curation_response(raw)
    assert parsed["selected_skill_ids"] == ["pipeline-inner-ii1"]
    assert parsed["analysis"]["primary_bottlenecks"] == ["ii_target_miss"]


def test_validate_json_only_rejects_unknown_ids():
    lib = _tiny_library()
    parsed = parse_skill_curation_response(json.dumps({
        "selected_skill_ids": ["pipeline-inner-ii1", "not-a-real-skill"],
        "avoid_skill_ids": [],
        "curated_guidance": [{"title": "x", "problem": "p", "solution": "s"}],
    }))
    resolved = validate_and_resolve_curation(
        parsed,
        lib,
        sector="json_only",
        include_avoids=True,
    )
    assert len(resolved["selected_skills"]) == 1
    assert resolved["selected_skills"][0].id == "pipeline-inner-ii1"
    assert "not-a-real-skill" in resolved["unknown_skill_ids"]
    assert resolved["curated_guidance"] == []


def test_build_curated_skill_prompt_block():
    lib = _tiny_library()
    parsed = parse_skill_curation_response(json.dumps({
        "selected_skill_ids": ["pipeline-inner-ii1"],
        "avoid_skill_ids": ["avoid-full-unroll"],
        "curated_guidance": [],
    }))
    resolved = validate_and_resolve_curation(
        parsed, lib, sector="json_only", include_avoids=True,
    )
    block = build_curated_skill_prompt_block(resolved, step_name="flash")
    assert "LLM-CURATED SKILL GUIDANCE" in block
    assert "pipeline-inner-ii1" in block
    assert "avoid-full-unroll" in block


def test_llm_curated_branch_wired_in_c2hls():
    src = (REPO_ROOT / "c2hls.py").read_text(encoding="utf-8")
    assert 'skill_mode == "llm_curated"' in src
    assert "skill_curation.curate_for_flash" in src
    assert '"llm_curated"' in src and "GLOBAL_SKILL_PROMPT_MODES" in src


def test_curation_prompt_builder_exists():
    from prompt_c2hls import build_skill_curation_user_prompt

    prompt = build_skill_curation_user_prompt(
        focus="combined",
        sector="json_plus_llm",
        include_avoids=True,
        benchmark_name="test_bench",
        step_name="flash",
        synth_summary="lat=1000",
        feedback_text="Top bottlenecks: ii_target_miss",
        diagnostic_text="1 warnings",
        catalog_text="- skill-a | high | ii_target_miss | pattern",
        code_excerpt="void workload() {}",
    )
    assert "load-compute-store" in prompt.lower() or "LCST" in prompt
    assert "json_plus_llm" in prompt
    assert "skill-a" in prompt


def main() -> int:
    test_render_diagnostic_for_prompt()
    test_parse_skill_curation_response_json()
    test_validate_json_only_rejects_unknown_ids()
    test_build_curated_skill_prompt_block()
    test_llm_curated_branch_wired_in_c2hls()
    test_curation_prompt_builder_exists()
    print("test_skill_curation_smoke: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
