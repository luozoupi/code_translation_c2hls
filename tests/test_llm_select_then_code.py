"""Offline tests for llm_select_then_code skill helpers (no LLM/Vitis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from skill_library import (  # noqa: E402
    Skill,
    SkillLibrary,
    TIER_AVOID,
    TIER_HIGH,
    build_select_then_code_prompt_block,
    parse_skill_selection_response,
    render_skill_for_prompt,
    render_skill_for_prompt_full,
    salvage_skill_selection_from_truncated_reply,
)


def test_full_render_keeps_all_steps_guards_template_lines():
    steps = [f"step-{i}" for i in range(15)]
    guards = [f"guard-{i}" for i in range(12)]
    template = "\n".join(f"line{i}" for i in range(25))
    sk = Skill(
        id="s1",
        pattern="p",
        strategy="s",
        confidence=TIER_HIGH,
        required_steps=steps,
        guards=guards,
        template=template,
    )
    text = render_skill_for_prompt_full(sk)
    assert "step-14" in text
    assert "guard-11" in text
    assert "line24" in text
    trunc = render_skill_for_prompt(sk)
    assert "step-14" not in trunc
    assert "..." in trunc


def test_parse_selection_own_knowledge_optional():
    raw = (
        '{"selected_skill_ids":["s1"],"avoid_skill_ids":[],'
        '"own_knowledge":[]}'
    )
    parsed = parse_skill_selection_response(raw)
    assert parsed["selected_skill_ids"] == ["s1"]
    assert parsed["own_knowledge"] == []


def test_parse_selection_maps_solution_to_recommendation():
    raw = (
        '{"selected_skill_ids":["s1"],"avoid_skill_ids":["a1"],'
        '"own_knowledge":[{"title":"t","problem":"p","solution":"do x"}]}'
    )
    parsed = parse_skill_selection_response(raw)
    assert parsed["avoid_skill_ids"] == ["a1"]
    assert parsed["own_knowledge"][0]["recommendation"] == "do x"


def test_coder_block_label_and_uncapped_skills():
    lib = SkillLibrary(store_path=Path("/tmp/nonexistent_skill_store_sel_test.json"))
    skills = [
        Skill(id=f"s{i}", pattern="p", strategy="s", confidence=TIER_HIGH)
        for i in range(12)
    ]
    for sk in skills:
        lib.add(sk, overwrite=True)
    avoid = Skill(
        id="avoid-x",
        pattern="bad",
        strategy="dont",
        confidence=TIER_AVOID,
    )
    lib.add(avoid, overwrite=True)
    block = build_select_then_code_prompt_block(
        selected_skills=skills,
        avoid_skills=[avoid],
        own_knowledge=[{"title": "t", "problem": "p", "recommendation": "r"}],
        step_name="flash",
    )
    assert "further notes: model HLS knowledge here, not library skills" in block
    assert all(f"[skill s{i}]" in block for i in range(12))
    assert "[skill avoid-x]" in block
    assert "Prefer implementing the selected library skills first." in block
    fb = build_select_then_code_prompt_block(
        selected_skills=skills[:2],
        avoid_skills=[avoid],
        step_name="flash",
        used_fallback=True,
    )
    assert "fallback: aav_n full library" in fb
    assert "bottleneck-matched" not in fb


def test_salvage_truncated_selection_json_recovers_complete_ids():
    # Mirrors smoke8 failure: cut mid-string inside avoid list.
    raw = (
        '{\n  "selected_skill_ids": [\n'
        '    "hls-baseline-load-compute-store-gate",\n'
        '    "hls-multi-phase-local-pipeline",\n'
        '    "hls-chained-gemm-local-temporary"\n'
        "  ],\n"
        '  "avoid_skill_ids": [\n'
        '    "hls-avoid-zero-pipeline-submit",\n'
        '    "ii-avoid-false'
    )
    assert parse_skill_selection_response(raw)["selected_skill_ids"] == []
    salvaged = salvage_skill_selection_from_truncated_reply(raw)
    assert salvaged["salvaged_from_truncation"] is True
    assert salvaged["selected_skill_ids"] == [
        "hls-baseline-load-compute-store-gate",
        "hls-multi-phase-local-pipeline",
        "hls-chained-gemm-local-temporary",
    ]
    assert salvaged["avoid_skill_ids"] == ["hls-avoid-zero-pipeline-submit"]


def test_salvage_empty_reply_returns_empty():
    out = salvage_skill_selection_from_truncated_reply("")
    assert out["selected_skill_ids"] == []
    assert out["salvaged_from_truncation"] is False


def test_select_then_code_wired_in_c2hls():
    src = (REPO_ROOT / "c2hls.py").read_text(encoding="utf-8")
    assert 'skill_mode == "llm_select_then_code"' in src
    assert "select_then_code_for_flash" in src
    assert '"llm_select_then_code"' in src
    assert "_skill_selection_turns" in src
    assert "SkillSelectThenCodeReply" in src
    assert "_persist_skill_selection_json" in src
    assert "salvage_skill_selection_from_truncated_reply" in src
    assert "aav_n full-library fallback" in src
    assert "fallback_bottleneck_skills" not in src.split("def select_then_code_for_flash")[1].split("def ")[0]


def test_pipelined_state_roundtrips_skill_selection_raw_reply(tmp_path):
    """Codegen→finalize handoff must keep selection turns (incl. raw_reply)."""
    from c2hls import C2HLSOrchestrator

    orch = C2HLSOrchestrator(gpt_model="test-model", turns_limitation=1)
    orch._artifact_output_dir = str(tmp_path)
    record = {
        "enabled": True,
        "mode": "llm_select_then_code",
        "step_name": "flash",
        "used_fallback": True,
        "parse_error": "no valid skill ids after validation",
        "raw_reply": '{"selected_skill_ids":["not-a-real-id"],"avoid_skill_ids":[]}',
        "parsed": {"selected_skill_ids": ["not-a-real-id"], "avoid_skill_ids": []},
        "selected_skill_ids": ["hls-fallback-a"],
        "avoid_skill_ids": ["avoid-fallback-b"],
        "unknown_skill_ids": ["not-a-real-id"],
        "own_knowledge": [],
        "library_skill_count": 99,
        "injected_block_chars": 100,
    }
    orch._skill_selection_turns = [record]
    orch._skill_selection_record = record
    orch._append_history("assistant", "[SkillSelectThenCodeReply]\n" + record["raw_reply"])

    state = orch.pipelined_export_state()
    assert state["_skill_selection_turns"][0]["raw_reply"] == record["raw_reply"]

    orch2 = C2HLSOrchestrator(gpt_model="test-model", turns_limitation=1)
    orch2.pipelined_import_state(state)
    assert orch2._skill_selection_turns[0]["raw_reply"] == record["raw_reply"]
    assert orch2._skill_selection_record["parse_error"] == record["parse_error"]
    assert any(
        (m.get("content") or "").startswith("[SkillSelectThenCodeReply]")
        for m in orch2.history
    )

    orch2.save_multistep_results(str(tmp_path), "hlsfactory_2mm", {
        "benchmark": "hlsfactory_2mm",
        "success": True,
        "steps": [],
    })
    sel_path = tmp_path / "skill_selection.json"
    assert sel_path.is_file()
    saved = json.loads(sel_path.read_text(encoding="utf-8"))
    assert saved["latest"]["raw_reply"] == record["raw_reply"]
    assert saved["turns"][0]["used_fallback"] is True


def test_eager_persist_skill_selection_json(tmp_path):
    from c2hls import C2HLSOrchestrator, _persist_skill_selection_json

    orch = C2HLSOrchestrator(gpt_model="test-model", turns_limitation=1)
    orch._artifact_output_dir = str(tmp_path)
    orch._skill_selection_turns = [{
        "raw_reply": "hello-selector",
        "selected_skill_ids": ["s1"],
        "avoid_skill_ids": [],
        "used_fallback": False,
    }]
    orch._skill_selection_record = orch._skill_selection_turns[0]
    _persist_skill_selection_json(orch)
    saved = json.loads((tmp_path / "skill_selection.json").read_text(encoding="utf-8"))
    assert saved["latest"]["raw_reply"] == "hello-selector"


def test_selection_prompt_builder_includes_full_library_and_schema():
    from prompt_c2hls import build_skill_selection_user_prompt

    prompt = build_skill_selection_user_prompt(
        benchmark_name="hlsfactory_2mm",
        step_name="flash",
        synth_summary="lat=1",
        feedback_text="bn",
        diagnostic_text="warn",
        full_library_text="[skill demo] pattern: x",
        code_excerpt="void kernel() {}",
    )
    assert "FULL SKILL LIBRARY" in prompt
    assert "selected_skill_ids" in prompt
    assert "own_knowledge" in prompt
    assert "[skill demo]" in prompt
    assert "NO upper limit" in prompt
    assert "Prefer MORE skills" in prompt
    assert "When unsure, INCLUDE" in prompt
    assert "ALWAYS reply with non-empty content" in prompt
    assert "Do NOT invent skill ids" in prompt
    assert "COMPACT JSON" in prompt
    assert "Finish the JSON object completely" in prompt
