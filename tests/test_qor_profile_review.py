from types import SimpleNamespace
from unittest.mock import Mock
import json
from pathlib import Path
import sys

import pytest

import c2hls
import qor_profile_review as review
from evaluation_repro import build_run_fingerprint


def report(cycles=100, **extra):
    return {"latency_cycles_worst": cycles, "latency_cycles": cycles,
            "dsp": 10, "bram": 4, "lut": 100, "ff": 100, "uram": 0,
            "estimated_clock_period_ns": 3.0, **extra}


def inspect(**overrides):
    skill = SimpleNamespace(id="pipeline", kind="optimization", tags=["pipeline"],
                            bottleneck_kinds=["pipeline_ii"])
    args = dict(parent_code="void f() {}", candidate_code="#pragma HLS PIPELINE II=1\nvoid f() {}",
                parent_report=report(), candidate_report=report(80, dsp=20),
                skill_meta={"rendered_skill_ids": ["pipeline"], "declared_applied_skill_ids": ["pipeline"]},
                library={"pipeline": skill}, capacity={k: 1000 for k in review.RESOURCES},
                feasibility={"feasible": True, "correctness_ok": True}, synth_success=True)
    args.update(overrides)
    return review.review_candidate(**args)


def test_faster_candidate_can_spend_more_resources():
    result = inspect()
    assert result["latency_effect"] == "improved"
    assert result["metric_deltas"]["dsp"] == 10
    assert result["pareto_relation"] == "tradeoff_or_equal"
    assert not result["retry_recommended"]
    assert result["skill_reviews"][0]["causal_attribution"] == "not_isolated_per_skill"
    assert result["skill_reviews"][0]["recipe_correctness"] == "not_proven_by_static_markers"


@pytest.mark.parametrize("code", ["// #pragma HLS PIPELINE II=1\nvoid f() {}",
                                  "/* #pragma HLS PIPELINE II=1 */ void f() {}",
                                  'const char *s = "#pragma HLS PIPELINE II=1";'])
def test_comments_and_strings_are_not_application_evidence(code):
    result = inspect(candidate_code=code, candidate_report=report())
    assert result["unsupported_declarations"] == ["pipeline"]
    assert result["retry_recommended"]


def test_existing_pragma_parameter_changes_are_visible():
    result = inspect(parent_code="#pragma HLS PIPELINE II=4\nvoid f() {}")
    assert result["skill_reviews"][0]["source_status"] == "new_or_changed_family_evidence"
    assert result["unsupported_declarations"] == []


def test_skillless_never_queries_library():
    library = Mock()
    library.get.side_effect = AssertionError("skillless must not read the library")
    result = inspect(skill_meta={"rendered_skill_ids": []}, library=library)
    assert result["skill_reviews"] == []
    library.get.assert_not_called()


@pytest.mark.parametrize("feasibility,synth_success", [
    ({"feasible": False, "correctness_ok": False}, True),
    ({"feasible": False, "correctness_ok": True}, True),
    ({"feasible": True, "correctness_ok": True}, False),
])
def test_invalid_candidate_is_not_measured_improvement(feasibility, synth_success):
    result = inspect(feasibility=feasibility, synth_success=synth_success)
    assert result["latency_effect"] == "unmeasured_or_invalid"
    assert not result["feasible"]
    assert not result["retry_recommended"]


def test_resource_saving_at_equal_latency_is_not_needlessly_retried():
    result = inspect(candidate_report=report(100, dsp=5), candidate_code="void f() {}")
    assert result["pareto_relation"] == "candidate_dominates_parent"
    assert not result["retry_recommended"]


def test_missing_resources_are_unknown_not_zero():
    result = inspect(candidate_report={"latency_cycles_worst": 80})
    assert result["pareto_relation"] == "insufficient_evidence"
    assert result["resource_utilization"] == {}


def test_profile_reviewer_reports_persistent_bottleneck_without_causal_claim():
    feedback = {"bottlenecks": [{"kind": "pipeline_ii", "scope_id": "loop_A", "evidence": "II=4 due to recurrence"}]}
    result = inspect(parent_report=report(feedback=feedback), candidate_report=report(feedback=feedback))
    skill = result["skill_reviews"][0]
    assert skill["profile_effect"] == "target_bottleneck_still_reported"
    assert result["retry_recommended"]
    assert "II=4 due to recurrence" in result["guidance"]
    missing = inspect(parent_report=report(feedback=feedback), candidate_report=report())
    assert missing["skill_reviews"][0]["profile_effect"] == "insufficient_profile_evidence"


def test_quality_agent_records_review_without_model_calls():
    orch = SimpleNamespace(gpt_model="deepseek-flash", hls_code="void f() {}", synth_report=report(), skill_library={},
                           testbench_code="test", part="xcu280-fsvh2892-2L-e", clock_ns=3.33)
    agent = c2hls.QualityRepairAgent(orch)
    result = agent.review_optimization_candidate(
        code="void f() { int x=0; }", report=report(80, dsp=20),
        csim={"ran": True, "passed": True}, skill_meta={"rendered_skill_ids": []}, synth_success=True)
    assert result["feasible"]
    assert orch.qor_profile_reviews == [result]
    assert result["parent_code_sha256"] != result["candidate_code_sha256"]


def test_review_retry_reaches_coder_within_existing_two_attempts(monkeypatch):
    monkeypatch.setenv(review.REVIEW_ENV, "1")
    first = {"success": True, "report": report(), "code": "ineffective",
             "skill_prompt": {"qor_review": {"retry_recommended": True, "guidance": "fix measured loop_A recurrence"}}}
    second = {"success": True, "report": report(80, dsp=20), "code": "improved", "csim": {"passed": True}}
    orch = SimpleNamespace(hls_code="parent", synth_report=report(), generated_csim={"passed": True},
                           generated_cosim=None, part="xcu280-fsvh2892-2L-e", _append_history=Mock(),
                           _optimization_step_attempt=Mock(side_effect=[first, second]))
    result = c2hls.C2HLSOrchestrator.run_optimization_step(orch, "pipeline")
    assert result["code"] == "improved"
    assert orch._optimization_step_attempt.call_count == 2
    assert orch._optimization_step_attempt.call_args.kwargs["additional_guidance"] == "fix measured loop_A recurrence"


def test_prompt_and_fingerprint_distinguish_opt_in_review(monkeypatch, tmp_path):
    monkeypatch.setenv(review.REVIEW_ENV, "1")
    prompt = c2hls._render_candidate_improvement_prompt("pipeline", 0, 1, 0, 1, report(), "code")
    assert "resource growth is allowed" in prompt
    assert "more conservative" not in c2hls._render_regression_guidance("pipeline", ["latency increased"])
    assert "not minimum hardware usage" in review.resource_guidance(report(), {})
    args = dict(repo=tmp_path, benchmark_dir=tmp_path, benchmark="tiny", model_id="deepseek-flash",
                model_label="deepseek_flash", skill_mode="matched", steps=[], profile="test")
    old = build_run_fingerprint(**args, environ={})
    new = build_run_fingerprint(**args, environ={review.REVIEW_ENV: "1"})
    assert old != new


def test_flash_campaign_labels_are_not_pro():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_qor_paired_model_campaign as runner
    config_path = Path(__file__).resolve().parents[1] / "tests/fixtures/protocols/deepseek_v4_pro_qor_paired27_axi_v2_20260908.json"
    config = json.loads(config_path.read_text())
    config["model"] = "deepseek-flash"
    cells = runner.cells_for(config)
    assert len(cells) == 54
    assert all("_deepseek_flash_matched/" in c["result_path"] for c in cells)
    assert runner.model_label({"model": "deepseek-v4-pro"}) == "deepseek_v4_pro"
