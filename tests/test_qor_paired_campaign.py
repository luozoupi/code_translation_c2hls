from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import analyze_qor_paired_model_campaign as analysis
import run_qor_paired_model_campaign as campaign
import c2hls
from qor_design_space import extract_qor_metrics


def fixture_pair(tmp_path):
    parent_code, final_code = "void f() {}", "void f() { int i = 1; }"
    parent_report = {"latency_cycles_worst": 100, "dsp": 10}
    final_report = {"latency_cycles_worst": 80, "dsp": 12}
    usage = {"calls": 2, "events": [{"model": "deepseek-v4-pro"}]}
    parent = {"hls_code": parent_code, "code_sha256": analysis.code_hash(parent_code),
              "final_report": parent_report, "llm_usage": usage, "csim": {"passed": True}}
    design = {
        "winner_candidate_id": "valid_winner",
        "parent_code_sha256": analysis.code_hash(parent_code),
        "winner_code_sha256": analysis.code_hash(final_code),
        "parent_integrity_passed": True,
        "winner_metrics": extract_qor_metrics(final_report),
        "parent": {"candidate_id": "frozen_parent", "feasible": True,
                   "metrics": extract_qor_metrics(parent_report)},
        "candidates": [
            {"candidate_id": "fast_but_invalid", "feasible": False,
             "metrics": {"latency_cycles_worst": 1}},
            {"candidate_id": "valid_winner", "feasible": True,
             "metrics": extract_qor_metrics(final_report)},
        ],
    }
    result = {"success": True, "hls_code": final_code, "final_report": final_report,
              "csim": {"passed": True}, "llm_usage": usage,
              "candidate_feasibility": {"feasible": True},
              "reference_isolation_audit": {"passed": True},
              "skill_snapshot_integrity": {"unchanged": True}, "qor_design_sweep": design}
    cell = {"cell_id": "tiny_flash_matched", "benchmark": "tiny", "family": "test",
            "strategy": "flash", "skill_mode": "matched", "status": "running",
            "result_path": str(tmp_path / "result.json"), "parent_path": str(tmp_path / "parent.json")}
    Path(cell["result_path"]).write_text(json.dumps(result))
    Path(cell["parent_path"]).write_text(json.dumps(parent))
    return cell, result, parent


def test_exact_pair_and_invalid_candidate_budget_fallback(tmp_path):
    cell, _, _ = fixture_pair(tmp_path)
    row = analysis.analyze_cell(cell, {"model": "deepseek-v4-pro"})
    assert row["pair_valid"] and row["speedup"] == 1.25
    budgets = analysis.budget_rows(cell, row)
    assert budgets[0]["best_cycles"] == 100
    assert all(row["best_cycles"] == 80 for row in budgets[1:])


@pytest.mark.parametrize("change,expected", [
    ("hash", "paired parent code hash mismatch"),
    ("model", "LLM events do not exclusively identify the registered model"),
    ("extra_call", "extra model calls occurred after the paired control"),
    ("regression", "QoR selection regressed cycles"),
    ("parent_metrics", "paired parent report differs from frozen QoR metrics"),
])
def test_pair_rejects_confounded_or_regressing_outcomes(tmp_path, change, expected):
    cell, result, parent = fixture_pair(tmp_path)
    if change == "hash":
        parent["hls_code"] = "changed"
    elif change == "model":
        result["llm_usage"] = {"calls": 2, "events": [{"model": "another-model"}]}
    elif change == "extra_call":
        result["llm_usage"] = {"calls": 3, "events": [{"model": "deepseek-v4-pro"}]}
    elif change == "parent_metrics":
        parent["final_report"]["latency_cycles_worst"] = 200
    else:
        result["final_report"]["latency_cycles_worst"] = 120
    Path(cell["result_path"]).write_text(json.dumps(result))
    Path(cell["parent_path"]).write_text(json.dumps(parent))
    row = analysis.analyze_cell(cell, {"model": "deepseek-v4-pro"})
    assert not row["pair_valid"]
    assert row["speedup"] is None
    assert expected in row["issues"]


def test_paired_parent_export_keeps_original_source_and_checks_hash(tmp_path):
    _, _, parent = fixture_pair(tmp_path)
    orch = SimpleNamespace(
        hls_code="final code", qor_frozen_parent_snapshot=parent,
        gpt_model="deepseek-v4-pro", _llm_usage_summary=lambda: {},
        llm_controller_transcript=[], history=[],
    )
    c2hls.C2HLSOrchestrator.save_multistep_results(orch, str(tmp_path), "tiny", {"steps": []})
    assert (tmp_path / "tiny_pre_qor.cpp").read_text() == parent["hls_code"]
    assert (tmp_path / "tiny_final.cpp").read_text() == "final code"
    parent["hls_code"] = "corrupted"
    with pytest.raises(RuntimeError, match="hash changed"):
        c2hls.C2HLSOrchestrator.save_multistep_results(orch, str(tmp_path), "tiny", {"steps": []})


def test_registered_matrix_has_two_structurally_distinct_smoke_cells():
    config = json.loads((REPO / "artifacts/experiment_matrix/deepseek_v4_pro_qor_paired12_20260906.json").read_text())
    cells = campaign.cells_for(config)
    assert len(cells) == 24
    assert len({cell["cell_id"] for cell in cells}) == 24
    assert [cell["cell_id"] for cell in cells[:2]] == config["smoke_cell_ids"]
    assert cells[0]["benchmark"] != cells[1]["benchmark"]
    assert config["environment"]["C2HLS_LLM_SEED"] == "42"


def test_preflight_refresh_rejects_any_existing_model_artifact(tmp_path):
    cell, _, _ = fixture_pair(tmp_path)
    root = tmp_path / "reports"
    root.mkdir()
    (root / "campaign_state.json").write_text(json.dumps({"status": "smoke_failed", "cells": [cell]}))
    with pytest.raises(ValueError, match="model or evaluation artifacts"):
        campaign.prepare(tmp_path / "config.json", {"report_root": str(root)}, refresh_empty_failure=True)


def test_characterization_separates_invalid_from_slower_and_resource_tradeoffs(tmp_path):
    _, result, _ = fixture_pair(tmp_path)
    result["qor_design_sweep"]["candidates"].extend([
        {"candidate_id": "slow", "feasible": True,
         "metrics": {"latency_cycles_worst": 120, "dsp": 8}},
        {"candidate_id": "same", "feasible": True,
         "metrics": {"latency_cycles_worst": 100, "dsp": 10}},
    ])
    evidence = analysis.characterize_result(result)
    assert evidence["outcomes"] == {"faster": 1, "slower": 1, "unchanged": 1, "invalid": 1}
    assert evidence["selected_effect"] == "improved"
    assert evidence["selected_metric_deltas"]["dsp"] == 2
    assert evidence["candidate_comparisons"][0]["latency_change_percent"] is None
    assert evidence["candidate_comparisons"][0]["metric_deltas"] == {}
    assert evidence["candidate_comparisons"][2]["latency_change_percent"] == pytest.approx(20)


def test_method_version_is_not_conflated_with_legacy(tmp_path):
    cell, _, _ = fixture_pair(tmp_path)
    row = analysis.analyze_cell(cell, {"model": "deepseek-v4-pro", "method_version": "axi_bundle_linked_v2"})
    assert not row["pair_valid"]
    assert "QoR method version differs from registered campaign" in row["issues"]


def test_characterization_export_with_empty_and_valid_matrix(tmp_path):
    cell, _, _ = fixture_pair(tmp_path)
    analysis.write_characterization([], tmp_path)
    row = analysis.analyze_cell(cell, {"model": "deepseek-v4-pro"})
    analysis.write_characterization([row], tmp_path)
    assert (tmp_path / "candidate_outcomes.png").stat().st_size > 1000
    assert "measured associations" in (tmp_path / "parameter_effects.md").read_text()


def test_expanded_matrix_covers_twenty_seven_benches_and_versions_method():
    config = json.loads((REPO / "artifacts/experiment_matrix/deepseek_v4_pro_qor_paired27_axi_v2_20260908.json").read_text())
    cells = campaign.cells_for(config)
    assert len(cells) == 54
    assert len({cell["benchmark"] for cell in cells}) == 27
    assert len({cell["cell_id"] for cell in cells}) == 54
    assert "hlsfactory_doitgen" not in {cell["benchmark"] for cell in cells}
    assert config["method_version"] == "axi_bundle_linked_v2"
    assert len(config["required_regression_outputs"]) == 2
    assert [cell["cell_id"] for cell in cells[:2]] == config["smoke_cell_ids"]
