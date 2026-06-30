#!/usr/bin/env python3
"""Tests for C2HLS_RECORD_FLOW flash artifact naming and selection."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flash_flow_artifacts import (  # noqa: E402
    capture_flash_step_skills,
    capture_step_skills,
    infer_flow_selected_from,
    infer_multistep_selected_from,
    record_flow_enabled,
    write_flow_artifacts,
    write_flash_skills_record,
    write_multistep_flow_artifacts,
)


def test_record_flow_disabled_by_default():
    old = os.environ.pop("C2HLS_RECORD_FLOW", None)
    try:
        assert record_flow_enabled() is False
    finally:
        if old is not None:
            os.environ["C2HLS_RECORD_FLOW"] = old


def test_infer_selected_from_phase_b_win():
    results = {
        "baseline_report": {"latency_cycles": 100},
        "final_report": {"latency_cycles": 100},
        "steps": [{"step_name": "flash", "success": True, "report": {"latency_cycles": 200}}],
        "best_so_far_promotion": {
            "promoted": True,
            "from_step_name": "baseline",
            "from_step_index": -1,
        },
    }
    assert infer_flow_selected_from(results) == "phase_b"


def test_infer_selected_from_flash_win():
    results = {
        "baseline_report": {"latency_cycles": 200},
        "final_report": {"latency_cycles": 100},
        "steps": [{"step_name": "flash", "success": True, "report": {"latency_cycles": 100}}],
        "best_so_far_promotion": {"promoted": False, "reason": "final state was already the best"},
    }
    assert infer_flow_selected_from(results) == "flash_opt"


def test_write_flow_artifacts_writes_flat_files():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        bench = "hlsfactory_trisolv"
        results = {
            "baseline_report": {"latency_cycles": 100},
            "final_report": {"latency_cycles": 100},
            "steps": [
                {
                    "step_name": "flash",
                    "success": True,
                    "code": "flash opt code\n",
                    "report": {"latency_cycles": 200},
                }
            ],
            "best_so_far_promotion": {
                "promoted": True,
                "from_step_name": "baseline",
                "from_step_index": -1,
            },
        }
        write_flow_artifacts(
            out,
            bench,
            plain_code="plain input\n",
            phase_b_code="phase b code\n",
            phase_b_report={"latency_cycles": 100},
            flash_opt_code="flash opt code\n",
            flash_opt_report={"latency_cycles": 200},
            selected_code="phase b code\n",
            selected_report={"latency_cycles": 100},
            results=results,
        )

        assert (out / "plain.cpp").read_text() == "plain input\n"
        assert (out / f"{bench}_phase_b.cpp").read_text() == "phase b code\n"
        assert (out / f"{bench}_flash_opt.cpp").read_text() == "flash opt code\n"
        assert (out / f"{bench}_selected.cpp").read_text() == "phase b code\n"
        assert (out / f"{bench}_final.cpp").read_text() == "phase b code\n"
        manifest = json.loads((out / f"{bench}_flow_manifest.json").read_text())
        assert manifest["selected_from"] == "phase_b"
        assert manifest["latency_cycles"]["phase_b"] == 100
        assert manifest["latency_cycles"]["flash_opt"] == 200


def test_write_flash_skills_record_copies_source_and_details():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        skills_src = out / "packaged_skills.json"
        skills_src.write_text(
            json.dumps(
                {
                    "schema": "1.1",
                    "skills": [
                        {
                            "id": "demo-skill",
                            "pattern": "p",
                            "strategy": "s",
                            "confidence": "high",
                            "guards": ["g1"],
                            "required_steps": ["r1"],
                            "template": "t",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        old = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON")
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(skills_src)
        try:
            context = capture_flash_step_skills(
                step_name="flash",
                skill_prompt_mode="all_skills_no_avoids_global",
                skill_header="HEADER\n",
                prompt_skills=[
                    {
                        "id": "demo-skill",
                        "pattern": "p",
                        "strategy": "s",
                        "confidence": "high",
                        "guards": ["g1"],
                        "required_steps": ["r1"],
                        "template": "t",
                        "bottleneck_kinds": [],
                        "applicable_versions": [],
                        "applicable_fpgas": [],
                        "tags": [],
                        "kind": "",
                        "occurrences": 0,
                        "sec_pass": 0,
                        "mean_advantage": 0.0,
                        "last_used_at": None,
                        "origin": "manual",
                    }
                ],
                injected_prompt_text="HEADER\n[skill demo-skill]",
                top_bottleneck_kind=None,
                skill_id=None,
                skill_curation_record=None,
                skill_library=None,
            )
            rel = write_flash_skills_record(out, "hlsfactory_demo", context)
            assert rel == "hlsfactory_demo_flash_skills.json"
            saved = json.loads((out / rel).read_text())
            assert saved["flash_opt"]["injected_skill_count"] == 1
            assert saved["flash_opt"]["injected_skills"][0]["id"] == "demo-skill"
            assert saved["flash_opt"]["injected_prompt_text"].startswith("HEADER")
            assert (out / "skills_source.json").exists()
        finally:
            if old is None:
                os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
            else:
                os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = old


if __name__ == "__main__":
    test_record_flow_disabled_by_default()
    test_infer_selected_from_phase_b_win()
    test_infer_selected_from_flash_win()
    test_write_flow_artifacts_writes_flat_files()
    test_write_flash_skills_record_copies_source_and_details()
    print("test_record_flow_artifacts: all ok")
