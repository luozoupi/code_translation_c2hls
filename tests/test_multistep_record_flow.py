#!/usr/bin/env python3
"""Tests for multistep C2HLS_RECORD_FLOW artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flash_flow_artifacts import (  # noqa: E402
    capture_step_skills,
    infer_multistep_selected_from,
    write_multistep_flow_artifacts,
)


def test_infer_multistep_selected_from_coalescing_win():
    results = {
        "baseline_report": {"latency_cycles": 500},
        "final_report": {"latency_cycles": 100},
        "steps": [
            {"step_name": "tiling", "success": True, "report": {"latency_cycles": 400}},
            {"step_name": "pipeline", "success": True, "report": {"latency_cycles": 200}},
            {"step_name": "coalescing", "success": True, "report": {"latency_cycles": 100}},
        ],
    }
    assert infer_multistep_selected_from(results) == "coalescing"


def test_write_multistep_flow_artifacts_two_steps():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        bench = "hlsfactory_2mm"
        results = {
            "baseline_report": {"latency_cycles": 200},
            "final_report": {"latency_cycles": 150},
            "steps": [
                {
                    "step_name": "tiling",
                    "success": True,
                    "code": "tiling code\n",
                    "report": {"latency_cycles": 180},
                },
                {
                    "step_name": "pipeline",
                    "success": True,
                    "code": "pipeline code\n",
                    "report": {"latency_cycles": 150},
                },
            ],
        }
        skills = [
            capture_step_skills(
                step_name="tiling",
                skill_prompt_mode="all_skills_avoids_global",
                skill_header="HDR\n",
                prompt_skills=[],
                injected_prompt_text="HDR\n",
                top_bottleneck_kind=None,
                skill_id=None,
                skill_curation_record=None,
                skill_library=None,
            )
        ]
        write_multistep_flow_artifacts(
            out,
            bench,
            plain_code="plain\n",
            phase_b_code="phase b\n",
            phase_b_report={"latency_cycles": 200},
            step_artifacts=[
                {"step_name": "tiling", "code": "tiling code\n", "report": {"latency_cycles": 180}, "success": True},
                {"step_name": "pipeline", "code": "pipeline code\n", "report": {"latency_cycles": 150}, "success": True},
            ],
            selected_code="pipeline code\n",
            selected_report={"latency_cycles": 150},
            results=results,
            skills_records=skills,
        )
        assert (out / f"{bench}_tiling.cpp").read_text() == "tiling code\n"
        assert (out / f"{bench}_pipeline.cpp").read_text() == "pipeline code\n"
        assert (out / f"{bench}_multistep_skills.json").is_file()
        manifest = json.loads((out / f"{bench}_flow_manifest.json").read_text())
        assert manifest["schema"] == "multistep_flow_manifest_v1"
        assert manifest["selected_from"] == "pipeline"
        assert manifest["origin_meta"]["note"]


def test_c2hls_save_multistep_results_record_flow():
    try:
        from c2hls import C2HLSOrchestrator
    except ImportError as exc:
        print(f"skip c2hls integration test: {exc}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        bench = "hlsfactory_demo"
        orch = C2HLSOrchestrator(gpt_model="test/model", turns_limitation=1)
        orch.strategy = "static"
        orch.c_code = "plain c\n"
        orch.hls_code = "selected\n"
        orch.synth_report = {"latency_cycles": 100}
        orch._flow_phase_b_code = "phase b\n"
        orch._flow_phase_b_report = {"latency_cycles": 200}
        results = {
            "baseline_report": {"latency_cycles": 200},
            "final_report": {"latency_cycles": 100},
            "steps": [
                {
                    "step_name": "tiling",
                    "success": True,
                    "code": "tiling\n",
                    "report": {"latency_cycles": 150},
                },
            ],
        }
        with mock.patch.dict("os.environ", {"C2HLS_RECORD_FLOW": "1"}, clear=False):
            orch.save_multistep_results(str(out), bench, results)
        assert (out / f"{bench}_flow_manifest.json").is_file()
        assert (out / f"{bench}_tiling.cpp").is_file()
        assert not (out / "steps").exists()


if __name__ == "__main__":
    test_infer_multistep_selected_from_coalescing_win()
    test_write_multistep_flow_artifacts_two_steps()
    test_c2hls_save_multistep_results_record_flow()
    print("test_multistep_record_flow: all ok")
