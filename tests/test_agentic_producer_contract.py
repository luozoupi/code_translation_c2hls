from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402
import run_agentic_sweep  # noqa: E402
from scripts.normalize_hpca_freeze_index import _agentic_candidate_events  # noqa: E402


PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33


def _report(cycles: int) -> dict:
    return {
        "latency_cycles": cycles,
        "latency_cycles_worst": cycles,
        "bram": 4,
        "dsp": 8,
        "ff": 1000,
        "lut": 2000,
        "uram": 0,
        "slack_ns": 0.05,
        "requested_clock_period_ns": CLOCK_NS,
    }


def _passing_csim() -> dict:
    return {
        "status": "passed",
        "supported": True,
        "ran": True,
        "success": True,
        "passed": True,
        "error": "",
    }


def _orchestrator() -> c2hls.C2HLSOrchestrator:
    orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
    orch.llm_usage_events = []
    orch.llm_candidate_request_count = 0
    orch.llm_candidate_budget = 5
    orch.synthesis_eval_events = []
    orch.synthesis_eval_count = 0
    orch.synthesis_eval_budget = 5
    orch._candidate_stream_started_monotonic = None
    orch.selected_winner_cosim_count = 0
    orch.selected_code_sha256 = None
    orch.cosim_target_code_sha256 = None
    orch.header_code = ""
    orch.header_name = "kernel.h"
    orch.translated_hls_top = "workload"
    orch.part = PART
    orch.clock_ns = CLOCK_NS
    orch.extra_files = []
    orch.testbench_code = "int main() { return 0; }"
    orch.supports_cosim = True
    orch.cosim_depths = {"a": 4}
    orch.reference_blind = True
    orch.cosim_reference_cycle_info = {}
    orch.independent_golden_output = ""
    orch.independent_golden_specs = {}
    orch.generated_cosim = None
    return orch


def _record_response(orch: c2hls.C2HLSOrchestrator, tokens: int) -> None:
    index = orch.llm_candidate_request_count
    orch.llm_candidate_request_count += 1
    orch._record_llm_usage(
        provider="openai",
        model="test-model",
        agent_name="synthesis",
        usage={
            "prompt_tokens": tokens - 10,
            "completion_tokens": 10,
            "total_tokens": tokens,
        },
        messages=[{"role": "user", "content": f"candidate {index}"}],
        max_tokens=128,
        decoding={"temperature": 0.0, "top_p": 1.0, "seed": 7},
        candidate_evaluation_index=index,
    )


class AgenticProducerContractTests(unittest.TestCase):
    def test_current_producer_stream_normalizes_without_counting_csim_only(self):
        orch = _orchestrator()
        bad_code = "void workload() { /* wrong */ }"
        selected_code = "void workload() { /* selected */ }"
        report = _report(600)
        csim_failure = {
            "status": "failed",
            "supported": True,
            "ran": True,
            "success": False,
            "passed": False,
            "error": "golden mismatch",
        }
        rejected = {
            "synth": {
                "success": False,
                "ran": False,
                "skipped": True,
                "skip_reason": "csim_correctness_gate_failed",
                "report": {},
                "error": "CSim gate failed",
            },
            "csim": csim_failure,
            "cosim": None,
        }
        accepted = {
            "synth": {
                "success": True,
                "ran": True,
                "report": report,
                "error": "",
            },
            "csim": _passing_csim(),
            "cosim": None,
        }

        _record_response(orch, 100)
        with patch.object(c2hls, "_run_synth_csim_cosim", return_value=rejected):
            orch._synth_and_test(bad_code, "candidate 0")
        _record_response(orch, 120)
        with patch.object(c2hls, "_run_synth_csim_cosim", return_value=accepted):
            orch._synth_and_test(selected_code, "candidate 1")

        orch.hls_code = selected_code
        raw_cosim = {
            "success": True,
            "passed": True,
            "error": "",
            "kernel_runtime_cycles": 650,
        }
        with (
            patch.dict(
                os.environ,
                {c2hls.COSIM_SELECTED_ONLY_ENV: "1"},
                clear=False,
            ),
            patch.object(c2hls, "run_cosim", return_value=raw_cosim),
        ):
            cosim = orch._run_selected_winner_cosim()

        summary = orch._synthesis_evaluation_summary()
        usage = orch._llm_usage_summary()
        root = {
            "llm_usage": usage,
            "synthesis_evaluations": summary,
            "selected_winner_cosim_count": orch.selected_winner_cosim_count,
            "total_synthesis_calls": orch._total_synthesis_calls(),
            "selected_code_sha256": orch.selected_code_sha256,
            "cosim_target_code_sha256": orch.cosim_target_code_sha256,
        }

        self.assertTrue(summary["complete_candidate_event_stream"])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(root["total_synthesis_calls"], 2)
        self.assertEqual(summary["events"][0]["correctness_status"], "failed")
        self.assertEqual(summary["events"][0]["synthesis_status"], "not_run")
        self.assertEqual(
            summary["events"][0]["cumulative_synthesis_evaluations"], 0
        )
        self.assertEqual(
            summary["events"][1]["cumulative_synthesis_evaluations"], 1
        )
        self.assertEqual(summary["events"][1]["report_sha256"],
                         orch._candidate_report_sha256(report))
        self.assertEqual(orch.selected_code_sha256, orch.cosim_target_code_sha256)
        self.assertTrue(cosim["passed"])

        normalized_events = _agentic_candidate_events(
            root, "agentic-run", "fixture.agentic"
        )
        self.assertEqual(len(normalized_events), 2)
        self.assertTrue(normalized_events[1]["selected_for_executed_cosim"])

    def test_compile_rejection_is_a_complete_zero_synthesis_event(self):
        orch = _orchestrator()
        _record_response(orch, 75)
        orch._finalize_candidate_evaluation(
            code="not valid C++",
            correctness_status="not_run",
            synthesis_status="not_run",
            failure_class="compile_or_interface_failure",
        )
        summary = orch._synthesis_evaluation_summary()
        root = {
            "llm_usage": orch._llm_usage_summary(),
            "synthesis_evaluations": summary,
            "selected_winner_cosim_count": 0,
            "total_synthesis_calls": 0,
            "selected_code_sha256": None,
            "cosim_target_code_sha256": None,
        }
        normalized = _agentic_candidate_events(
            root, "compile-rejected", "fixture.compile"
        )
        self.assertTrue(summary["complete_candidate_event_stream"])
        self.assertEqual(summary["count"], 0)
        self.assertEqual(normalized[0]["failure_class"],
                         "compile_or_interface_failure")
        self.assertEqual(normalized[0]["cumulative_synthesis_evaluations"], 0)

    def test_cosim_target_hash_tamper_fails_runner_contract(self):
        orch = _orchestrator()
        _record_response(orch, 50)
        code = "void workload() {}"
        report = _report(500)
        accepted = {
            "synth": {"success": True, "ran": True, "report": report, "error": ""},
            "csim": _passing_csim(),
            "cosim": None,
        }
        with patch.object(c2hls, "_run_synth_csim_cosim", return_value=accepted):
            orch._synth_and_test(code)
        orch.hls_code = code
        with (
            patch.dict(os.environ, {c2hls.COSIM_SELECTED_ONLY_ENV: "1"}, clear=False),
            patch.object(
                c2hls,
                "run_cosim",
                return_value={
                    "success": True,
                    "passed": True,
                    "error": "",
                    "kernel_runtime_cycles": 525,
                },
            ),
        ):
            orch._run_selected_winner_cosim()
        result = {
            "llm_usage": orch._llm_usage_summary(),
            "synthesis_evaluations": orch._synthesis_evaluation_summary(),
            "selected_winner_cosim_count": 1,
            "total_synthesis_calls": orch._total_synthesis_calls(),
            "selected_code_sha256": orch.selected_code_sha256,
            "cosim_target_code_sha256": orch.cosim_target_code_sha256,
        }
        self.assertTrue(
            run_agentic_sweep._candidate_telemetry_contract(result)["complete"]
        )
        tampered = copy.deepcopy(result)
        tampered["cosim_target_code_sha256"] = "0" * 64
        self.assertFalse(
            run_agentic_sweep._candidate_telemetry_contract(tampered)["complete"]
        )


if __name__ == "__main__":
    unittest.main()
