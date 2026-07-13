from __future__ import annotations

import copy
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402
import hls_eval  # noqa: E402
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
    orch.post_route_implementation_count = 0
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
            **orch._tool_call_attribution(),
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
            **orch._tool_call_attribution(),
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
            **orch._tool_call_attribution(),
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
        tampered_post_route = copy.deepcopy(result)
        tampered_post_route["post_route_implementation_count"] = 1
        self.assertFalse(
            run_agentic_sweep._candidate_telemetry_contract(tampered_post_route)[
                "synthesis_attribution_complete"
            ]
        )
        tampered_total_tools = copy.deepcopy(result)
        tampered_total_tools["total_tool_calls"] -= 1
        self.assertFalse(
            run_agentic_sweep._candidate_telemetry_contract(tampered_total_tools)[
                "synthesis_attribution_complete"
            ]
        )

    def test_post_route_attempt_is_separate_and_included_in_total_tool_calls(self):
        orch = _orchestrator()
        orch.hls_code = "void workload() { ap_uint<512> wide; }"
        orch.synthesis_eval_count = 2
        orch.selected_winner_cosim_count = 1
        variant = {
            "bench_dir": "/public/nova/pathfinder/step3",
            "kernel_basename": "pathfinder",
            "variant_step": "step3",
            "variant_name": "pathfinder-step3",
            "variant_index": 3,
            "source_repo": "rodinia-hls-nova",
        }
        hw_result = {
            "ran": True,
            "passed": False,
            "success": False,
            "implementation_call_count": 1,
            "kernel_runtime_us": None,
            "error": "implementation failed",
            "log": "large tool log",
        }
        with tempfile.TemporaryDirectory() as tmp:
            emu_script = Path(tmp) / "setup_emu_env.sh"
            emu_script.write_text("export XCL_EMULATION_MODE=hw_emu\n", encoding="utf-8")
            emu_script_sha256 = hashlib.sha256(emu_script.read_bytes()).hexdigest()
            environment = {
                "C2HLS_HW_EMU_FINAL": "1",
                "C2HLS_ALLOW_WIDE_ABI": "1",
                "C2HLS_EMU_ENV_SCRIPT": str(emu_script),
                "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS": "1",
                "C2HLS_HW_EMU_CLOCK_MHZ": "300",
                "C2HLS_HW_EMU_CLOCK_NS": "3.33",
                "C2HLS_HW_EMU_TIMEOUT": "123",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(c2hls, "_resolve_rodinia_variant", return_value=(variant, "")),
                patch.object(hls_eval, "run_hw_emu_via_nova", return_value=hw_result) as run_hw,
            ):
                results = {}
                c2hls._maybe_run_hw_emu_final(
                    orch, results, "pathfinder", variant_step="step3"
                )

        run_hw.assert_called_once_with(
            variant["bench_dir"],
            orch.hls_code,
            kernel_basename=variant["kernel_basename"],
            timeout=123,
        )
        attribution = orch._tool_call_attribution()
        self.assertEqual(2, attribution["synthesis_evaluation_count"])
        self.assertEqual(1, attribution["selected_winner_cosim_count"])
        self.assertEqual(1, attribution["post_route_implementation_count"])
        self.assertEqual(4, attribution["total_synthesis_calls"])
        self.assertEqual(4, attribution["total_tool_calls"])
        self.assertNotIn("log", results["hw_emu"])
        self.assertEqual(1, results["hw_emu"]["implementation_call_count"])
        configuration = results["post_route_configuration"]
        self.assertTrue(configuration["allow_wide_abi"])
        self.assertTrue(configuration["disable_debug_symbols"])
        self.assertEqual("300", configuration["clock_mhz_override"])
        self.assertEqual("3.33", configuration["clock_ns_override"])
        self.assertEqual(
            emu_script_sha256,
            configuration["emu_env_script"]["sha256"],
        )

    def test_hw_emu_producer_counts_failed_invocation_but_not_staging_failure(self):
        with patch.object(
            hls_eval, "_stage_nova_workdir", return_value=(None, "missing benchmark")
        ):
            not_staged = hls_eval.run_hw_emu_via_nova("/missing")
        self.assertEqual(0, not_staged["implementation_call_count"])

        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp)
            with (
                patch.object(
                    hls_eval, "_stage_nova_workdir", return_value=(staged, staged)
                ),
                patch.object(
                    hls_eval,
                    "_run_make_check_emu",
                    return_value=("ERROR: link failed", False),
                ),
                patch.object(hls_eval, "_choose_latest", return_value=None),
                patch.object(hls_eval, "_parse_runtime_us", return_value=(None, 0)),
                patch.object(hls_eval, "_find_hw_emu_crash_marker", return_value=("", "")),
                patch.object(
                    hls_eval,
                    "_resolve_hw_emu_clock",
                    return_value=(300.0, None, True, "u280_default"),
                ),
            ):
                attempted = hls_eval.run_hw_emu_via_nova("/public/nova")
        self.assertTrue(attempted["ran"])
        self.assertFalse(attempted["success"])
        self.assertEqual(1, attempted["implementation_call_count"])

    def test_disabled_post_route_skip_consumes_zero_calls(self):
        orch = _orchestrator()
        orch.hls_code = "void workload() {}"
        orch.synthesis_eval_count = 1
        with patch.dict(os.environ, {}, clear=True):
            results = {}
            c2hls._maybe_run_hw_emu_final(orch, results, "pathfinder")
        self.assertFalse(results["hw_emu"]["ran"])
        self.assertEqual(0, results["hw_emu"]["implementation_call_count"])
        self.assertEqual(0, orch.post_route_implementation_count)
        self.assertEqual(1, orch._tool_call_attribution()["total_tool_calls"])


if __name__ == "__main__":
    unittest.main()
