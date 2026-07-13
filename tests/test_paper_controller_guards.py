from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import c2hls  # noqa: E402
import hls_eval  # noqa: E402


PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33


def feasible_report(cycles: int, **overrides) -> dict:
    report = {
        "latency_cycles": cycles,
        "latency_cycles_worst": cycles,
        "bram": 4,
        "dsp": 8,
        "ff": 1_000,
        "lut": 2_000,
        "uram": 0,
        "slack_ns": 0.05,
        "requested_clock_period_ns": CLOCK_NS,
    }
    report.update(overrides)
    return report


def passing_csim() -> dict:
    return {
        "status": "passed",
        "supported": True,
        "ran": True,
        "success": True,
        "passed": True,
        "error": "",
    }


def make_run_multistep_harness(*, dynamic: bool = False):
    """Construct an orchestrator without constructing an API client."""
    orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
    orch.run_phase_a = MagicMock(return_value=True)
    orch.run_phase_b = MagicMock(return_value=True)
    orch._baseline_alignment_loop = MagicMock(return_value={"attempted": False})
    orch._record_phase_b_fast_candidate = MagicMock()
    orch.run_phase_c = MagicMock(return_value={"success": True})
    orch._record_best_so_far = MagicMock()
    orch._promote_best_so_far = MagicMock(return_value=None)
    orch._run_selected_winner_cosim = MagicMock(return_value=None)
    orch._llm_usage_summary = MagicMock(return_value={"calls": 0})
    orch._synthesis_evaluation_summary = MagicMock(
        return_value={"count": 0, "budget": 5}
    )
    orch.run_optimization_step = MagicMock(
        return_value={"step_name": "tiling", "success": False, "error": "mocked"}
    )
    orch.run_optimization_step_forward = MagicMock()
    orch._append_history = MagicMock()

    orch.hls_code = "void workload() {}"
    orch.synth_report = feasible_report(100)
    orch.generated_csim = passing_csim()
    orch.generated_cosim = None
    orch.testbench_code = ""
    orch.header_code = ""
    orch.header_name = "kernel.h"
    orch.translated_hls_top = "workload"
    orch.reference_hls_top = "workload"
    orch.part = PART
    orch.clock_ns = CLOCK_NS
    orch.supports_cosim = False
    orch.cosim_depths = {}
    orch.independent_golden_output = ""
    orch.independent_golden_specs = {}
    orch.independent_golden_provenance = {}
    orch.preflight_patches = []
    orch.phase_b_fast_candidate = None
    orch.phaseb_mode = "canonical"
    orch.history = []
    orch.robustness_log = []
    orch.skill_library = None
    orch.skill_library_provenance = {}
    orch.vitis_version = "2023.2"
    orch.strategy = "dynamic" if dynamic else "static"
    orch.dynamic_routing = dynamic
    orch.reference_blind = False
    orch._baseline_report = {}
    orch._gt_step_reports = {"tiling": feasible_report(50)}
    orch._gt_baseline_report = feasible_report(80)
    orch.cosim_reference_cycle_info = {"cycles": 50, "source": "expert"}
    return orch


class ReferenceBlindControllerTests(unittest.TestCase):
    def test_frozen_skill_snapshot_is_loaded_exactly_before_phase_a(self):
        orch = make_run_multistep_harness(dynamic=True)
        no_step = SimpleNamespace(
            step_name="",
            reason="done",
            bottleneck_kind=None,
            skill_id=None,
            confidence=None,
            fallback=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "skills.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "schema": "1.1",
                        "skills": [
                            {
                                "id": "validated-only",
                                "pattern": "p",
                                "strategy": "s",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "C2HLS_SKILL_MODE": "skill_on",
                        c2hls.REFERENCE_BLIND_ENV: "1",
                        c2hls.SKILL_LIBRARY_FROZEN_ENV: "1",
                        "C2HLS_SKILL_LIBRARY_PATH": str(snapshot),
                        c2hls.FEASIBILITY_SELECTION_ENV: "1",
                    },
                    clear=False,
                ),
                patch("bottleneck_router.select_next_step", return_value=no_step),
            ):
                success, result = c2hls.C2HLSOrchestrator.run_multistep(
                    orch, "void workload() {}", steps=["tiling"]
                )

        self.assertTrue(success)
        self.assertEqual(
            ["validated-only"], result["skill_library_provenance"]["loaded_skill_ids"]
        )
        self.assertEqual(
            "exact_frozen_snapshot",
            result["skill_library_provenance"]["source_mode"],
        )
        self.assertFalse(result["skill_library_provenance"]["package_merged"])
        orch.run_phase_a.assert_called_once()

    def test_invalid_frozen_skill_snapshot_fails_before_phase_a(self):
        orch = make_run_multistep_harness(dynamic=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "skills.json"
            snapshot.write_text('{"schema":"1.1","skills":[]}\n', encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "C2HLS_SKILL_MODE": "skill_on",
                    c2hls.REFERENCE_BLIND_ENV: "1",
                    c2hls.SKILL_LIBRARY_FROZEN_ENV: "1",
                    "C2HLS_SKILL_LIBRARY_PATH": str(snapshot),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "non-empty skills"):
                    c2hls.C2HLSOrchestrator.run_multistep(
                        orch, "void workload() {}", steps=["tiling"]
                    )

        orch.run_phase_a.assert_not_called()

    def test_skill_off_removes_library_from_dynamic_router(self):
        orch = make_run_multistep_harness(dynamic=True)
        orch.skill_library = MagicMock()
        no_step = SimpleNamespace(
            step_name="",
            reason="done",
            bottleneck_kind=None,
            skill_id=None,
            confidence=None,
            fallback=True,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "C2HLS_SKILL_MODE": "skill_off",
                    c2hls.REFERENCE_BLIND_ENV: "1",
                    c2hls.FEASIBILITY_SELECTION_ENV: "1",
                },
                clear=False,
            ),
            patch(
                "bottleneck_router.select_next_step", return_value=no_step
            ) as router,
        ):
            success, result = c2hls.C2HLSOrchestrator.run_multistep(
                orch, "void workload() {}", steps=["tiling"]
            )

        self.assertTrue(success)
        self.assertIsNone(router.call_args.kwargs["library"])
        self.assertFalse(result["skill_library_provenance"]["control_enabled"])

    def test_reference_blind_clears_all_gt_controller_inputs(self):
        orch = make_run_multistep_harness()
        env = {
            c2hls.REFERENCE_BLIND_ENV: "1",
            c2hls.GT_COMPARISON_IN_CONTROL_ENV: "1",
            "C2HLS_PHASE5_GT_PREPOP": "1",
            c2hls.FEASIBILITY_SELECTION_ENV: "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(c2hls, "run_hls_synthesis") as gt_synth,
        ):
            success, _ = c2hls.C2HLSOrchestrator.run_multistep(
                orch,
                "void workload() {}",
                steps=["tiling"],
                gt_variants={"tiling": "// EXPERT SECRET\nvoid workload() {}"},
                gt_variant_headers={"tiling": "// EXPERT HEADER"},
                reference_report=feasible_report(10),
            )

        self.assertTrue(success)
        orch._baseline_alignment_loop.assert_called_once_with({})
        orch._record_phase_b_fast_candidate.assert_called_once_with({})
        orch.run_phase_c.assert_not_called()
        gt_synth.assert_not_called()
        orch.run_optimization_step.assert_called_once_with(
            "tiling", gt_code=None, gt_header_code=None
        )
        self.assertEqual(orch._gt_step_reports, {})
        self.assertEqual(orch._gt_baseline_report, {})
        self.assertEqual(orch.cosim_reference_cycle_info, {})

    def test_frozen_skill_library_cannot_update_promote_or_save(self):
        orch = make_run_multistep_harness(dynamic=True)
        library = MagicMock()
        library.all.return_value = [SimpleNamespace(skill_id="tiling")]
        library.store_path = "/tmp/frozen-skills.json"
        orch.skill_library = library
        orch.skill_library_provenance = {"sha256": "frozen-snapshot"}
        orch.run_optimization_step.return_value = {
            "step_name": "tiling",
            "success": True,
            "report": feasible_report(90),
            "csim": passing_csim(),
            "code": "void workload() {}",
        }
        decision = SimpleNamespace(
            step_name="tiling",
            reason="mock bottleneck",
            bottleneck_kind="latency",
            skill_id="tiling",
            confidence=1.0,
            fallback=False,
        )
        env = {
            c2hls.REFERENCE_BLIND_ENV: "0",
            c2hls.GT_COMPARISON_IN_CONTROL_ENV: "0",
            c2hls.SKILL_LIBRARY_FROZEN_ENV: "1",
            c2hls.SKILL_UPDATE_STATS_ENV: "1",
            "C2HLS_SKILL_LIBRARY_PERSIST": "1",
            c2hls.FEASIBILITY_SELECTION_ENV: "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch("bottleneck_router.select_next_step", return_value=decision),
        ):
            success, result = c2hls.C2HLSOrchestrator.run_multistep(
                orch,
                "void workload() {}",
                steps=["tiling"],
                reference_report={},
            )

        self.assertTrue(success)
        library.update_skill_statistics.assert_not_called()
        library.promote_demote.assert_not_called()
        library.save.assert_not_called()
        self.assertNotIn("skill_update", result["steps"][0])


class MatchedBudgetAndSelectionTests(unittest.TestCase):
    def test_paper_correctness_gate_skips_synthesis_for_bad_candidate(self):
        bad_csim = {
            "success": False,
            "passed": False,
            "error": "golden mismatch",
            "log": "mismatch",
        }
        with (
            patch.dict(
                os.environ,
                {c2hls.CORRECTNESS_BEFORE_SYNTH_ENV: "1"},
                clear=False,
            ),
            patch.object(c2hls, "run_csim", return_value=bad_csim) as csim,
            patch.object(c2hls, "run_hls_synthesis") as synth,
        ):
            outcome = c2hls._run_synth_csim_cosim(
                "void workload() {}",
                header_code="",
                header_name="kernel.h",
                top_function="workload",
                part=PART,
                clock_ns=CLOCK_NS,
                extra_files=[],
                testbench_code="int main() { return 1; }",
                run_csim_check=True,
            )

        csim.assert_called_once()
        synth.assert_not_called()
        self.assertFalse(outcome["synth"]["ran"])
        self.assertEqual(
            outcome["synth"]["skip_reason"], "csim_correctness_gate_failed"
        )

    def test_total_synthesis_budget_rejects_before_invoking_tool(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.synthesis_eval_budget = 1
        orch.synthesis_eval_count = 0
        orch.synthesis_eval_events = []
        orch.header_code = ""
        orch.header_name = "kernel.h"
        orch.translated_hls_top = "workload"
        orch.part = PART
        orch.clock_ns = CLOCK_NS
        orch.extra_files = []
        orch.testbench_code = "int main() { return 0; }"
        orch.supports_cosim = True
        orch.cosim_depths = {}
        orch.reference_blind = True
        orch.cosim_reference_cycle_info = {"cycles": 1, "source": "expert"}
        orch.independent_golden_output = ""
        orch.independent_golden_specs = {}
        tool_result = {
            "synth": {"success": True, "report": feasible_report(100), "error": ""},
            "csim": passing_csim(),
            "cosim": None,
        }
        with patch.object(
            c2hls, "_run_synth_csim_cosim", return_value=tool_result
        ) as synth_tool:
            first = c2hls.C2HLSOrchestrator._synth_and_test(orch, "candidate 1")
            second = c2hls.C2HLSOrchestrator._synth_and_test(orch, "candidate 2")

        self.assertTrue(first["synth"]["success"])
        self.assertTrue(second["synth"]["budget_exhausted"])
        self.assertEqual(orch.synthesis_eval_count, 1)
        self.assertEqual(len(orch.synthesis_eval_events), 1)
        synth_tool.assert_called_once()
        self.assertEqual(
            synth_tool.call_args.kwargs["cosim_reference_cycle_info"], {}
        )

    def test_llm_candidate_budget_rejects_before_second_provider_call(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.max_completion_tokens = 128
        orch.gpt_model = "local-test-model"
        orch.llm_candidate_budget = 1
        orch.llm_candidate_request_count = 0
        orch.llm_usage_events = []
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="candidate"))],
            usage=SimpleNamespace(
                prompt_tokens=3, completion_tokens=2, total_tokens=5
            ),
        )
        create = MagicMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        orch._client_for_model = MagicMock(return_value=("openai", client))

        with patch.dict(
            os.environ,
            {
                c2hls.LLM_TEMPERATURE_ENV: "0.2",
                c2hls.LLM_TOP_P_ENV: "0.9",
                c2hls.LLM_SEED_ENV: "7",
            },
            clear=False,
        ):
            first = c2hls.C2HLSOrchestrator._call_llm_with_model(
                orch, [{"role": "user", "content": "optimize"}]
            )
            with self.assertRaisesRegex(
                RuntimeError, "llm_candidate_budget_exhausted"
            ):
                c2hls.C2HLSOrchestrator._call_llm_with_model(
                    orch, [{"role": "user", "content": "repair"}]
                )

        self.assertEqual(first, "candidate")
        create.assert_called_once()
        self.assertEqual(orch.llm_candidate_request_count, 1)
        self.assertEqual(create.call_args.kwargs["temperature"], 0.2)
        self.assertEqual(create.call_args.kwargs["top_p"], 0.9)
        self.assertEqual(create.call_args.kwargs["seed"], 7)

    def test_reference_blind_mode_removes_handwritten_benchmark_policy(self):
        with patch.dict(
            os.environ, {c2hls.REFERENCE_BLIND_ENV: "1"}, clear=False
        ):
            self.assertEqual([], c2hls._policy("srad", "translation", []))
            self.assertIsNone(c2hls._policy("StreamCluster", "priority"))
        with patch.dict(
            os.environ, {c2hls.REFERENCE_BLIND_ENV: "0"}, clear=False
        ):
            self.assertTrue(c2hls._policy("srad", "translation", []))

    def test_reference_blind_quality_helpers_ignore_oracle_and_benchmark(self):
        report = feasible_report(100)
        fake_oracle = {
            "latency_ns": {"ratio": 99.0},
            "latency_cycles": {"ratio": 88.0},
            "fmax_mhz": {"ratio": 0.01},
        }
        with patch.dict(
            os.environ,
            {
                c2hls.REFERENCE_BLIND_ENV: "1",
                "C2HLS_PART": PART,
                "C2HLS_CLOCK_NS": str(CLOCK_NS),
            },
            clear=False,
        ):
            stream_guidance = c2hls._build_quality_guidance(
                "StreamCluster", report, feasible_report(1), fake_oracle
            )
            unknown_guidance = c2hls._build_quality_guidance(
                "unseen_kernel", report, feasible_report(999), {}
            )
            context = c2hls._build_quality_context(report, fake_oracle)
            stream_score = c2hls._quality_score(
                "StreamCluster", report, fake_oracle
            )
            unknown_score = c2hls._quality_score("unseen_kernel", report, {})
            stream_focus = c2hls._quality_focus(
                "StreamCluster", report, fake_oracle
            )
            unknown_focus = c2hls._quality_focus("unseen_kernel", report, {})
            ratios = c2hls._reference_ratio_summary(fake_oracle)

        self.assertEqual(stream_guidance, unknown_guidance)
        self.assertNotIn("reference", stream_guidance.lower())
        self.assertNotIn("ratio", stream_guidance.lower())
        self.assertIn("no expert comparison", context.lower())
        self.assertNotIn("_ratio=", context)
        self.assertEqual(stream_score, unknown_score)
        self.assertEqual("latency", stream_focus)
        self.assertEqual(stream_focus, unknown_focus)
        self.assertEqual({}, ratios)

    def test_reference_blind_quality_improvement_uses_absolute_latency(self):
        current = feasible_report(100)
        candidate = feasible_report(80)
        misleading_current = {"latency_ns": {"ratio": 0.1}}
        misleading_candidate = {"latency_ns": {"ratio": 10.0}}
        with patch.dict(
            os.environ,
            {
                c2hls.REFERENCE_BLIND_ENV: "1",
                "C2HLS_PART": PART,
                "C2HLS_CLOCK_NS": str(CLOCK_NS),
            },
            clear=False,
        ):
            improved = c2hls._quality_focus_improved(
                "StreamCluster",
                "latency",
                current,
                misleading_current,
                candidate,
                misleading_candidate,
            )
        self.assertTrue(improved)

    def test_reference_blind_srad_preflight_does_not_semantically_patch_code(self):
        code = (
            "void workload(float *J, float *Jout) {\n"
            "  memcpy(Jout + t*TILE_ROWS*COLS, local, sizeof(local));\n"
            "  memcpy(J + t*TILE_ROWS*COLS, local, sizeof(local));\n"
            "}\n"
        )
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.header_code = ""
        orch.testbench_code = ""
        orch.translated_hls_top = "workload"
        orch.benchmark_name = "srad"
        orch.preflight_patches = []
        orch.history = []
        with patch.dict(
            os.environ, {c2hls.REFERENCE_BLIND_ENV: "1"}, clear=False
        ):
            blind = c2hls.C2HLSOrchestrator._preflight_generated_hls_code(
                orch, code, "test"
            )
        self.assertEqual(code, blind)
        self.assertEqual([], orch.preflight_patches)

        with patch.dict(
            os.environ, {c2hls.REFERENCE_BLIND_ENV: "0"}, clear=False
        ):
            legacy = c2hls.C2HLSOrchestrator._preflight_generated_hls_code(
                orch, code, "test"
            )
        self.assertIn("(t*TILE_ROWS+1)*COLS", legacy)
        self.assertTrue(orch.preflight_patches)

    def test_reference_blind_regression_guard_uses_only_generic_hard_gates(self):
        previous = feasible_report(100)
        current = feasible_report(
            1_000,
            bram=400,
            dsp=800,
            ff=100_000,
            lut=200_000,
            slack_ns=0.01,
        )
        with patch.dict(
            os.environ,
            {
                c2hls.REFERENCE_BLIND_ENV: "1",
                "C2HLS_STEP_REGRESSION_THRESHOLDS_JSON": json.dumps({
                    "tiling": {"latency": 1.01, "resources": {"default": 1.01}}
                }),
            },
            clear=False,
        ):
            thresholds = c2hls._resolve_step_thresholds("tiling")
            reasons = c2hls._step_regression_reasons(
                current, previous, step_name="tiling", part=PART
            )
            rendered = c2hls._render_step_resource_constraints(
                "tiling", current, PART
            )
        self.assertEqual(float("inf"), thresholds["latency"])
        self.assertEqual([], reasons)
        self.assertIn("No benchmark-specific", rendered)
        self.assertNotIn("automatic revert", rendered)

    def test_feasibility_filter_beats_a_faster_infeasible_candidate(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.turns_limitation = 1
        orch._append_history = MagicMock()
        fast_but_infeasible = {
            "success": True,
            "step_name": "tiling",
            "code": "// fast but timing failed",
            "report": feasible_report(10, slack_ns=-0.5),
            "feasibility": {
                "feasible": False,
                "reasons": ["target_timing_failed"],
            },
        }
        slower_feasible = {
            "success": True,
            "step_name": "tiling",
            "code": "// slower and feasible",
            "report": feasible_report(30),
            "feasibility": {"feasible": True, "reasons": []},
        }
        orch._optimization_step_attempt_single = MagicMock(
            side_effect=[fast_but_infeasible, slower_feasible]
        )
        with patch.dict(
            os.environ,
            {
                c2hls.REFERENCE_BLIND_ENV: "1",
                c2hls.STEP_CANDIDATES_ENV: "2",
                c2hls.EXHAUSTIVE_CANDIDATE_ATTEMPTS_ENV: "0",
            },
            clear=False,
        ):
            selected = c2hls.C2HLSOrchestrator._optimization_step_attempt(
                orch, "tiling"
            )

        self.assertTrue(selected["success"])
        self.assertEqual(selected["code"], "// slower and feasible")
        self.assertEqual(selected["selected_candidate_index"], 1)
        self.assertEqual(selected["candidate_search"]["feasible_candidates"], 1)
        self.assertEqual(
            selected["candidate_attempts"][0]["feasibility"]["reasons"],
            ["target_timing_failed"],
        )

    def test_u280_feasibility_rejects_missing_uram_evidence(self):
        report = feasible_report(30)
        report.pop("uram")
        result = c2hls._paper_candidate_feasibility(
            report,
            csim=passing_csim(),
            part=PART,
            clock_ns=CLOCK_NS,
        )
        self.assertFalse(result["feasible"])
        self.assertFalse(result["resource_evidence_complete"])
        self.assertIn("resource_evidence_incomplete", result["reasons"])

    def test_candidate_evaluations_skip_cosim_then_selected_winner_runs_once(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.synthesis_eval_budget = 5
        orch.synthesis_eval_count = 0
        orch.synthesis_eval_events = []
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
        orch.hls_code = "void workload() { /* selected */ }"
        orch.generated_cosim = None
        tool_result = {
            "synth": {"success": True, "report": feasible_report(100), "error": ""},
            "csim": passing_csim(),
            "cosim": None,
        }
        raw_cosim = {
            "success": True,
            "passed": True,
            "error": "",
            "log": "RTL simulation passed",
            "kernel_runtime_cycles": 123,
        }
        env = {
            c2hls.COSIM_SELECTED_ONLY_ENV: "1",
            "C2HLS_COSIM_REQUIRED": "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                c2hls, "_run_synth_csim_cosim", return_value=tool_result
            ) as candidate_tool,
            patch.object(c2hls, "run_cosim", return_value=raw_cosim) as cosim_tool,
        ):
            c2hls.C2HLSOrchestrator._synth_and_test(orch, "candidate 1")
            c2hls.C2HLSOrchestrator._synth_and_test(orch, "candidate 2")
            selected_cosim = c2hls.C2HLSOrchestrator._run_selected_winner_cosim(orch)

        self.assertEqual(candidate_tool.call_count, 2)
        self.assertTrue(
            all(not item.kwargs["run_cosim_check"] for item in candidate_tool.call_args_list)
        )
        cosim_tool.assert_called_once()
        self.assertEqual(cosim_tool.call_args.args[0], orch.hls_code)
        self.assertTrue(selected_cosim["passed"])
        self.assertEqual(selected_cosim["kernel_runtime_cycles"], 123)


class IndependentGoldenAndExpertFrontierTests(unittest.TestCase):
    def test_cpu_golden_preparation_rejects_a_corrupted_candidate(self):
        kernel = (
            '#include "kernel.h"\n'
            "void kernel(int out[2]) { out[0] = 7; out[1] = 11; }\n"
        )
        header = "void kernel(int out[2]);\n"
        testbench = r'''
#include <cstdio>
#include "kernel.h"
int main() {
    int out[2] = {0, 0};
    kernel(out);
    std::fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    std::fprintf(stderr, "begin dump: out\n%d %d\n", out[0], out[1]);
    std::fprintf(stderr, "end dump: out\n");
    std::fprintf(stderr, "==END DUMP_ARRAYS==\n");
    return 0;
}
'''
        inputs = {
                "meta": {
                    "source_repo": "synthetic_unit_test",
                    "independent_golden_required": True,
                    "golden_output_specs": {"out": {"shape": [2], "kind": "integer"}},
                },
            "c_code": kernel,
            "header_code": header,
            "header_name": "kernel.h",
            "testbench_code": testbench,
            "extra_files": [],
        }

        # The repository's normal temp root can be a cluster-local mount.
        # Keep this unit/integration test self-contained in the writable /tmp.
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir, patch.object(
            hls_eval, "make_tempdir", return_value=str(Path(tmpdir) / "cpu_golden")
        ):
            prepared = c2hls._prepare_independent_golden(inputs)

        self.assertTrue(prepared["success"], prepared.get("error"))
        self.assertTrue(prepared["required"])
        self.assertEqual(prepared["provenance"]["status"], "passed")
        self.assertEqual(prepared["provenance"]["outputs"]["out"]["count"], 2)
        self.assertNotIn("output", prepared["provenance"])
        corrupted = prepared["output"].replace("7 11", "7 99")
        gated = hls_eval._apply_independent_golden(
            {
                "success": True,
                "passed": True,
                "error": "",
                "log": corrupted,
            },
            golden_output_text=prepared["output"],
            golden_output_specs=prepared["specs"],
        )
        self.assertFalse(gated["success"])
        self.assertFalse(gated["passed"])
        self.assertEqual(gated["correctness"]["correctness_status"], "failed")
        self.assertEqual(gated["correctness"]["reason"], "integer_mismatch")

    def test_expert_frontier_selects_fastest_correct_feasible_variant(self):
        candidates = [
            {"variant_name": "baseline", "file": "baseline.cpp", "step_name": "baseline", "code": "b"},
            {"variant_name": "preferred", "file": "preferred.cpp", "step_name": "pipeline", "code": "p"},
            {"variant_name": "fastest", "file": "fastest.cpp", "step_name": "unroll", "code": "f"},
            {"variant_name": "too_fast", "file": "too_fast.cpp", "step_name": "partition", "code": "x"},
        ]

        def workflow_entry(candidate: dict, cycles: int, *, ready=True, feasible=True):
            report = feasible_report(cycles)
            return {
                "variant_name": candidate["variant_name"],
                "file": candidate["file"],
                "step_name": candidate["step_name"],
                "source_path": candidate["file"],
                "benchmark_ready": ready,
                "invalid_reason": "" if ready else "target timing failed",
                "synthesis": {"status": "passed", "success": True, "report": report},
                "csim": passing_csim(),
                "cosim": {"status": "not_run", "supported": True, "ran": False, "passed": False},
                "report": report,
                "selected": False,
                "feasibility": {
                    "feasible": feasible,
                    "reasons": [] if feasible else ["target_timing_failed"],
                },
            }

        entries = {
            "baseline.cpp": workflow_entry(candidates[0], 100),
            "preferred.cpp": workflow_entry(candidates[1], 80),
            "fastest.cpp": workflow_entry(candidates[2], 40),
            "too_fast.cpp": workflow_entry(
                candidates[3], 10, ready=False, feasible=False
            ),
        }

        def validate(candidate, *_args, **_kwargs):
            return copy.deepcopy(entries[candidate["file"]])

        inputs = {
            "meta": {
                "benchmark": "frontier_test",
                "source_repo": "Rodinia-HLS",
                "hls_top": "workload",
                "part": PART,
                "clock_ns": CLOCK_NS,
                "supports_csim": True,
                # Exercise the paper's forced selected-reference policy even
                # when legacy metadata says generated cosim is unsupported.
                "supports_cosim": False,
                "preferred_gt_file": "preferred.cpp",
            },
            "testbench_code": "int main() { return 0; }",
            "extra_files": [],
        }
        env = {
            c2hls.REFERENCE_BLIND_ENV: "1",
            "C2HLS_REFERENCE_VALIDATE_MODE": "all",
            "C2HLS_REFERENCE_COSIM": "0",
            c2hls.REFERENCE_COSIM_SELECTED_ONLY_ENV: "1",
            c2hls.REFERENCE_COSIM_BASELINE_ENV: "1",
            c2hls.COSIM_SELECTED_ONLY_ENV: "1",
            c2hls.FORCE_SELECTED_COSIM_ENV: "1",
        }

        def reference_cosim(code, *_args, **_kwargs):
            return {
                "success": True,
                "passed": True,
                "kernel_runtime_cycles": 55 if code == "f" else 125,
            }

        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(c2hls, "_ground_truth_candidates", return_value=candidates),
            patch.object(
                c2hls, "_validate_ground_truth_candidate", side_effect=validate
            ) as validate_mock,
            patch.object(c2hls, "compare_reports", return_value={}),
            patch.object(
                c2hls,
                "run_cosim",
                side_effect=reference_cosim,
            ) as selected_cosim,
        ):
            frontier = c2hls._validate_gold_reference_uncached(inputs)
            self.assertTrue(c2hls._reference_validation_cacheable(inputs, frontier))

        self.assertTrue(frontier["benchmark_ready"])
        self.assertEqual(frontier["selected_variant_file"], "fastest.cpp")
        self.assertEqual(frontier["report"]["latency_cycles_worst"], 40)
        self.assertIn("fastest correct", frontier["selection_reason"])
        self.assertTrue(frontier["selected_reference_cosim_measurement_valid"])
        self.assertTrue(frontier["baseline_reference_cosim_measurement_valid"])
        self.assertTrue(frontier["rtl_measurement_pair_valid"])
        self.assertEqual(55, frontier["cosim"]["kernel_runtime_cycles"])
        self.assertEqual(
            125,
            frontier["baseline_reference"]["cosim"]["kernel_runtime_cycles"],
        )
        self.assertEqual(2, selected_cosim.call_count)
        self.assertEqual(
            ["f", "b"],
            [call.args[0] for call in selected_cosim.call_args_list],
        )
        self.assertEqual(validate_mock.call_count, 4)
        self.assertTrue(
            all(
                item.kwargs["run_cosim_check"] is False
                for item in validate_mock.call_args_list
            )
        )
        selected = [item["file"] for item in frontier["workflow"] if item["selected"]]
        self.assertEqual(selected, ["fastest.cpp"])

        def expert_pass_baseline_missing(code, *_args, **_kwargs):
            return {
                "success": True,
                "passed": True,
                "kernel_runtime_cycles": 55 if code == "f" else None,
            }

        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(c2hls, "_ground_truth_candidates", return_value=candidates),
            patch.object(
                c2hls, "_validate_ground_truth_candidate", side_effect=validate
            ),
            patch.object(c2hls, "compare_reports", return_value={}),
            patch.object(
                c2hls, "run_cosim", side_effect=expert_pass_baseline_missing
            ),
        ):
            missing_baseline = c2hls._validate_gold_reference_uncached(inputs)
            self.assertFalse(
                c2hls._reference_validation_cacheable(inputs, missing_baseline)
            )
        self.assertFalse(missing_baseline["benchmark_ready"])
        self.assertFalse(missing_baseline["rtl_measurement_pair_valid"])
        self.assertFalse(
            missing_baseline["baseline_reference_cosim_measurement_valid"]
        )
        self.assertIn("designated baseline", missing_baseline["invalid_reason"])

        all_infeasible = {
            candidate["file"]: workflow_entry(
                candidate, cycles=100 - index * 10, ready=True, feasible=False
            )
            for index, candidate in enumerate(candidates)
        }

        def validate_infeasible(candidate, *_args, **_kwargs):
            return copy.deepcopy(all_infeasible[candidate["file"]])

        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(c2hls, "_ground_truth_candidates", return_value=candidates),
            patch.object(
                c2hls,
                "_validate_ground_truth_candidate",
                side_effect=validate_infeasible,
            ),
            patch.object(c2hls, "compare_reports", return_value={}),
        ):
            rejected = c2hls._validate_gold_reference_uncached(inputs)

        self.assertFalse(rejected["benchmark_ready"])
        self.assertEqual("", rejected["selected_variant_file"])
        self.assertIn("device-fitting", rejected["invalid_reason"])

    def test_reference_baseline_selected_as_expert_runs_one_cosim(self):
        candidate = {
            "variant_name": "baseline",
            "file": "baseline.cpp",
            "step_name": "baseline",
            "code": "b",
        }
        report = feasible_report(100)
        entry = {
            "variant_name": "baseline",
            "file": "baseline.cpp",
            "step_name": "baseline",
            "source_path": "baseline.cpp",
            "benchmark_ready": True,
            "invalid_reason": "",
            "synthesis": {"status": "passed", "success": True, "report": report},
            "csim": passing_csim(),
            "cosim": {
                "status": "not_run",
                "supported": True,
                "ran": False,
                "passed": False,
            },
            "report": report,
            "selected": False,
            "feasibility": {"feasible": True, "reasons": []},
        }
        inputs = {
            "meta": {
                "benchmark": "baseline_only",
                "source_repo": "Rodinia-HLS",
                "hls_top": "workload",
                "part": PART,
                "clock_ns": CLOCK_NS,
                "supports_csim": True,
                "supports_cosim": False,
            },
            "testbench_code": "int main() { return 0; }",
            "extra_files": [],
        }
        env = {
            c2hls.REFERENCE_BLIND_ENV: "1",
            "C2HLS_REFERENCE_VALIDATE_MODE": "all",
            "C2HLS_REFERENCE_COSIM": "0",
            c2hls.REFERENCE_COSIM_SELECTED_ONLY_ENV: "1",
            c2hls.REFERENCE_COSIM_BASELINE_ENV: "1",
            c2hls.COSIM_SELECTED_ONLY_ENV: "1",
            c2hls.FORCE_SELECTED_COSIM_ENV: "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(c2hls, "_ground_truth_candidates", return_value=[candidate]),
            patch.object(
                c2hls,
                "_validate_ground_truth_candidate",
                return_value=copy.deepcopy(entry),
            ),
            patch.object(
                c2hls,
                "run_cosim",
                return_value={
                    "success": True,
                    "passed": True,
                    "kernel_runtime_cycles": 100,
                },
            ) as cosim,
        ):
            frontier = c2hls._validate_gold_reference_uncached(inputs)

        cosim.assert_called_once()
        self.assertTrue(frontier["benchmark_ready"])
        self.assertTrue(frontier["rtl_measurement_pair_valid"])
        self.assertEqual(
            frontier["cosim"]["kernel_runtime_cycles"],
            frontier["baseline_reference"]["cosim"]["kernel_runtime_cycles"],
        )


class PaperTimingScopeTests(unittest.TestCase):
    def test_dynamic_method_wall_time_excludes_common_reference_preflight(self):
        inputs = {
            "bench_name": "timing_test",
            "c_code": "void workload() {}",
            "header_code": "void workload();",
            "header_name": "kernel.h",
            "testbench_code": "int main() { return 0; }",
            "extra_files": [],
            "gt_variants": {},
            "gt_variant_headers": {},
            "benchmark_context": "",
            "meta": {
                "benchmark": "timing_test",
                "translated_hls_top": "workload",
                "hls_top": "workload",
                "part": PART,
                "clock_ns": CLOCK_NS,
                "supports_cosim": False,
                "cosim_depths": {},
            },
        }
        reference = {
            "benchmark_ready": True,
            "invalid_reason": "",
            "reference_source": "local_vitis",
            "synthesis": {"status": "passed"},
            "report": feasible_report(100),
            "workflow": [],
            "selected_variant_name": "baseline",
            "selected_variant_file": "baseline.cpp",
            "selected_variant_step": "baseline",
            "selection_reason": "test",
            "selection_fallback": False,
            "selection_fallback_reason": "",
            "csim": passing_csim(),
            "cosim": {
                "status": "passed",
                "ran": True,
                "passed": True,
                "kernel_runtime_cycles": 100,
            },
        }
        orchestrator = MagicMock()
        orchestrator.cosim_reference_cycle_info = {}
        orchestrator.synth_report = feasible_report(80)
        orchestrator.run_multistep.return_value = (
            True,
            {
                "phase": "multistep",
                "generated_step_history": [],
                "csim": passing_csim(),
                "cosim": {
                    "status": "passed",
                    "ran": True,
                    "passed": True,
                    "kernel_runtime_cycles": 80,
                },
            },
        )
        independent = {
            "success": True,
            "output": "gold",
            "specs": {},
            "provenance": {"status": "passed"},
        }
        # 100 seconds of arbitrary common preflight, five seconds of search,
        # and one second of optional post-route work.
        monotonic_values = [0.0, 100.0, 100.0, 105.0, 105.0, 106.0, 106.0]
        with (
            patch.object(c2hls, "_load_benchmark_inputs", return_value=inputs),
            patch.object(c2hls, "_prepare_independent_golden", return_value=independent),
            patch.object(c2hls, "validate_gold_reference", return_value=reference),
            patch.object(c2hls, "C2HLSOrchestrator", return_value=orchestrator),
            patch.object(c2hls, "_ground_truth_control_enabled", return_value=False),
            patch.object(c2hls, "_build_run_attribution", return_value={}),
            patch.object(c2hls, "_maybe_run_hw_emu_final"),
            patch.object(c2hls, "sanitize_saved_result_record", side_effect=lambda result, _ref: result),
            patch.object(c2hls.time, "monotonic", side_effect=monotonic_values),
        ):
            result = c2hls.run_benchmark_multistep(
                "/unused", output_dir="/tmp/timing-test", steps=["flash"]
            )

        self.assertEqual(100.0, result["preflight_elapsed_seconds"])
        self.assertEqual(5.0, result["search_elapsed_seconds"])
        self.assertEqual(1.0, result["post_route_elapsed_seconds"])
        self.assertEqual(106.0, result["total_elapsed_seconds"])
        self.assertEqual(
            "search_elapsed_seconds",
            result["timing_scope"]["paper_method_wall_time_field"],
        )


if __name__ == "__main__":
    unittest.main()
