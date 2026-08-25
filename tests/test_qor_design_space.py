from __future__ import annotations

import hashlib
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import c2hls  # noqa: E402
import export_schema_jsonl  # noqa: E402
import qor_design_space as qds  # noqa: E402
import run_agentic_sweep  # noqa: E402


PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33


def _report(cycles: int, *, dsp: int = 8, ii: int = 1) -> dict:
    return {
        "latency_cycles": cycles,
        "latency_cycles_worst": cycles,
        "bram": 4,
        "dsp": dsp,
        "ff": 1000,
        "lut": 2000,
        "uram": 0,
        "slack_ns": 0.05,
        "requested_clock_period_ns": CLOCK_NS,
        "estimated_clock_period_ns": 3.20,
        "feedback": {
            "scopes": [
                {"scope_id": "loop_i", "pipeline_ii": ii, "trip_count": 64}
            ]
        },
    }


def _passing_csim() -> dict:
    return {
        "status": "passed",
        "ran": True,
        "success": True,
        "passed": True,
        "error": "",
    }


class QorDesignSpaceTests(unittest.TestCase):
    def test_schema_export_keeps_qor_evidence_but_drops_code(self):
        compact = export_schema_jsonl._qor_design_sweep({
            "quality_repair": {
                "design_sweep": {
                    "enabled": True,
                    "winner_candidate_id": "qor-ofat-1",
                    "candidates": [
                        {
                            "candidate_id": "qor-ofat-1",
                            "code": "void workload() {}",
                            "metrics": {"latency_cycles_worst": 90},
                        }
                    ],
                }
            }
        })
        self.assertTrue(compact["enabled"])
        self.assertEqual("qor-ofat-1", compact["winner_candidate_id"])
        self.assertNotIn("code", compact["candidates"][0])

    def test_discovers_diverse_typed_knobs_and_multiple_interface_options(self):
        code = """\
#define TILE_M 16
static const int BLOCK_N = 8;
const int ORDINARY = 7;
void workload(int *a) {
#pragma HLS PIPELINE II=2
#pragma HLS PIPELINE II=4
#pragma HLS PIPELINE
#pragma HLS UNROLL
#pragma HLS ARRAY_PARTITION variable=a factor=4 dim=1
#pragma HLS INTERFACE m_axi port=a max_widen_bitwidth=512 num_read_outstanding=16 max_read_burst_length=64
}
"""
        limited = qds.discover_qor_knobs(code, max_knobs=4)
        self.assertEqual(
            ["pipeline_ii", "unroll_factor", "partition_factor", "tile_size"],
            [knob.kind for knob in limited],
        )

        all_knobs = qds.discover_qor_knobs(code)
        names = {knob.name for knob in all_knobs}
        kinds = {knob.kind for knob in all_knobs}
        self.assertIn("TILE_M", names)
        self.assertIn("BLOCK_N", names)
        self.assertNotIn("ORDINARY", names)
        self.assertIn("interface_max_widen_bitwidth", kinds)
        self.assertIn("interface_num_read_outstanding", kinds)
        self.assertIn("interface_max_read_burst_length", kinds)
        self.assertTrue(
            any(
                knob.kind == "pipeline_ii" and knob.current_label == "auto"
                for knob in all_knobs
            )
        )

    def test_discovers_step_native_tile_axi_and_resource_toggles(self):
        code = """\
void workload(float *a, float *b) {
const int tk = 16; // Tile size for the K dimension
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS BIND_OP variable=x op=mul impl=dsp
#pragma HLS BIND_STORAGE variable=b type=ram_2p impl=bram
#pragma HLS RESOURCE variable=a core=RAM_2P_BRAM
}
"""
        knobs = qds.discover_qor_knobs(code)
        kinds = {knob.kind for knob in knobs}
        self.assertTrue({
            "tile_size",
            "interface_max_widen_bitwidth",
            "dataflow_enabled",
            "partition_enabled",
            "bind_op_enabled",
            "bind_storage_enabled",
            "resource_enabled",
        }.issubset(kinds))

        tile = next(knob for knob in knobs if knob.kind == "tile_size")
        widened = next(
            knob for knob in knobs
            if knob.kind == "interface_max_widen_bitwidth"
        )
        dataflow = next(knob for knob in knobs if knob.kind == "dataflow_enabled")
        candidate = qds.apply_knob_values(
            code,
            [(tile, 8), (widened, 512), (dataflow, 0)],
        )
        self.assertIn("const int tk = 8", candidate)
        self.assertIn("max_widen_bitwidth=512", candidate)
        self.assertIn("#if 0\n#pragma HLS DATAFLOW\n#endif", candidate)

    def test_resource_latency_and_step_preference_are_typed(self):
        code = """\
void workload(float *a) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=4 dim=1
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem
#pragma HLS BIND_OP variable=x op=mul impl=dsp latency=2
#pragma HLS BIND_STORAGE variable=a type=ram_2p impl=bram latency=2
#pragma HLS RESOURCE variable=a core=RAM_2P_BRAM latency=2
}
"""
        all_knobs = qds.discover_qor_knobs(code)
        kinds = {knob.kind for knob in all_knobs}
        self.assertTrue({
            "bind_op_latency",
            "bind_storage_latency",
            "resource_latency",
        }.issubset(kinds))
        self.assertNotIn("bind_op_enabled", kinds)

        focused = qds.discover_qor_knobs(
            code,
            max_knobs=2,
            preferred_kinds=qds.STEP_PREFERRED_KINDS["coalescing"],
        )
        self.assertEqual(2, len(focused))
        self.assertEqual(
            ["interface_max_widen_bitwidth", "interface_max_widen_bitwidth"],
            [knob.kind for knob in focused],
        )

    def test_ofat_and_interaction_candidates_always_use_frozen_parent(self):
        parent = """\
#define TILE_M 8
void workload(int *a) {
#pragma HLS PIPELINE II=2
#pragma HLS ARRAY_PARTITION variable=a factor=2 dim=1
}
"""
        knobs = qds.discover_qor_knobs(
            parent,
            factor_values=(1, 2, 4),
            ii_values=(1, 2, 4),
            tile_values=(4, 8, 16),
        )
        candidates = qds.build_ofat_candidates(parent, knobs, max_candidates=6)
        self.assertEqual(6, len(candidates))
        self.assertTrue(all(len(item["changed_knobs"]) == 1 for item in candidates))
        self.assertTrue(all(item["code"] != parent for item in candidates))
        self.assertEqual(parent, "#define TILE_M 8\nvoid workload(int *a) {\n#pragma HLS PIPELINE II=2\n#pragma HLS ARRAY_PARTITION variable=a factor=2 dim=1\n}\n")

        preferred = {
            knobs[0].knob_id: knobs[0].candidate_values[0],
            knobs[1].knob_id: knobs[1].candidate_values[0],
        }
        interactions = qds.build_interaction_candidates(
            parent,
            knobs,
            preferred,
            max_candidates=1,
        )
        self.assertEqual(1, len(interactions))
        self.assertEqual(2, len(interactions[0]["changed_knobs"]))
        self.assertEqual(hashlib.sha256(interactions[0]["code"].encode()).hexdigest(),
                         interactions[0]["code_sha256"])

    def test_pareto_and_trends_include_parent_observation(self):
        code = "void workload() {\n#pragma HLS PIPELINE II=1\n}\n"
        knob = qds.discover_qor_knobs(code, ii_values=(1, 2, 4))[0]
        parent = {
            "candidate_id": "frozen_parent",
            "feasible": True,
            "metrics": qds.extract_qor_metrics(_report(100, dsp=8, ii=1)),
        }
        records = [
            {
                "candidate_id": "ii2",
                "feasible": True,
                "changed_knobs": [{"knob_id": knob.knob_id, "to": 2}],
                "metrics": qds.extract_qor_metrics(_report(120, dsp=7, ii=2)),
            },
            {
                "candidate_id": "ii4",
                "feasible": True,
                "changed_knobs": [{"knob_id": knob.knob_id, "to": 4}],
                "metrics": qds.extract_qor_metrics(_report(160, dsp=6, ii=4)),
            },
        ]
        trends = qds.summarize_knob_trends(records, [knob], parent=parent)
        self.assertEqual([1.0, 2.0, 4.0], trends[0]["tested_values"])
        self.assertEqual(0, trends[0]["monotonicity_violations"])
        self.assertAlmostEqual(1.0, trends[0]["spearman_value_vs_worst_cycles"])
        self.assertTrue(trends[0]["observations"][0]["is_parent"])

        pareto_records = [
            parent,
            records[0],
            {
                "candidate_id": "dominated",
                "feasible": True,
                "metrics": qds.extract_qor_metrics(_report(180, dsp=12)),
            },
        ]
        frontier = qds.pareto_candidate_ids(pareto_records)
        self.assertNotIn("dominated", frontier)
        self.assertIn("frozen_parent", frontier)


class _FakeOrchestrator:
    def __init__(self):
        self.gpt_model = "unused"
        self.hls_code = "#define TILE_M 8\nvoid workload() {}\n"
        self.synth_report = _report(100)
        self.generated_csim = _passing_csim()
        self.generated_cosim = None
        self.testbench_code = "int main() { return 0; }"
        self.part = PART
        self.clock_ns = CLOCK_NS
        self.vitis_version = "2023.2"
        self.seen_parent_codes: list[str] = []

    @staticmethod
    def _candidate_report_sha256(report):
        return c2hls.C2HLSOrchestrator._candidate_report_sha256(report)

    @staticmethod
    def _best_so_far_score(report):
        return c2hls.C2HLSOrchestrator._best_so_far_score(report)

    def _evaluate_qor_candidate(self, code, label, metadata):
        self.seen_parent_codes.append(self.hls_code)
        value = int(re.search(r"TILE_M\s+(\d+)", code).group(1))
        cycles = {4: 120, 16: 70}[value]
        report = _report(cycles, dsp=8 + value // 4)
        return {
            "success": True,
            "code": code,
            "report": report,
            "csim": _passing_csim(),
            "cosim": None,
            "feasibility": {
                "feasible": True,
                "correctness_ok": True,
                "resource_fit": True,
                "timing_met": True,
                "reasons": [],
            },
            "error": "",
            "event": {
                "qor_evaluation_index": len(self.seen_parent_codes) - 1,
                "failure_class": None,
            },
        }


class _ResourceTieOrchestrator(_FakeOrchestrator):
    def _evaluate_qor_candidate(self, code, label, metadata):
        self.seen_parent_codes.append(self.hls_code)
        value = int(re.search(r"TILE_M\s+(\d+)", code).group(1))
        report = _report(100, dsp={4: 4, 16: 12}[value])
        return {
            "success": True,
            "code": code,
            "report": report,
            "csim": _passing_csim(),
            "cosim": None,
            "feasibility": {
                "feasible": True,
                "correctness_ok": True,
                "resource_fit": True,
                "timing_met": True,
                "reasons": [],
            },
            "error": "",
            "event": {
                "qor_evaluation_index": len(self.seen_parent_codes) - 1,
                "failure_class": None,
            },
        }


class QualityRepairIntegrationTests(unittest.TestCase):
    def test_design_sweep_with_no_explicit_knobs_retains_parent(self):
        orch = _FakeOrchestrator()
        orch.hls_code = "void workload() {}\n"
        agent = c2hls.QualityRepairAgent(orch)
        with patch.dict(
            os.environ,
            {c2hls.QOR_DESIGN_SWEEP_ENV: "1"},
            clear=False,
        ):
            summary = agent.run_design_sweep()
        self.assertTrue(summary["attempted"])
        self.assertFalse(summary["applied"])
        self.assertEqual(0, summary["candidate_count"])
        self.assertEqual("frozen_parent", summary["winner_candidate_id"])
        self.assertTrue(summary["parent"]["pareto_frontier"])

    def test_design_sweep_refuses_parent_without_passing_csim(self):
        orch = _FakeOrchestrator()
        orch.generated_csim = None
        agent = c2hls.QualityRepairAgent(orch)
        with patch.dict(
            os.environ,
            {c2hls.QOR_DESIGN_SWEEP_ENV: "1"},
            clear=False,
        ):
            summary = agent.run_design_sweep()
        self.assertFalse(summary["attempted"])
        self.assertIn("passing CSim", summary["reason"])
        self.assertEqual([], orch.seen_parent_codes)

    def test_design_sweep_promotes_best_feasible_candidate_after_frozen_evaluation(self):
        orch = _FakeOrchestrator()
        orch.qor_parent_origin = {
            "step_name": "tiling",
            "step_index": 0,
            "source": "step",
        }
        parent_code = orch.hls_code
        agent = c2hls.QualityRepairAgent(orch)
        environment = {
            c2hls.QOR_DESIGN_SWEEP_ENV: "1",
            c2hls.QOR_SWEEP_MAX_KNOBS_ENV: "1",
            c2hls.QOR_SWEEP_MAX_CANDIDATES_ENV: "2",
            c2hls.QOR_SWEEP_TILE_VALUES_ENV: "4,8,16",
            c2hls.QOR_SWEEP_VALUES_ENV: "1,2,4",
            c2hls.QOR_SWEEP_II_VALUES_ENV: "1,2,4",
            c2hls.QOR_SWEEP_INTERACTIONS_ENV: "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            summary = agent.run_design_sweep()

        self.assertTrue(summary["attempted"])
        self.assertTrue(summary["applied"])
        self.assertEqual(2, summary["candidate_count"])
        self.assertEqual(
            list(qds.STEP_PREFERRED_KINDS["tiling"]),
            summary["configuration"]["preferred_knob_kinds"],
        )
        self.assertEqual("tiling", summary["parent_origin"]["step_name"])
        self.assertEqual(70.0, summary["winner_metrics"]["latency_cycles_worst"])
        self.assertIn("TILE_M 16", orch.hls_code)
        self.assertTrue(all(code == parent_code for code in orch.seen_parent_codes))
        self.assertIsNone(orch.generated_cosim)
        winner = next(
            item
            for item in summary["candidates"]
            if item["candidate_id"] == summary["winner_candidate_id"]
        )
        changed_id = winner["changed_knobs"][0]["knob_id"]
        self.assertNotIn(changed_id, winner["fixed_knob_values"])
        self.assertEqual(16, winner["effective_knob_values"][changed_id])

    def test_design_sweep_uses_resources_to_break_exact_latency_tie(self):
        orch = _ResourceTieOrchestrator()
        agent = c2hls.QualityRepairAgent(orch)
        environment = {
            c2hls.QOR_DESIGN_SWEEP_ENV: "1",
            c2hls.QOR_SWEEP_MAX_KNOBS_ENV: "1",
            c2hls.QOR_SWEEP_MAX_CANDIDATES_ENV: "2",
            c2hls.QOR_SWEEP_TILE_VALUES_ENV: "4,8,16",
            c2hls.QOR_SWEEP_INTERACTIONS_ENV: "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            summary = agent.run_design_sweep()

        self.assertTrue(summary["applied"])
        self.assertEqual(100.0, summary["winner_metrics"]["latency_cycles_worst"])
        self.assertEqual(4.0, summary["winner_metrics"]["dsp"])
        self.assertIn("TILE_M 4", orch.hls_code)
        self.assertIn("aggregate-resource tie-break", summary["winner_explanation"])

    def test_qor_candidate_forces_csim_before_csynth_and_never_cosim(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.qor_synthesis_eval_count = 0
        orch.qor_synthesis_eval_events = []
        orch.synthesis_eval_count = 0
        orch.synthesis_eval_events = []
        orch.synthesis_eval_budget = 3
        orch.llm_usage_events = []
        orch.llm_candidate_request_count = 0
        orch.llm_candidate_budget = 5
        orch.selected_winner_cosim_count = 0
        orch.post_route_implementation_count = 0
        orch.header_code = ""
        orch.header_name = "kernel.h"
        orch.extra_files = []
        orch.translated_hls_top = "workload"
        orch.part = PART
        orch.clock_ns = CLOCK_NS
        orch.testbench_code = "int main() { return 0; }"
        orch.cosim_depths = {}
        orch.independent_golden_output = ""
        orch.independent_golden_specs = {}
        outcome = {
            "synth": {
                "success": True,
                "ran": True,
                "report": _report(90),
                "error": "",
            },
            "csim": _passing_csim(),
            "cosim": None,
        }
        with (
            patch.object(c2hls, "compile_check_cpp", return_value=(True, "")),
            patch.object(
                c2hls,
                "_run_synth_csim_cosim",
                return_value=outcome,
            ) as runner,
        ):
            result = orch._evaluate_qor_candidate(
                "void workload() {}", "qor", {"candidate_id": "qor-1"}
            )

        self.assertTrue(result["success"])
        self.assertEqual(1, orch.qor_synthesis_eval_count)
        self.assertTrue(runner.call_args.kwargs["correctness_first_override"])
        self.assertFalse(runner.call_args.kwargs["run_cosim_check"])
        self.assertEqual("csynth_latency_cycles_worst",
                         result["event"]["latency_source"])
        self.assertFalse(result["event"]["selected_for_executed_cosim"])
        root = {
            "llm_usage": orch._llm_usage_summary(),
            "synthesis_evaluations": orch._synthesis_evaluation_summary(),
            **orch._tool_call_attribution(),
            "selected_code_sha256": None,
            "cosim_target_code_sha256": None,
        }
        contract = run_agentic_sweep._candidate_telemetry_contract(root)
        self.assertTrue(contract["complete"])
        self.assertTrue(contract["qor_attribution_complete"])
        self.assertEqual(1, contract["qor_synthesis_count"])
        self.assertEqual(1, root["synthesis_evaluation_count"])
        self.assertEqual(0, root["llm_candidate_synthesis_evaluation_count"])

    def test_qor_candidate_records_csim_timeout_without_counting_synthesis(self):
        orch = c2hls.C2HLSOrchestrator.__new__(c2hls.C2HLSOrchestrator)
        orch.qor_synthesis_eval_count = 0
        orch.qor_synthesis_eval_events = []
        orch.synthesis_eval_count = 0
        orch.synthesis_eval_events = []
        orch.synthesis_eval_budget = 1
        orch.llm_usage_events = []
        orch.llm_candidate_request_count = 0
        orch.header_code = ""
        orch.header_name = "kernel.h"
        orch.extra_files = []
        orch.translated_hls_top = "workload"
        orch.part = PART
        orch.clock_ns = CLOCK_NS
        orch.testbench_code = "int main() { return 0; }"
        orch.cosim_depths = {}
        orch.independent_golden_output = ""
        orch.independent_golden_specs = {}
        outcome = {
            "synth": {
                "success": False,
                "ran": False,
                "skipped": True,
                "status": "not_run",
                "report": {},
                "error": "Synthesis skipped after CSim timeout",
            },
            "csim": {
                "status": "timeout",
                "ran": True,
                "success": False,
                "passed": False,
                "error": "Csim timed out after 180s",
            },
            "cosim": None,
        }
        with (
            patch.object(c2hls, "compile_check_cpp", return_value=(True, "")),
            patch.object(c2hls, "_run_synth_csim_cosim", return_value=outcome),
        ):
            result = orch._evaluate_qor_candidate(
                "void workload() {}", "qor", {"candidate_id": "qor-timeout"}
            )

        self.assertFalse(result["success"])
        self.assertEqual("csim_timeout", result["event"]["failure_class"])
        self.assertEqual("timeout", result["event"]["status"])
        self.assertTrue(result["event"]["timed_out"])
        self.assertFalse(result["event"]["tool_failure"])
        self.assertEqual(0, orch.qor_synthesis_eval_count)

    def test_sweep_runner_adds_qor_budget_after_paper_profile(self):
        env = {
            "C2HLS_QOR_DESIGN_SWEEP": "1",
            "C2HLS_QOR_SWEEP_MAX_CANDIDATES": "8",
            "C2HLS_SYNTHESIS_EVAL_BUDGET": "5",
        }
        profile = {"name": "paper_reference_blind_v1"}
        with patch.dict(os.environ, env, clear=True):
            run_agentic_sweep._apply_post_profile_qor_budget(profile)
            self.assertEqual("13", os.environ["C2HLS_SYNTHESIS_EVAL_BUDGET"])
        self.assertEqual(
            13,
            profile["post_profile_overrides"]["qor_synthesis_budget"][
                "effective_total"
            ],
        )


if __name__ == "__main__":
    unittest.main()
