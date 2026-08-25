from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import c2hls  # noqa: E402
import export_schema_jsonl as schema_export  # noqa: E402


class CosimGoldPrecheckTests(unittest.TestCase):
    def _enabled_env(self) -> dict[str, str]:
        return {
            c2hls.COSIM_SKIP_SLOWER_THAN_GOLD_ENV: "1",
            c2hls.COSIM_SKIP_GOLD_RATIO_ENV: "10.0",
        }

    def test_precheck_classifies_long_candidate_without_running_cosim(self):
        synth_result = {
            "success": True,
            "report": {"latency_cycles": 25_296_077},
        }
        csim_result = {"success": True, "passed": True, "error": ""}

        with (
            patch.dict(os.environ, self._enabled_env(), clear=False),
            patch.object(c2hls, "run_hls_synthesis", return_value=synth_result),
            patch.object(c2hls, "run_csim", return_value=csim_result),
            patch.object(c2hls, "run_cosim") as run_cosim_mock,
        ):
            outcome = c2hls._run_synth_csim_cosim(
                "void workload() {}",
                header_code="",
                header_name="kernel.h",
                top_function="workload",
                part="xcu280-fsvh2892-2L-e",
                clock_ns=3.33,
                extra_files=[],
                testbench_code="int main() { return 0; }",
                run_csim_check=True,
                run_cosim_check=True,
                cosim_reference_cycle_info={
                    "cycles": 1_893_899,
                    "source": "reference_validation.cosim.kernel_runtime_cycles",
                    "metric": "rtl_runtime_cycles",
                },
            )

        run_cosim_mock.assert_not_called()
        self.assertTrue(outcome["csim"]["passed"])
        self.assertEqual(outcome["cosim"]["status"], "timeout")
        self.assertFalse(outcome["cosim"]["ran"])
        self.assertEqual(outcome["cosim"]["skip_reason"], "predicted_longer_than_gold")
        self.assertEqual(
            outcome["cosim"]["cosim_policy"]["classification"],
            "predicted_timeout",
        )
        normalized = c2hls._normalize_saved_test_summary(
            outcome["cosim"], available=True, ran=False
        )
        self.assertEqual(normalized["status"], "timeout")
        self.assertFalse(normalized["ran"])
        self.assertIn("not run", normalized["error"])

    def test_precheck_does_not_skip_below_ratio(self):
        synth_result = {
            "success": True,
            "report": {"latency_cycles": 9_000},
        }
        passing_test = {"success": True, "passed": True, "error": ""}

        with (
            patch.dict(os.environ, self._enabled_env(), clear=False),
            patch.object(c2hls, "run_hls_synthesis", return_value=synth_result),
            patch.object(c2hls, "run_csim", return_value=passing_test),
            patch.object(c2hls, "run_cosim", return_value=passing_test) as run_cosim_mock,
        ):
            outcome = c2hls._run_synth_csim_cosim(
                "void workload() {}",
                header_code="",
                header_name="kernel.h",
                top_function="workload",
                part="xcu280-fsvh2892-2L-e",
                clock_ns=3.33,
                extra_files=[],
                testbench_code="int main() { return 0; }",
                run_csim_check=True,
                run_cosim_check=True,
                cosim_reference_cycle_info={"cycles": 1_000},
            )

        run_cosim_mock.assert_called_once()
        self.assertEqual(outcome["cosim"]["status"], "passed")
        self.assertTrue(outcome["cosim"]["ran"])

    def test_cosim_timeout_is_not_a_correctness_repair_signal(self):
        agent = c2hls.SynthesisAgent(SimpleNamespace(gpt_model="base-model"))
        outcome = {
            "csim": {"status": "passed", "ran": True, "passed": True},
            "cosim": {
                "status": "failed",
                "ran": True,
                "passed": False,
                "error": "Cosim timed out after 10800s",
            },
        }
        with patch.dict(os.environ, {"C2HLS_COSIM_REQUIRED": "1"}, clear=False):
            self.assertEqual(agent._correctness_gate_failure(outcome), ("", ""))

    def test_predictive_timeout_emits_schema_valid_rtl_record(self):
        policy = {
            "schema_version": "1.0",
            "policy": "gold_relative_csynth_precheck",
            "decision": "skip",
            "classification": "predicted_timeout",
            "ran": False,
            "reason": "generated_csynth_latency_exceeds_gold_ratio_threshold",
            "generated_csynth_latency_cycles": 25_296_077,
            "gold_reference_cycles": 1_893_899,
            "gold_reference_source": "reference_validation.cosim.kernel_runtime_cycles",
            "gold_reference_metric": "rtl_runtime_cycles",
            "ratio_generated_over_gold": 13.356614,
            "threshold_ratio": 10.0,
        }
        cosim = c2hls._predicted_cosim_timeout_summary(policy)
        records: list[dict] = []

        schema_export._emit_cosim_record(
            records,
            meta={"benchmark": "hlsfactory_2mm", "source_repo": "HLSFactory"},
            run_meta={"vitis_version": "2023.2", "clock_ns": 3.33},
            part="xcu280-fsvh2892-2L-e",
            model_id="qwen3.6-27b",
            variant_name="baseline",
            variant_index=0,
            origin_meta={"phase": "flash"},
            cosim=cosim,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(schema_export._validate_record(record), [])
        self.assertEqual(record["rtl_sim"]["status"], "timeout")
        self.assertIsNone(record["rtl_sim"]["kernel_runtime_cycles"])
        origin_meta = record["implementation"]["origin_meta"]
        self.assertFalse(origin_meta["cosim_ran"])
        self.assertEqual(origin_meta["cosim_skip_reason"], "predicted_longer_than_gold")
        self.assertEqual(
            origin_meta["cosim_policy"]["classification"],
            "predicted_timeout",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = Path(tmpdir) / "predictive_timeout.jsonl"
            jsonl.write_text(schema_export._strict_json_dumps(record) + "\n")
            validation = schema_export.validate_jsonl(jsonl)
        self.assertEqual(validation["invalid"], 0)
        self.assertEqual(validation["total"], 1)

    def test_multistep_export_keeps_baseline_cosim_separate_from_final(self):
        policy = {
            "policy": "gold_relative_csynth_precheck",
            "classification": "predicted_timeout",
            "decision": "skip",
            "ran": False,
            "threshold_ratio": 10.0,
            "generated_csynth_latency_cycles": 25_296_077,
            "gold_reference_cycles": 1_893_899,
            "ratio_generated_over_gold": 13.356614,
        }
        baseline_cosim = c2hls._predicted_cosim_timeout_summary(policy)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bench_dir = root / "hlsfactory_2mm"
            bench_dir.mkdir()
            (bench_dir / "metadata.json").write_text(json.dumps({
                "benchmark": "hlsfactory_2mm",
                "source_repo": "HLSFactory",
                "variants": [
                    {"name": "hlsfactory_2mm_0_baseline"},
                    {"name": "hlsfactory_2mm_1_flash"},
                ],
            }))
            result_path = root / "hlsfactory_2mm_multistep_results.json"
            result_path.write_text(json.dumps({
                "run": {
                    "model": "qwen3.6-27b",
                    "part": "xcu280-fsvh2892-2L-e",
                    "clock_ns": 3.33,
                },
                "baseline_report": {"latency_cycles": 25_296_077},
                "baseline_csim": {"status": "passed", "ran": True, "passed": True},
                "baseline_cosim": baseline_cosim,
                "final_report": {"latency_cycles": 2_037_918},
                "csim": {"status": "passed", "ran": True, "passed": True},
                "cosim": {
                    "status": "passed",
                    "ran": True,
                    "passed": True,
                    "kernel_runtime_cycles": 2_081_970,
                },
                "steps": [{
                    "step_name": "flash",
                    "success": True,
                    "report": {"latency_cycles": 2_037_918},
                    "csim": {"status": "passed", "ran": True, "passed": True},
                    "cosim": {
                        "status": "passed",
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 2_081_970,
                    },
                }],
            }))

            records = schema_export._records_from_multistep(
                bench_dir,
                result_path,
                "xcu280-fsvh2892-2L-e",
                3.33,
            )

        rtl_records = [r for r in records if r["report_type"] == "rtl_sim"]
        self.assertEqual(len(rtl_records), 2)
        by_step = {
            r["implementation"]["origin_meta"]["step"]: r
            for r in rtl_records
        }
        self.assertEqual(by_step["baseline"]["rtl_sim"]["status"], "timeout")
        self.assertFalse(by_step["baseline"]["implementation"]["origin_meta"]["cosim_ran"])
        self.assertEqual(by_step["flash"]["rtl_sim"]["status"], "pass")
        self.assertEqual(by_step["flash"]["rtl_sim"]["kernel_runtime_cycles"], 2_081_970)

    def test_reference_cycle_info_prefers_measured_cosim(self):
        info = c2hls._reference_cycle_info({
            "cosim": {
                "status": "passed",
                "passed": True,
                "kernel_runtime_cycles": 1_893_899,
            },
            "report": {"latency_cycles": 25_296_077},
        })
        self.assertEqual(info["cycles"], 1_893_899)
        self.assertEqual(info["metric"], "rtl_runtime_cycles")

    def test_explicit_skill_off_suppresses_dynamic_prompt_injection(self):
        with patch.dict(os.environ, {"C2HLS_FORCE_SKILL_PROMPTS": "0"}, clear=False):
            self.assertFalse(c2hls._skill_prompt_injection_enabled())

    def test_unset_skill_prompt_flag_preserves_legacy_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(c2hls._skill_prompt_injection_enabled())


if __name__ == "__main__":
    unittest.main()
