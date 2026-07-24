"""Tests for explorer cosim status, winning-step selection, and run issues."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from metrics import (  # noqa: E402
    bench_cosim_metrics_from_multistep_doc,
    bench_cosim_latency_from_multistep_doc,
    bench_run_issues_from_multistep_doc,
    bench_speedup_from_multistep_doc,
    classify_cosim_status,
    find_winning_step_name,
    geomean_cosim_speedup_from_benches,
)


class TestCosimStatus(unittest.TestCase):
    def test_pass_requires_cycles(self) -> None:
        self.assertEqual(
            classify_cosim_status(
                {"ran": True, "passed": True, "kernel_runtime_cycles": 1234}
            ),
            "pass",
        )

    def test_fail_when_ran_not_passed(self) -> None:
        self.assertEqual(
            classify_cosim_status({"ran": True, "passed": False, "status": "failed"}),
            "fail",
        )

    def test_crash_on_sigsegv(self) -> None:
        self.assertEqual(
            classify_cosim_status(
                {
                    "ran": True,
                    "passed": False,
                    "error": "ERROR: System received a signal named SIGSEGV",
                }
            ),
            "crash",
        )

    def test_not_run_when_missing(self) -> None:
        self.assertEqual(classify_cosim_status(None), "not_run")


class TestWinningStepCosim(unittest.TestCase):
    def test_flash_cosim_only_when_flash_wins_and_passes(self) -> None:
        doc = {
            "success": True,
            "baseline_report": {"latency_cycles": 1000},
            "final_report": {"latency_cycles": 100},
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "report": {"latency_cycles": 1000},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 1000,
                    },
                }
            ],
            "steps": [
                {
                    "success": True,
                    "step_name": "flash",
                    "report": {"latency_cycles": 100},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 90,
                    },
                }
            ],
        }
        self.assertEqual(find_winning_step_name(doc), "flash")
        metrics = bench_cosim_metrics_from_multistep_doc(doc)
        assert metrics is not None
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["generated"], 90)
        self.assertEqual(bench_cosim_latency_from_multistep_doc(doc), 90)

    def test_baseline_cosim_when_flash_failed_and_phase_b_wins(self) -> None:
        doc = {
            "success": True,
            "baseline_report": {"latency_cycles": 1000},
            "final_report": {"latency_cycles": 1000},
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "report": {"latency_cycles": 1000},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 1000,
                    },
                }
            ],
            "steps": [
                {
                    "success": False,
                    "step_name": "flash",
                    "report": {"latency_cycles": 100},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 90,
                    },
                }
            ],
        }
        self.assertEqual(find_winning_step_name(doc), "baseline")
        metrics = bench_cosim_metrics_from_multistep_doc(doc)
        assert metrics is not None
        self.assertEqual(metrics["status"], "pass")
        self.assertEqual(metrics["generated"], 1000)
        self.assertEqual(metrics["winning_step"], "baseline")

    def test_fail_when_flash_wins_but_cosim_failed(self) -> None:
        doc = {
            "success": True,
            "baseline_report": {"latency_cycles": 1000},
            "final_report": {"latency_cycles": 100},
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "report": {"latency_cycles": 1000},
                    "cosim": {
                        "ran": True,
                        "passed": True,
                        "kernel_runtime_cycles": 1000,
                    },
                }
            ],
            "steps": [
                {
                    "success": True,
                    "step_name": "flash",
                    "report": {"latency_cycles": 100},
                    "cosim": {
                        "ran": True,
                        "passed": False,
                        "status": "failed",
                        "error": "COSIM 212-4 FAIL",
                    },
                }
            ],
        }
        metrics = bench_cosim_metrics_from_multistep_doc(doc)
        assert metrics is not None
        self.assertEqual(metrics["status"], "fail")
        self.assertIsNone(metrics["generated"])
        self.assertIsNone(metrics["speedup"])

    def test_geomean_excludes_fail_and_not_run(self) -> None:
        benches = {
            "a": {
                "status": "ok",
                "cosim": {"status": "pass", "speedup": 4.0},
            },
            "b": {
                "status": "ok",
                "cosim": {"status": "fail", "speedup": 100.0},
            },
            "c": {
                "status": "ok",
                "cosim": {"status": "not_run"},
            },
        }
        gm = geomean_cosim_speedup_from_benches(benches)
        self.assertEqual(gm["n"], 1)
        self.assertAlmostEqual(gm["geomean"], 4.0)

    def test_geomean_excludes_llm_and_csynth_timeout_issues(self) -> None:
        benches = {
            "ok": {
                "status": "ok",
                "cosim": {"status": "pass", "speedup": 4.0},
            },
            "llm": {
                "status": "ok",
                "run_issues": ["llm_connection_error"],
                "cosim": {"status": "pass", "speedup": 1.0, "winning_step": "baseline"},
            },
            "timeout": {
                "status": "ok",
                "run_issues": ["csynth_timeout"],
                "cosim": {"status": "pass", "speedup": 1.0, "winning_step": "baseline"},
            },
            "revert": {
                "status": "ok",
                "run_issues": ["flash_reverted"],
                "cosim": {"status": "pass", "speedup": 2.0, "winning_step": "baseline"},
            },
        }
        gm = geomean_cosim_speedup_from_benches(benches)
        self.assertEqual(gm["n"], 2)
        self.assertAlmostEqual(gm["geomean"], math.sqrt(8.0))


class TestRunIssues(unittest.TestCase):
    def test_llm_connection_error(self) -> None:
        doc = {
            "success": True,
            "steps": [
                {
                    "step_name": "flash",
                    "success": False,
                    "attempt_error": "Connection error.",
                    "exception_type": "APIConnectionError",
                }
            ],
        }
        self.assertEqual(bench_run_issues_from_multistep_doc(doc), ["llm_connection_error"])

    def test_llm_timeout(self) -> None:
        doc = {
            "success": True,
            "steps": [
                {
                    "step_name": "flash",
                    "success": False,
                    "attempt_error": "Request timed out.",
                    "exception_type": "APITimeoutError",
                    "report": {},
                }
            ],
            "baseline_report": {"latency_cycles": 1000},
            "final_report": {"latency_cycles": 1000},
        }
        self.assertEqual(bench_run_issues_from_multistep_doc(doc), ["llm_timeout"])
        self.assertIsNone(bench_speedup_from_multistep_doc(doc))
        metrics = bench_cosim_metrics_from_multistep_doc(
            {**doc, "baseline_cosim": {"ran": False, "skip_reason": "disabled"}},
        )
        assert metrics is not None
        self.assertEqual(metrics["status"], "fail")

    def test_csynth_timeout(self) -> None:
        doc = {
            "success": True,
            "steps": [
                {
                    "step_name": "flash",
                    "success": False,
                    "attempt_results": [
                        {
                            "stage": "synthesis",
                            "success": False,
                            "error": "Synthesis timed out after 7200s",
                        }
                    ],
                }
            ],
        }
        self.assertEqual(bench_run_issues_from_multistep_doc(doc), ["csynth_timeout"])

    def test_flash_reverted(self) -> None:
        doc = {
            "success": True,
            "steps": [
                {
                    "step_name": "flash",
                    "success": False,
                    "reverted_to_prev": True,
                }
            ],
        }
        self.assertEqual(bench_run_issues_from_multistep_doc(doc), ["flash_reverted"])


if __name__ == "__main__":
    unittest.main()
