"""Tests for Fir batch_parallel dashboard progress chips."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from batch_parallel.dashboard_progress import (  # noqa: E402
    bench_hls_progress,
    cosim_enabled_for_campaign,
    hls_chips_from_artifacts,
    infer_hls_progress,
)


class FirDashboardProgressTests(unittest.TestCase):
    def test_cosim_enabled_for_zero_shot_campaign(self) -> None:
        campaign = {
            "stamp": "20260707_abs_zero_shot_direct",
            "config": {
                "artifact_prefix": "abs_zero_shot_cosim",
                "pilot": {"workflow": "zero_shot_direct"},
            },
        }
        self.assertTrue(cosim_enabled_for_campaign(campaign))

    def test_cosim_not_off_in_blank_when_enabled(self) -> None:
        out = infer_hls_progress("", cosim_enabled=True)
        self.assertEqual(out["cosim"], "—")

    def test_csim_pass_sets_cosim_next(self) -> None:
        log = "=== [step: flash] ===\n[Step: flash] csim passed\n"
        out = infer_hls_progress(log, cosim_enabled=True)
        self.assertEqual(out["csim"], "pass")
        self.assertEqual(out["cosim"], "next")
        self.assertEqual(out["phase"], "cosim")

    def test_running_cosim_detected(self) -> None:
        log = "[Step: flash] Running co-simulation (cosim)...\n"
        out = infer_hls_progress(log, cosim_enabled=True)
        self.assertEqual(out["cosim"], "running")
        self.assertEqual(out["phase"], "cosim")

    def test_done_reads_cosim_pass_from_multistep(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "hlsfactory_foo" / "devstral2__flash__test"
            cell.mkdir(parents=True)
            (cell / "hlsfactory_foo_multistep_results.json").write_text(
                json.dumps({
                    "success": True,
                    "steps": [{
                        "step_name": "flash",
                        "success": True,
                        "report": {"latency_cycles": 100},
                        "cosim": {
                            "passed": True,
                            "ran": True,
                            "success": True,
                            "kernel_runtime_cycles": 42_000,
                        },
                    }],
                    "csim": {"passed": True, "ran": True},
                    "cosim": {
                        "passed": True,
                        "ran": True,
                        "success": True,
                        "kernel_runtime_cycles": 42_000,
                    },
                    "final_report": {"latency_cycles": 100},
                }),
                encoding="utf-8",
            )
            (cell / "hlsfactory_foo_flow_manifest.json").write_text(
                json.dumps({"flash_step_success": True}),
                encoding="utf-8",
            )
            out = bench_hls_progress(
                root,
                bench="hlsfactory_foo",
                status="done",
                model_id="devstral",
                setup_tag="flash__test",
                node_index=None,
                slurm_job_id=None,
                cosim_enabled=True,
                llm_in_flight=None,
            )
            self.assertEqual(out["csynth"], "pass")
            self.assertEqual(out["csim"], "pass")
            self.assertEqual(out["cosim"], "pass")

    def test_done_ignores_top_level_success_when_flash_failed(self) -> None:
        csynth, csim = hls_chips_from_artifacts({
            "success": True,
            "phase": "flash",
            "steps": [{
                "step_name": "flash",
                "success": False,
                "attempt_results": [{"stage": "compile_check", "success": False}],
            }],
        })
        self.assertEqual(csynth, "fail")
        self.assertEqual(csim, "—")

    def test_manifest_flash_step_success_false(self) -> None:
        csynth, csim = hls_chips_from_artifacts(
            {
                "phase": "flash",
                "steps": [{"step_name": "flash", "success": True}],
            },
            manifest_doc={"flash_step_success": False},
        )
        self.assertEqual(csynth, "fail")
        self.assertEqual(csim, "—")

    def test_phase_b_failure_before_flash(self) -> None:
        csynth, csim = hls_chips_from_artifacts({
            "phase": "B",
            "success": False,
            "error": "Baseline HLS synthesis/correctness failed",
            "steps": [],
        })
        self.assertEqual(csynth, "fail")
        self.assertEqual(csim, "—")

    def test_failed_queue_status_still_reads_artifact_chips(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "hlsfactory_bar" / "devstral2__flash__test"
            cell.mkdir(parents=True)
            (cell / "hlsfactory_bar_multistep_results.json").write_text(
                json.dumps({
                    "phase": "reference",
                    "success": False,
                    "error": "Gold HLS synthesis failed",
                }),
                encoding="utf-8",
            )
            out = bench_hls_progress(
                root,
                bench="hlsfactory_bar",
                status="failed",
                model_id="devstral",
                setup_tag="flash__test",
                node_index=None,
                slurm_job_id=None,
                cosim_enabled=True,
                llm_in_flight=None,
            )
            self.assertEqual(out["phase"], "fail")
            self.assertEqual(out["csynth"], "—")
            self.assertEqual(out["csim"], "—")


if __name__ == "__main__":
    unittest.main()
