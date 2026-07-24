"""Tests for TierABatchParallelBenchSession followups."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_queue import BatchParallelJob
from tier_a_batch_parallel_bench import TierABatchParallelBenchSession


class TierABenchSessionTests(unittest.TestCase):
    def _session(self) -> TierABatchParallelBenchSession:
        with patch.object(TierABatchParallelBenchSession, "__init__", lambda self, **kwargs: None):
            session = TierABatchParallelBenchSession(
                variant_key="tier_a_90",
                bench="spector_hls_dct",
                bench_dir=Path("/tmp/bench"),
                cell_dir=Path("/tmp/cell"),
                model_id="m",
                turns=4,
            )
        session.variant_key = "tier_a_90"
        session.bench = "spector_hls_dct"
        session.cell_dir = Path("/tmp/cell")
        session.inputs = {"meta": {}}
        return session

    def test_reference_pass_enqueues_phase_b_codegen(self) -> None:
        session = self._session()
        job = BatchParallelJob(
            id=1,
            variant="tier_a_90",
            bench="spector_hls_dct",
            kind="synth",
            phase="reference",
            attempt=0,
            stage="gold_gate",
            meta={},
        )
        ref_ok = {"benchmark_ready": True, "invalid_reason": ""}
        with patch.object(TierABatchParallelBenchSession, "_validate_gold_reference", return_value=ref_ok):
            with patch.object(session, "_save_reference_validation"):
                followups = session._run_reference_synth(job)
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "phase_b")
        self.assertEqual(followups[0]["stage"], "translate")

    def test_phase_b_synth_success_no_cosim_followup(self) -> None:
        session = self._session()
        job = BatchParallelJob(
            id=2,
            variant="tier_a_90",
            bench="spector_hls_dct",
            kind="synth",
            phase="phase_b",
            attempt=0,
            stage="synth",
            meta={},
        )
        mock_orch = MagicMock()
        mock_orch.hls_code = "code"
        mock_orch.header_code = ""
        mock_orch.header_name = "kernel.h"
        mock_orch.translated_hls_top = "top"
        mock_orch.part = "xcu280-fsvh2892-2L-e"
        mock_orch.clock_ns = 3.33
        mock_orch.extra_files = []
        mock_orch.testbench_code = "tb"
        mock_orch.turns_limitation = 4
        mock_orch.turn_results = []
        mock_orch.synthesis.revert_threshold = 3
        mock_orch.synthesis._should_revert.return_value = False
        mock_orch.synthesis._record_best.return_value = {}
        mock_orch._pipelined_ctx = {}
        session.orchestrator = mock_orch
        session.reference_validation = {"benchmark_ready": True}
        outcome = {
            "synth": {"success": True, "report": {"latency_ns": 1}},
            "csim": {"ran": True, "passed": True},
            "cosim": None,
        }
        with patch.object(session, "_ensure_orchestrator", return_value=mock_orch):
            with patch.object(session, "_compile_check_cpp", return_value=(True, "")):
                with patch.object(session, "_synth_csim_only", return_value=outcome):
                    followups = session._run_synth_phase_b(job)
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "flash")
        kinds = [spec["kind"] for spec in followups]
        self.assertNotIn("cosim", kinds)

    def test_finalize_failure_promotes_ground_truth_report(self) -> None:
        """Failure path must keep gold bookkeeping in sync with reference_validation."""
        import tempfile

        session = self._session()
        with tempfile.TemporaryDirectory() as tmp:
            session.cell_dir = Path(tmp)
            session.orchestrator = None
            session.reference_validation = {
                "benchmark_ready": True,
                "invalid_reason": "",
                "synthesis": {"status": "passed"},
                "report": {
                    "latency_cycles": 1234,
                    "bram": 10,
                    "dsp": 2,
                    "ff": 100,
                    "lut": 200,
                    "uram": 0,
                },
            }

            def _fake_sanitize(results, reference_validation):
                out = dict(results)
                out["reference_validation"] = reference_validation
                out["ground_truth_report"] = dict(reference_validation.get("report") or {})
                out["ground_truth_status"] = (
                    "valid" if reference_validation.get("benchmark_ready") else "invalid"
                )
                out["baseline_status"] = (
                    reference_validation.get("synthesis", {}) or {}
                ).get("status", "failed")
                return out

            with patch(
                "tier_a_batch_parallel_bench._sanitize_saved_result_record",
                side_effect=_fake_sanitize,
            ):
                session._finalize_failure("opt failed")

            result_path = session.cell_dir / f"{session.bench}_multistep_results.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["success"])
            self.assertEqual(payload["ground_truth_report"]["latency_cycles"], 1234)
            self.assertEqual(payload["ground_truth_report"]["lut"], 200)
            self.assertEqual(payload["ground_truth_status"], "valid")
            self.assertEqual(payload["baseline_status"], "passed")
            self.assertEqual(
                payload["ground_truth_report"],
                payload["reference_validation"]["report"],
            )


if __name__ == "__main__":
    unittest.main()
