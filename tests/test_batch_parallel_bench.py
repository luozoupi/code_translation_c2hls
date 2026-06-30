"""Tests for batch_parallel_bench orchestrator lifecycle."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_bench import BatchParallelBenchSession
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue


class BatchParallelBenchSessionTests(unittest.TestCase):
    def test_synth_success_enqueues_cosim(self) -> None:
        with patch.object(BatchParallelBenchSession, "__init__", lambda self, **kwargs: None):
            session = BatchParallelBenchSession(
                variant_key="aav_n",
                bench="hlsfactory_jacobi-1d",
                bench_dir=Path("/tmp/bench"),
                cell_dir=Path("/tmp/cell"),
                model_id="test-model",
                turns=2,
            )
        session.variant_key = "aav_n"
        session.bench = "hlsfactory_jacobi-1d"
        job = BatchParallelJob(
            id=1,
            variant="aav_n",
            bench="hlsfactory_jacobi-1d",
            kind="synth",
            phase="phase_b",
            attempt=0,
            stage="synth",
            meta={},
            assigned_role="synth",
            assigned_node=0,
            assigned_slot=0,
        )
        mock_orch = MagicMock()
        mock_orch.hls_code = "code"
        mock_orch.header_code = ""
        mock_orch.header_name = "kernel.h"
        mock_orch.translated_hls_top = "top"
        mock_orch.part = "part"
        mock_orch.clock_ns = 4.0
        mock_orch.extra_files = []
        mock_orch.testbench_code = "tb"
        mock_orch.supports_cosim = True
        mock_orch.cosim_depths = {}
        mock_orch.turns_limitation = 4
        mock_orch.turn_results = []
        mock_orch.synthesis.revert_threshold = 3
        mock_orch.synthesis._should_revert.return_value = False
        mock_orch.synthesis._record_best.return_value = {"code": "code"}
        mock_orch._pipelined_ctx = {}
        session.orchestrator = mock_orch

        with patch("c2hls.compile_check_cpp", return_value=(True, "")):
            with patch.object(
                session,
                "_synth_only",
                return_value={"synth": {"success": True, "report": {"LUT": 1}}},
            ):
                followups = session._run_synth(job)

        self.assertEqual(len(followups), 1)
        self.assertEqual(followups[0]["kind"], "cosim")
        self.assertEqual(followups[0]["phase"], "phase_b")

    def test_synth_ensures_orchestrator_before_run(self) -> None:
        with patch.object(BatchParallelBenchSession, "__init__", lambda self, **kwargs: None):
            session = BatchParallelBenchSession(
                variant_key="aav_n",
                bench="hlsfactory_jacobi-1d",
                bench_dir=Path("/tmp/bench"),
                cell_dir=Path("/tmp/cell"),
                model_id="test-model",
                turns=2,
            )
        session.variant_key = "aav_n"
        session.bench = "hlsfactory_jacobi-1d"
        session.orchestrator = None
        job = BatchParallelJob(
            id=1,
            variant="aav_n",
            bench="hlsfactory_jacobi-1d",
            kind="synth",
            phase="phase_b",
            attempt=0,
            stage="synth",
            meta={},
            assigned_role="synth",
            assigned_node=0,
            assigned_slot=0,
        )
        queue = MagicMock(spec=BatchParallelQueue)
        mock_orch = MagicMock()

        with patch.object(session, "_ensure_orchestrator", return_value=mock_orch) as ensure:
            with patch.object(session, "_run_synth", return_value=[{
                "kind": "cosim",
                "phase": "phase_b",
                "attempt": 0,
                "stage": "cosim",
            }]) as run_synth:
                with patch.object(session, "_save_state") as save_state:
                    with patch.object(session, "_apply_followups") as apply_followups:
                        session.handle_job(job, queue)

        ensure.assert_called_once()
        run_synth.assert_called_once_with(job)
        save_state.assert_called_once_with(mock_orch)
        apply_followups.assert_called_once()


if __name__ == "__main__":
    unittest.main()
