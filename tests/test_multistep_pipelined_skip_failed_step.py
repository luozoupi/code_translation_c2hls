"""Pipelined multistep must not abort the trajectory on a failed opt step.

Sequential run_multistep continues after a csim-failed step and promotes
best-so-far (phase_b or an earlier successful step). The pipelined runner
used to finalize the whole bench as FAIL, which is why jacobi-1d/2d and
gramschmidt (phase_b OK, tiling csim_failed) and bicg (tiling/pipeline/unroll
OK, doublebuffer csim_failed) were marked fail.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from flash_pipelined_queue import PipelinedJob
from multistep_pipelined_bench import MultistepPipelinedBenchSession


class SkipFailedOptStepTests(unittest.TestCase):
    def _session(self) -> MultistepPipelinedBenchSession:
        with patch.object(MultistepPipelinedBenchSession, "__init__", lambda self, **kwargs: None):
            session = MultistepPipelinedBenchSession(
                variant_key="aav_n",
                bench="hlsfactory_jacobi-1d",
                bench_dir=Path("/tmp/bench"),
                cell_dir=Path("/tmp/cell"),
                model_id="test-model",
                turns=20,
            )
        session.variant_key = "aav_n"
        session.bench = "hlsfactory_jacobi-1d"
        session.opt_steps = ["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]
        return session

    def _job(self, phase: str, attempt: int = 19) -> PipelinedJob:
        return PipelinedJob(
            id=1,
            variant="aav_n",
            bench="hlsfactory_jacobi-1d",
            kind="synth",
            phase=phase,
            attempt=attempt,
            stage="synth",
            meta={},
        )

    def _orch(self, *, phase: str, step_results: list | None = None):
        failed = {
            "success": False,
            "step_name": phase,
            "error": "csim_failed",
        }
        ctx = {
            f"{phase}_step_result": failed,
            "step_results": list(step_results or []),
        }
        orch = SimpleNamespace(
            _pipelined_ctx=ctx,
            _flow_phase_b_report={"latency_cycles": 41121},
            synth_report={"latency_cycles": 41121},
        )

        def synth_once(step_name: str, attempt: int) -> dict:
            return {"status": "step_done", "success": False, "error": "csim_failed"}

        orch.pipelined_multistep_step_synth_once = synth_once
        orch._pipelined_step_done_status = lambda step_name: "step_done"
        orch._pipelined_step_result_key = lambda step_name: f"{step_name}_step_result"
        return orch

    def test_tiling_csim_fail_continues_to_pipeline(self) -> None:
        session = self._session()
        session.orchestrator = self._orch(phase="tiling")
        followups = session._run_synth(self._job("tiling"))
        self.assertEqual(len(followups), 1)
        self.assertNotEqual(followups[0].get("phase"), "failed")
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "pipeline")
        recorded = session.orchestrator._pipelined_ctx["step_results"]
        self.assertEqual(len(recorded), 1)
        self.assertFalse(recorded[0]["success"])
        self.assertEqual(recorded[0]["step_name"], "tiling")

    def test_doublebuffer_csim_fail_keeps_prior_successes_and_continues(self) -> None:
        session = self._session()
        session.bench = "hlsfactory_bicg"
        prior = [
            {"success": True, "step_name": "tiling", "report": {"latency_cycles": 100976}},
            {"success": True, "step_name": "pipeline", "report": {"latency_cycles": 100976}},
            {"success": True, "step_name": "unroll", "report": {"latency_cycles": 100976}},
        ]
        session.orchestrator = self._orch(phase="doublebuffer", step_results=prior)
        followups = session._run_synth(self._job("doublebuffer"))
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "coalescing")
        names = [s["step_name"] for s in session.orchestrator._pipelined_ctx["step_results"]]
        self.assertEqual(names, ["tiling", "pipeline", "unroll", "doublebuffer"])

    def test_last_step_fail_finalizes_success_when_phase_b_exists(self) -> None:
        session = self._session()
        session.orchestrator = self._orch(phase="coalescing")
        followups = session._run_synth(self._job("coalescing"))
        self.assertEqual(followups[0]["kind"], "finalize")
        self.assertEqual(followups[0]["phase"], "finalize")

    def test_last_step_fail_without_phase_b_still_fails(self) -> None:
        session = self._session()
        orch = self._orch(phase="coalescing")
        orch._flow_phase_b_report = None
        orch.synth_report = None
        session.orchestrator = orch
        followups = session._run_synth(self._job("coalescing"))
        self.assertEqual(followups[0]["phase"], "failed")

    def test_missing_opt_step_kernel_continues_instead_of_failing(self) -> None:
        session = self._session()
        orch = self._orch(phase="doublebuffer")
        orch.pipelined_multistep_step_codegen = lambda step, repair=None: {
            "ok": False,
            "error": "no code in doublebuffer repair response",
        }
        session.orchestrator = orch
        job = PipelinedJob(
            id=2,
            variant="aav_n",
            bench="hlsfactory_jacobi-1d",
            kind="codegen",
            phase="doublebuffer",
            attempt=0,
            stage="repair",
            meta={"repair": {"kind": "compile", "error": "missing include"}},
        )
        followups = session._run_codegen(job)
        self.assertEqual(len(followups), 1)
        self.assertNotEqual(followups[0].get("phase"), "failed")
        self.assertEqual(followups[0]["kind"], "codegen")
        self.assertEqual(followups[0]["phase"], "coalescing")
        recorded = orch._pipelined_ctx["step_results"]
        self.assertEqual(recorded[-1]["step_name"], "doublebuffer")
        self.assertFalse(recorded[-1]["success"])

    def test_phase_b_missing_code_still_fails(self) -> None:
        session = self._session()
        orch = SimpleNamespace(_pipelined_ctx={})
        orch.pipelined_phase_b_translate = lambda: {"ok": False, "error": "no code in translate response"}
        session.orchestrator = orch
        job = PipelinedJob(
            id=3,
            variant="aav_n",
            bench="hlsfactory_jacobi-1d",
            kind="codegen",
            phase="phase_b",
            attempt=0,
            stage="translate",
            meta={},
        )
        followups = session._run_codegen(job)
        self.assertEqual(followups[0]["kind"], "finalize")
        self.assertEqual(followups[0]["phase"], "failed")


if __name__ == "__main__":
    unittest.main()
