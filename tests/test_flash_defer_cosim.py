"""Tests for C2HLS_FLASH_DEFER_COSIM support in batch_parallel_bench."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_bench import BatchParallelBenchSession, flash_defer_cosim_enabled
from batch_parallel_queue import BatchParallelJob


class FlashDeferCosimEnvTests(unittest.TestCase):
  def test_flash_defer_cosim_enabled_truthy_values(self) -> None:
    for value in ("1", "true", "True", "yes", "on", " ON "):
      with patch.dict(os.environ, {"C2HLS_FLASH_DEFER_COSIM": value}):
        self.assertTrue(flash_defer_cosim_enabled(), msg=value)

  def test_flash_defer_cosim_enabled_falsy_values(self) -> None:
    for value in ("", "0", "false", "no", "off"):
      with patch.dict(os.environ, {"C2HLS_FLASH_DEFER_COSIM": value}):
        self.assertFalse(flash_defer_cosim_enabled(), msg=value)

  def test_flash_defer_cosim_enabled_unset(self) -> None:
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("C2HLS_FLASH_DEFER_COSIM", None)
      self.assertFalse(flash_defer_cosim_enabled())


class FlashDeferCosimPhaseBTests(unittest.TestCase):
  def _session(self) -> BatchParallelBenchSession:
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
    return session

  def _job(self) -> BatchParallelJob:
    return BatchParallelJob(
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

  def _mock_orch(self) -> MagicMock:
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
    return mock_orch

  def test_defer_enabled_phase_b_success_advances_to_flash_codegen(self) -> None:
    session = self._session()
    job = self._job()
    mock_orch = self._mock_orch()
    session.orchestrator = mock_orch

    synth_outcome = {
      "synth": {"success": True, "report": {"LUT": 1}},
      "csim": {"ran": True, "passed": True},
      "cosim": None,
    }
    with patch.dict(os.environ, {"C2HLS_FLASH_DEFER_COSIM": "1"}):
      with patch("c2hls.compile_check_cpp", return_value=(True, "")):
        with patch.object(session, "_synth_only", return_value=synth_outcome):
          with patch("c2hls.record_flow_enabled", return_value=False):
            followups = session._run_synth(job)

    self.assertEqual(len(followups), 1)
    kinds = [spec["kind"] for spec in followups]
    self.assertNotIn("cosim", kinds)
    self.assertEqual(followups[0]["kind"], "codegen")
    self.assertEqual(followups[0]["phase"], "flash")
    self.assertEqual(followups[0]["stage"], "optimize")

  def test_defer_disabled_phase_b_success_enqueues_cosim(self) -> None:
    session = self._session()
    job = self._job()
    mock_orch = self._mock_orch()
    session.orchestrator = mock_orch

    synth_outcome = {"synth": {"success": True, "report": {"LUT": 1}}}
    with patch.dict(os.environ, {"C2HLS_FLASH_DEFER_COSIM": "0"}):
      with patch("c2hls.compile_check_cpp", return_value=(True, "")):
        with patch.object(session, "_synth_only", return_value=synth_outcome):
          followups = session._run_synth(job)

    self.assertEqual(len(followups), 1)
    self.assertEqual(followups[0]["kind"], "cosim")
    self.assertEqual(followups[0]["phase"], "phase_b")

  def test_defer_enabled_phase_b_csim_failure_triggers_repair(self) -> None:
    session = self._session()
    job = self._job()
    mock_orch = self._mock_orch()
    session.orchestrator = mock_orch

    synth_outcome = {
      "synth": {"success": True, "report": {"LUT": 1}},
      "csim": {"ran": True, "passed": False, "error": "assertion failed"},
      "cosim": None,
    }
    with patch.dict(os.environ, {"C2HLS_FLASH_DEFER_COSIM": "1"}):
      with patch("c2hls.compile_check_cpp", return_value=(True, "")):
        with patch.object(session, "_synth_only", return_value=synth_outcome):
          followups = session._run_synth(job)

    self.assertEqual(len(followups), 1)
    self.assertEqual(followups[0]["kind"], "codegen")
    self.assertEqual(followups[0]["phase"], "phase_b")
    self.assertEqual(followups[0]["stage"], "repair")
    self.assertEqual(followups[0]["meta"]["repair"]["kind"], "csim")


if __name__ == "__main__":
  unittest.main()
