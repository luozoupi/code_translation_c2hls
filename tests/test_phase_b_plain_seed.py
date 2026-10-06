#!/usr/bin/env python3
"""Phase B must seed from plain when skip_phase_a and FROM_GOLD is off."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from c2hls import C2HLSOrchestrator


class PhaseBPlainSeedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._skip = os.environ.get("C2HLS_SKIP_PHASE_B")
        os.environ.pop("C2HLS_SKIP_PHASE_B", None)

    def tearDown(self) -> None:
        if self._skip is None:
            os.environ.pop("C2HLS_SKIP_PHASE_B", None)
        else:
            os.environ["C2HLS_SKIP_PHASE_B"] = self._skip
    def test_skip_phase_a_seeds_from_plain_not_gold(self) -> None:
        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.skip_phase_a = True
        orch.c_code = "PLAIN_SEED_CODE"
        orch._gold_hls_baseline_code = "GOLD_SHOULD_NOT_BE_USED"
        orch.phaseb_mode = ""
        orch._append_history = MagicMock()
        orch.translator = MagicMock()

        os.environ["C2HLS_PHASEB_FROM_GOLD"] = "0"
        method = C2HLSOrchestrator.pipelined_phase_b_translate.__get__(orch, C2HLSOrchestrator)
        out = method()
        self.assertTrue(out.get("ok"))
        self.assertEqual(orch.hls_code, "PLAIN_SEED_CODE")
        orch.translator.translate_initial.assert_not_called()

    def test_from_gold_still_wins_when_enabled(self) -> None:
        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.skip_phase_a = True
        orch.c_code = "PLAIN_SEED_CODE"
        orch._gold_hls_baseline_code = "GOLD_SEED"
        orch.phaseb_mode = ""
        orch._append_history = MagicMock()

        os.environ["C2HLS_PHASEB_FROM_GOLD"] = "1"
        method = C2HLSOrchestrator.pipelined_phase_b_translate.__get__(orch, C2HLSOrchestrator)
        out = method()
        self.assertTrue(out.get("ok"))
        self.assertEqual(orch.hls_code, "GOLD_SEED")

    def test_skip_phase_b_seeds_plain_without_llm(self) -> None:
        orch = MagicMock(spec=C2HLSOrchestrator)
        orch.c_code = "PLAIN_SEED_CODE"
        orch._gold_hls_baseline_code = "GOLD_SHOULD_NOT_BE_USED"
        orch.phaseb_mode = ""
        orch._append_history = MagicMock()
        orch.translator = MagicMock()

        prev_skip = os.environ.get("C2HLS_SKIP_PHASE_B")
        os.environ["C2HLS_SKIP_PHASE_B"] = "1"
        os.environ["C2HLS_PHASEB_FROM_GOLD"] = "0"
        try:
            method = C2HLSOrchestrator.pipelined_phase_b_translate.__get__(orch, C2HLSOrchestrator)
            out = method()
        finally:
            if prev_skip is None:
                os.environ.pop("C2HLS_SKIP_PHASE_B", None)
            else:
                os.environ["C2HLS_SKIP_PHASE_B"] = prev_skip
        self.assertTrue(out.get("ok"))
        self.assertEqual(orch.hls_code, "PLAIN_SEED_CODE")
        orch.translator.translate_initial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
