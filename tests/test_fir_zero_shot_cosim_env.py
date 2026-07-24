"""Fir 0-shot cosim env must override site defaults."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from c2hls_paths import configure_site  # noqa: E402
from zero_shot_lib import VARIANTS, configure_fir_zero_shot_env  # noqa: E402


class FirZeroShotCosimEnvTests(unittest.TestCase):
    def test_zero_shot_overrides_fir_run_cosim_default(self) -> None:
        configure_site("fir")
        self.assertEqual(os.environ.get("C2HLS_RUN_COSIM"), "0")
        configure_fir_zero_shot_env(VARIANTS["direct"])
        self.assertEqual(os.environ.get("C2HLS_RUN_COSIM"), "1")


if __name__ == "__main__":
    unittest.main()
