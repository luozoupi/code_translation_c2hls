#!/usr/bin/env python3
"""Flash flow must assign distinct gmemN bundles per m_axi pointer port."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prompt_c2hls import Instruction_c2hls_flash, q_optimize_flash


class FlashGmemBundlePromptTest(unittest.TestCase):
    def test_flash_user_prompt_requires_distinct_bundles(self) -> None:
        self.assertIn("bundle=gmem0", q_optimize_flash)
        self.assertIn("bundle=gmem1", q_optimize_flash)
        self.assertIn("Never", q_optimize_flash)
        self.assertIn("bundle=gmem`", q_optimize_flash)

    def test_flash_system_instruction_requires_distinct_bundles(self) -> None:
        self.assertIn("bundle=gmem0", Instruction_c2hls_flash)
        self.assertIn("bundle=gmem1", Instruction_c2hls_flash)
        self.assertIn("MANDATORY", Instruction_c2hls_flash)


if __name__ == "__main__":
    unittest.main()
