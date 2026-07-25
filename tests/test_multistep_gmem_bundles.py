#!/usr/bin/env python3
"""Multistep flow must require distinct gmemN bundles (same bar as flash)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prompt_c2hls import (  # noqa: E402
    Instruction_c2hls_multistep,
    q_optimize_coalescing,
    q_optimize_pipeline,
    q_optimize_tiling,
    q_optimize_unroll,
)


class MultistepGmemBundlePromptTest(unittest.TestCase):
    def test_multistep_system_instruction_requires_distinct_bundles(self) -> None:
        self.assertIn("bundle=gmem0", Instruction_c2hls_multistep)
        self.assertIn("bundle=gmem1", Instruction_c2hls_multistep)
        self.assertIn("MANDATORY", Instruction_c2hls_multistep)
        self.assertIn("Never", Instruction_c2hls_multistep)

    def test_opt_step_prompts_mention_distinct_bundles(self) -> None:
        for prompt in (
            q_optimize_tiling,
            q_optimize_pipeline,
            q_optimize_unroll,
            q_optimize_coalescing,
        ):
            self.assertIn("gmem0", prompt)


if __name__ == "__main__":
    unittest.main()
