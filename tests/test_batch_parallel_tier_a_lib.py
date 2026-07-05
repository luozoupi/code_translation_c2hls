"""Tests for tier_A batch_parallel helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_tier_a_lib import SETUP_TAG, TIER_A_VARIANT, resolve_tier_a_bench_map


class TierALibTests(unittest.TestCase):
    def test_resolve_dct(self) -> None:
        mapping = resolve_tier_a_bench_map(["spector_hls_dct"])
        self.assertIn("spector_hls_dct", mapping)
        self.assertTrue((mapping["spector_hls_dct"] / "plain.cpp").is_file())

    def test_constants(self) -> None:
        self.assertEqual(TIER_A_VARIANT, "tier_a_90")
        self.assertIn("tier_a", SETUP_TAG)


if __name__ == "__main__":
    unittest.main()
