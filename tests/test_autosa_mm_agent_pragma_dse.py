#!/usr/bin/env python3
"""Pragma-only DSE on frozen autosa_mm flash kernel: no algorithm rewrite."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.autosa_mm_agent_pragma_dse import (  # noqa: E402
    RANK1_CYCLES,
    apply_pragmas,
    enumerate_points,
    load_seed_kernel,
)

MAC = "local_C[i][j] += local_A[i][k] * local_B[j][k];"


class AutosaMmAgentPragmaDseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = load_seed_kernel()

    def test_seed_is_the_flash_selected_kernel(self) -> None:
        self.assertIn(MAC, self.seed)
        self.assertNotIn("ARRAY_PARTITION", self.seed)
        self.assertNotIn("#pragma HLS UNROLL", self.seed)

    def test_baseline_point_is_bit_identical_to_seed(self) -> None:
        baseline = next(p for p in enumerate_points() if p["id"] == "baseline")
        self.assertEqual(apply_pragmas(self.seed, baseline), self.seed)

    def test_points_keep_the_ijk_mac_and_loop_bounds(self) -> None:
        for point in enumerate_points():
            code = apply_pragmas(self.seed, point)
            self.assertIn(MAC, code, point["id"])
            self.assertIn("for (int i = 0; i < I; i++)", code)
            self.assertIn("for (int j = 0; j < J; j++)", code)
            self.assertIn("for (int k = 0; k < K; k++)", code)
            self.assertEqual(code.count("void autosa_mm("), 1, point["id"])

    def test_unroll_k_and_matching_partition_are_in_the_space(self) -> None:
        ids = {p["id"] for p in enumerate_points()}
        self.assertIn("uk4_partAB_cyc4", ids)
        point = next(p for p in enumerate_points() if p["id"] == "uk4_partAB_cyc4")
        code = apply_pragmas(self.seed, point)
        self.assertIn("#pragma HLS UNROLL factor=4", code)
        self.assertIn("#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=4 dim=2", code)
        self.assertIn("#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=4 dim=2", code)
        self.assertNotIn("void PE", code)
        self.assertNotIn("A_IO_L2", code)

    def test_complete_partition_is_in_the_space(self) -> None:
        ids = {p["id"] for p in enumerate_points()}
        self.assertIn("partAB_complete_d2", ids)
        self.assertIn("uk8_partAB_complete_d2", ids)
        self.assertIn("partAB_complete_all", ids)
        kinds = {
            spec[0]
            for point in enumerate_points()
            for spec in (point.get("part_a"), point.get("part_b"), point.get("part_c"))
            if spec
        }
        self.assertIn("complete", kinds)
        point = next(p for p in enumerate_points() if p["id"] == "uk8_partAB_complete_d2")
        code = apply_pragmas(self.seed, point)
        self.assertIn("#pragma HLS ARRAY_PARTITION variable=local_A complete dim=2", code)
        self.assertIn("#pragma HLS ARRAY_PARTITION variable=local_B complete dim=2", code)
        self.assertNotIn("complete factor=", code)
        all_dims = next(p for p in enumerate_points() if p["id"] == "partAB_complete_all")
        all_code = apply_pragmas(self.seed, all_dims)
        self.assertIn("#pragma HLS ARRAY_PARTITION variable=local_A complete dim=0", all_code)
        self.assertIn("#pragma HLS ARRAY_PARTITION variable=local_B complete dim=0", all_code)

    def test_rank1_reference_is_4228(self) -> None:
        self.assertEqual(RANK1_CYCLES, 4228)

    def test_space_is_coordinated_not_a_cartesian_blowup(self) -> None:
        points = enumerate_points()
        self.assertGreaterEqual(len(points), 10)
        self.assertLessEqual(len(points), 32)
        self.assertEqual(len({p["id"] for p in points}), len(points))


if __name__ == "__main__":
    unittest.main()
