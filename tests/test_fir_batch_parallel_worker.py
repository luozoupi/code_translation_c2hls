"""Tests for Fir batch_parallel worker helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from batch_parallel.config import FirBatchParallelConfig
from batch_parallel.worker import release_compute_after_bench


class FirBatchParallelWorkerTests(unittest.TestCase):
    def test_release_when_one_bench_per_node(self) -> None:
        cfg = FirBatchParallelConfig(
            compute_nodes_match_benches=True,
            workers_per_node=1,
        )
        self.assertTrue(release_compute_after_bench(cfg))

    def test_no_release_when_pooling_workers(self) -> None:
        cfg = FirBatchParallelConfig(
            compute_nodes_match_benches=False,
            workers_per_node=2,
        )
        self.assertFalse(release_compute_after_bench(cfg))


if __name__ == "__main__":
    unittest.main()
