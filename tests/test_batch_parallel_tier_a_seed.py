"""Tests for tier_A reference seeding in batch_parallel queue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_queue import BatchParallelQueue


class TierASeedTests(unittest.TestCase):
    def test_seed_reference_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            queue = BatchParallelQueue(Path(td) / "q.db")
            queue.register_benches("tier_a_90", ["spector_hls_dct"])
            queue.seed_bench(
                "tier_a_90",
                "spector_hls_dct",
                initial_kind="synth",
                initial_phase="reference",
                initial_stage="gold_gate",
            )
            with queue._conn() as conn:
                row = conn.execute(
                    "SELECT kind, phase, stage FROM jobs WHERE bench=?",
                    ("spector_hls_dct",),
                ).fetchone()
            self.assertEqual(row["kind"], "synth")
            self.assertEqual(row["phase"], "reference")
            self.assertEqual(row["stage"], "gold_gate")


if __name__ == "__main__":
    unittest.main()
