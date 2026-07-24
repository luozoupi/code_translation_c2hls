"""Tests for Fir batch_parallel queue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from batch_parallel.queue import FirBatchParallelQueue


class FirBatchParallelQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = FirBatchParallelQueue(Path(self.tmp.name) / "queue.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_requeue_benches_from_done(self) -> None:
        benches = ["hlsfactory_2mm", "hlsfactory_symm", "hlsfactory_lu"]
        self.queue.register_benches(benches)
        for bench in benches:
            job = self.queue.claim(node_index=0, worker_slot=0, worker_id="w0")
            self.assertIsNotNone(job)
            self.assertEqual(job.bench, bench)
            self.queue.complete(job.id, success=True, result_path=f"/tmp/{bench}.json")

        requeued = self.queue.requeue_benches(["hlsfactory_2mm", "hlsfactory_lu"])
        self.assertEqual(requeued, ["hlsfactory_2mm", "hlsfactory_lu"])
        jobs = {row["bench"]: row for row in self.queue.all_jobs()}
        self.assertEqual(jobs["hlsfactory_2mm"]["status"], "pending")
        self.assertIsNone(jobs["hlsfactory_2mm"]["result_path"])
        self.assertEqual(jobs["hlsfactory_symm"]["status"], "done")
        self.assertEqual(jobs["hlsfactory_lu"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
