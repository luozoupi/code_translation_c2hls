"""Tests for batch_parallel_queue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_queue import BatchParallelQueue


class BatchParallelQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.queue = BatchParallelQueue(Path(self.tmp.name) / "queue.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_seed_wave_and_claim_assigns_node(self) -> None:
        variant = "aav_n"
        benches = ["jacobi-1d", "gesummv", "correlation", "fdtd-2d"]
        self.queue.register_benches(variant, benches)
        seeded = self.queue.seed_initial_wave(variant, benches, max_inflight=3)
        self.assertEqual(seeded, ["jacobi-1d", "gesummv", "correlation"])
        self.queue.register_node_slot(
            variant=variant, role="synth", node_index=0, worker_slot=0
        )
        job = self.queue.claim(
            kind="codegen",
            worker_id="gpu-1",
        )
        self.assertIsNotNone(job)
        self.assertEqual(job.bench, "jacobi-1d")

    def test_two_workers_same_node_parallel(self) -> None:
        variant = "aav_n"
        self.queue.seed_bench(variant, "jacobi-1d")
        self.queue.seed_bench(variant, "gesummv")
        self.queue.enqueue(
            variant=variant, bench="jacobi-1d", kind="synth", phase="phase_b",
            attempt=0, stage="synth",
        )
        self.queue.enqueue(
            variant=variant, bench="gesummv", kind="synth", phase="phase_b",
            attempt=0, stage="synth",
        )
        self.queue.register_node_slot(variant=variant, role="synth", node_index=0, worker_slot=0)
        self.queue.register_node_slot(variant=variant, role="synth", node_index=0, worker_slot=1)
        j1 = self.queue.claim(
            kind="synth", variant=variant, role="synth", node_index=0, worker_slot=0,
        )
        j2 = self.queue.claim(
            kind="synth", variant=variant, role="synth", node_index=0, worker_slot=1,
        )
        self.assertIsNotNone(j1)
        self.assertIsNotNone(j2)
        self.assertNotEqual(j1.bench, j2.bench)
        self.assertEqual(j1.assigned_node, 0)
        self.assertEqual(j1.assigned_slot, 0)

    def test_cosim_fail_enqueues_codegen_repair(self) -> None:
        variant = "aav_n"
        self.queue.seed_bench(variant, "correlation")
        self.queue.enqueue(
            variant=variant, bench="correlation", kind="codegen", phase="phase_b",
            attempt=1, stage="repair", meta={"repair": {"kind": "cosim"}},
        )
        self.assertEqual(self.queue.pending_codegen(), 2)

    def test_claim_kinds_prefers_first_matching_kind(self) -> None:
        variant = "aav_n"
        self.queue.seed_bench(variant, "b1")
        self.queue.seed_bench(variant, "b2")
        # cosim enqueued first (older created_at) but synth should still win
        # because it comes first in the requested `kinds` preference order.
        self.queue.enqueue(
            variant=variant, bench="b1", kind="cosim", phase="flash",
            attempt=0, stage="cosim",
        )
        self.queue.enqueue(
            variant=variant, bench="b2", kind="synth", phase="flash",
            attempt=0, stage="synth",
        )
        job = self.queue.claim(kinds=("synth", "cosim"), variant=variant)
        self.assertIsNotNone(job)
        self.assertEqual(job.kind, "synth")
        self.assertEqual(job.bench, "b2")

        # Complete the synth job and free its bench lock, then the next claim
        # with the same kind preference should fall back to the cosim job.
        self.queue.complete(job.id)
        job2 = self.queue.claim(kinds=("synth", "cosim"), variant=variant)
        self.assertIsNotNone(job2)
        self.assertEqual(job2.kind, "cosim")
        self.assertEqual(job2.bench, "b1")

    def test_claim_kind_still_works(self) -> None:
        variant = "aav_n"
        self.queue.seed_bench(variant, "b1")
        self.queue.enqueue(
            variant=variant, bench="b1", kind="synth", phase="flash",
            attempt=0, stage="synth",
        )
        job = self.queue.claim(kind="synth", variant=variant)
        self.assertIsNotNone(job)
        self.assertEqual(job.kind, "synth")
        self.assertEqual(job.bench, "b1")

    def test_claim_requires_kind_or_kinds(self) -> None:
        with self.assertRaises(ValueError):
            self.queue.claim(variant="aav_n")

    def test_maybe_seed_deferred_bench(self) -> None:
        variant = "aav_n"
        benches = ["jacobi-1d", "gesummv", "correlation", "fdtd-2d"]
        self.queue.register_benches(variant, benches)
        self.queue.seed_initial_wave(variant, benches, max_inflight=3)
        self.queue.set_bench_status(variant, "jacobi-1d", "done")
        self.queue.set_bench_status(variant, "gesummv", "done")
        self.queue.set_bench_status(variant, "correlation", "done")
        next_bench = self.queue.maybe_seed_next_bench(variant, benches, max_inflight=3)
        self.assertEqual(next_bench, "fdtd-2d")


if __name__ == "__main__":
    unittest.main()
