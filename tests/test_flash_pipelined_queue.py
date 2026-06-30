"""Tests for pipelined flash queue."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from flash_pipelined_queue import FlashPipelinedQueue


class FlashPipelinedQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.queue = FlashPipelinedQueue(Path(self._tmpdir.name) / "queue.db")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_seed_and_claim_one_bench_at_a_time(self) -> None:
        queue = self.queue
        queue.seed_bench("nav_o", "hlsfactory_2mm")

        job_a = queue.claim(kind="codegen", variant="nav_o", worker_id="codegen-1")
        self.assertIsNotNone(job_a)
        assert job_a is not None
        self.assertEqual(job_a.bench, "hlsfactory_2mm")

        job_b = queue.claim(kind="codegen", variant="nav_o", worker_id="codegen-1")
        self.assertIsNone(job_b)

        queue.complete(job_a.id)
        job_c = queue.claim(kind="codegen", variant="nav_o", worker_id="codegen-1")
        self.assertIsNone(job_c)

    def test_different_benches_can_run_in_parallel(self) -> None:
        queue = self.queue
        queue.seed_bench("nav_o", "hlsfactory_2mm")
        queue.seed_bench("nav_o", "hlsfactory_3mm")

        job_a = queue.claim(kind="codegen", variant="nav_o", worker_id="codegen-1")
        job_b = queue.claim(kind="codegen", variant="nav_o", worker_id="codegen-1")
        self.assertIsNotNone(job_a)
        self.assertIsNotNone(job_b)
        assert job_a is not None and job_b is not None
        self.assertNotEqual(job_a.bench, job_b.bench)

    def test_same_bench_blocks_second_job(self) -> None:
        queue = self.queue
        queue.seed_bench("nav_o", "hlsfactory_syrk")
        job_a = queue.claim(kind="codegen", variant="nav_o", worker_id="cg")
        self.assertIsNotNone(job_a)
        queue.enqueue(
            variant="nav_o",
            bench="hlsfactory_syrk",
            kind="synth",
            phase="phase_b",
            attempt=0,
            stage="synth",
        )
        job_b = queue.claim(kind="synth", variant="nav_o", worker_id="sy")
        self.assertIsNone(job_b)
        assert job_a is not None
        queue.complete(job_a.id)
        job_c = queue.claim(kind="synth", variant="nav_o", worker_id="sy")
        self.assertIsNotNone(job_c)

    def test_enqueue_followup_and_synth_claim(self) -> None:
        queue = self.queue
        queue.seed_bench("nav_o", "hlsfactory_syrk")
        codegen_job = queue.claim(kind="codegen", variant="nav_o", worker_id="cg")
        self.assertIsNotNone(codegen_job)
        assert codegen_job is not None
        queue.enqueue(
            variant="nav_o",
            bench="hlsfactory_syrk",
            kind="synth",
            phase="phase_b",
            attempt=0,
            stage="synth",
        )
        queue.complete(codegen_job.id)

        synth_job = queue.claim(kind="synth", variant="nav_o", worker_id="sy")
        self.assertIsNotNone(synth_job)
        assert synth_job is not None
        self.assertEqual(synth_job.kind, "synth")
        self.assertEqual(synth_job.phase, "phase_b")

    def test_different_benches_can_run_synth_in_parallel(self) -> None:
        queue = self.queue
        for bench in ("hlsfactory_2mm", "hlsfactory_3mm", "hlsfactory_syrk"):
            queue.seed_bench("nav_o", bench)
            queue.enqueue(
                variant="nav_o",
                bench=bench,
                kind="synth",
                phase="phase_b",
                attempt=0,
                stage="synth",
            )

        claimed = []
        for worker_idx in range(3):
            job = queue.claim(kind="synth", variant="nav_o", worker_id=f"synth-{worker_idx}")
            if job is not None:
                claimed.append(job)

        self.assertEqual(len(claimed), 3)
        benches = {job.bench for job in claimed}
        self.assertEqual(benches, {"hlsfactory_2mm", "hlsfactory_3mm", "hlsfactory_syrk"})

    def test_bench_status_terminal(self) -> None:
        queue = self.queue
        queue.seed_bench("nav_o", "hlsfactory_gemm")
        queue.set_bench_status("nav_o", "hlsfactory_gemm", "done")
        self.assertTrue(queue.all_benches_terminal("nav_o"))

    def test_multistep_phase_enqueue(self) -> None:
        queue = self.queue
        queue.seed_bench("aav_n", "hlsfactory_2mm")
        seed_job = queue.claim(kind="codegen", variant="aav_n", worker_id="cg")
        self.assertIsNotNone(seed_job)
        assert seed_job is not None
        queue.complete(seed_job.id)
        queue.enqueue(
            variant="aav_n",
            bench="hlsfactory_2mm",
            kind="codegen",
            phase="tiling",
            attempt=0,
            stage="optimize",
        )
        job = queue.claim(kind="codegen", variant="aav_n", worker_id="cg")
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.phase, "tiling")


if __name__ == "__main__":
    unittest.main()
