"""Tests for batch_parallel GPU park policy and busy ledger."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from batch_parallel_config import BatchParallelConfig
from batch_parallel_gpu_state import (
    begin_llm_request,
    end_llm_request,
    gpu_codegen_busy,
    gpu_must_stay_up,
    is_retriable_llm_error,
    read_llm_in_flight,
)
from batch_parallel_park import (
    can_hard_park,
    codegen_idle,
    evaluate_park_request,
    load_long_cosim_benches_from_profile,
    normalize_bench_name,
    park_grace_elapsed,
    should_unpark,
)
from batch_parallel_queue import BatchParallelQueue

PROFILE_CSV = (
    REPO
    / "artifacts/pc2/analysis/20260628_fixed_cosim_flash_r2_pipelined/csynth_cosim_time_profile.csv"
)


class BatchParallelParkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.queue = BatchParallelQueue(self.root / "queue.db")
        self.cfg = BatchParallelConfig(
            park_threshold_s=7200.0,
            long_cosim_park_s=3600.0,
            park_grace_s=1800.0,
            long_cosim_benches=["hlsfactory_fdtd-2d"],
            gpu_batch_threshold=5,
            gpu_batch_flush_s=3600,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_cosim(self, bench: str, *, claimed_at: float) -> int:
        variant = "aav_n"
        self.queue.register_benches(variant, [bench])
        job_id = self.queue.enqueue(
            variant=variant,
            bench=bench,
            kind="cosim",
            phase="phase_b",
            attempt=0,
            stage="cosim",
        )
        with self.queue._conn() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='claimed', claimed_at=?, worker_id='test-cosim'
                WHERE id=?
                """,
                (claimed_at, job_id),
            )
        return job_id

    def test_no_park_on_recent_cosim_start(self) -> None:
        now = time.time()
        self._seed_cosim("hlsfactory_jacobi-1d", claimed_at=now - 30)
        campaign = {"gpu_mode": "up"}
        self.assertIsNone(
            evaluate_park_request(self.queue, campaign, self.cfg, self.root, now=now)
        )

    def test_no_park_while_codegen_claimed(self) -> None:
        now = time.time()
        variant = "aav_n"
        self.queue.register_benches(variant, ["hlsfactory_fdtd-2d", "hlsfactory_atax"])
        self.queue.seed_bench(variant, "hlsfactory_atax")
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 3600)
        self.queue.claim(kind="codegen", worker_id="gpu-1")
        campaign = {"gpu_mode": "up"}
        self.assertIsNone(
            evaluate_park_request(self.queue, campaign, self.cfg, self.root, now=now)
        )
        self.assertFalse(codegen_idle(self.queue, self.root))

    def test_no_park_while_llm_in_flight(self) -> None:
        now = time.time()
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 7200)
        begin_llm_request(
            self.root,
            job_id=99,
            variant="aav_n",
            bench="hlsfactory_correlation",
            phase="phase_b",
            worker="test",
        )
        campaign = {"gpu_mode": "up"}
        self.assertTrue(gpu_must_stay_up(self.queue, self.root, campaign))
        self.assertIsNone(
            evaluate_park_request(self.queue, campaign, self.cfg, self.root, now=now)
        )
        end_llm_request(self.root, job_id=99)

    def test_park_only_after_long_cosim_runtime_on_allowlist(self) -> None:
        now = time.time()
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 3700)
        campaign = {"gpu_mode": "up"}
        reason = evaluate_park_request(self.queue, campaign, self.cfg, self.root, now=now)
        self.assertIsNotNone(reason)
        self.assertIn("hlsfactory_fdtd-2d", reason or "")

    def test_short_bench_never_parks_within_typical_cosim(self) -> None:
        now = time.time()
        self._seed_cosim("hlsfactory_gesummv", claimed_at=now - 120)
        campaign = {"gpu_mode": "up"}
        self.assertIsNone(
            evaluate_park_request(self.queue, campaign, self.cfg, self.root, now=now)
        )

    def test_hard_park_blocked_during_grace(self) -> None:
        now = time.time()
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 3700)
        campaign = {"gpu_mode": "up", "park_pending_at": now - 60}
        ready, _ = can_hard_park(self.queue, campaign, self.cfg, self.root, now=now)
        self.assertFalse(ready)

    def test_hard_park_allowed_after_grace_when_idle(self) -> None:
        now = time.time()
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 3700)
        campaign = {"gpu_mode": "up", "park_pending_at": now - 2000}
        ready, reason = can_hard_park(self.queue, campaign, self.cfg, self.root, now=now)
        self.assertTrue(ready)
        self.assertIn("fdtd-2d", reason or "")

    def test_profile_loader_finds_fdtd(self) -> None:
        long_benches = load_long_cosim_benches_from_profile(PROFILE_CSV, min_cosim_s=3600.0)
        self.assertIn(normalize_bench_name("fdtd-2d"), long_benches)

    def test_park_grace_elapsed(self) -> None:
        campaign = {"park_pending_at": time.time() - 1900}
        self.assertTrue(park_grace_elapsed(campaign, self.cfg))

    def test_retriable_llm_error(self) -> None:
        self.assertTrue(is_retriable_llm_error(RuntimeError("Connection error.")))

    def test_requeue_restores_pending_codegen(self) -> None:
        variant = "aav_n"
        self.queue.register_benches(variant, ["hlsfactory_correlation"])
        self.queue.seed_bench(variant, "hlsfactory_correlation")
        job = self.queue.claim(kind="codegen", worker_id="gpu-1")
        assert job is not None
        self.assertTrue(self.queue.requeue(job.id))
        self.assertEqual(self.queue.pending_codegen(), 1)
        self.assertEqual(self.queue.claimed_codegen(), 0)

    def test_llm_ledger_roundtrip(self) -> None:
        begin_llm_request(
            self.root,
            job_id=1,
            variant="aav_n",
            bench="hlsfactory_gesummv",
            phase="flash",
            worker="gpu-drain-1",
        )
        busy, blockers = gpu_codegen_busy(self.queue, self.root)
        self.assertTrue(busy)
        self.assertTrue(any("llm_in_flight" in b for b in blockers))
        self.assertIsNotNone(read_llm_in_flight(self.root))
        end_llm_request(self.root, job_id=1)
        busy, _ = gpu_codegen_busy(self.queue, self.root)
        self.assertFalse(busy)

    def test_unpark_on_codegen_backlog_when_vitis_quiet(self) -> None:
        variant = "aav_n"
        self.queue.register_benches(variant, ["hlsfactory_2mm"])
        self.queue.seed_bench(variant, "hlsfactory_2mm")
        self.queue.enqueue(
            variant=variant,
            bench="hlsfactory_2mm",
            kind="codegen",
            phase="flash",
            attempt=0,
            stage="flash",
        )
        campaign = {"gpu_mode": "parked"}
        reason = should_unpark(self.queue, self.cfg, campaign)
        self.assertEqual(reason, "codegen_backlog_low_vitis")

    def test_no_unpark_below_batch_threshold_while_vitis_busy(self) -> None:
        variant = "aav_n"
        benches = [f"hlsfactory_b{i}" for i in range(4)]
        self.queue.register_benches(variant, benches)
        for bench in benches:
            self.queue.enqueue(
                variant=variant,
                bench=bench,
                kind="codegen",
                phase="flash",
                attempt=0,
                stage="flash",
            )
        for i in range(5):
            jid = self.queue.enqueue(
                variant=variant,
                bench="hlsfactory_fdtd-2d",
                kind="cosim",
                phase="phase_b",
                attempt=0,
                stage="cosim",
            )
            with self.queue._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET status='claimed', worker_id=? WHERE id=?",
                    (f"w{i}", jid),
                )
        campaign = {"gpu_mode": "parked"}
        self.assertIsNone(should_unpark(self.queue, self.cfg, campaign))

    def test_unpark_at_batch_threshold_while_vitis_busy(self) -> None:
        variant = "aav_n"
        benches = [f"hlsfactory_b{i}" for i in range(5)]
        self.queue.register_benches(variant, benches)
        for bench in benches:
            self.queue.enqueue(
                variant=variant,
                bench=bench,
                kind="codegen",
                phase="flash",
                attempt=0,
                stage="flash",
            )
        for i in range(5):
            jid = self.queue.enqueue(
                variant=variant,
                bench="hlsfactory_fdtd-2d",
                kind="cosim",
                phase="phase_b",
                attempt=0,
                stage="cosim",
            )
            with self.queue._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET status='claimed', worker_id=? WHERE id=?",
                    (f"w{i}", jid),
                )
        campaign = {"gpu_mode": "parked"}
        reason = should_unpark(self.queue, self.cfg, campaign)
        self.assertEqual(reason, "codegen_batch:5")

    def test_unpark_batch_flush_after_wait(self) -> None:
        variant = "aav_n"
        self.queue.register_benches(variant, ["hlsfactory_2mm"])
        self.queue.enqueue(
            variant=variant,
            bench="hlsfactory_2mm",
            kind="codegen",
            phase="flash",
            attempt=0,
            stage="flash",
        )
        for i in range(5):
            jid = self.queue.enqueue(
                variant=variant,
                bench="hlsfactory_fdtd-2d",
                kind="cosim",
                phase="phase_b",
                attempt=0,
                stage="cosim",
            )
            with self.queue._conn() as conn:
                conn.execute(
                    "UPDATE jobs SET status='claimed', worker_id=? WHERE id=?",
                    (f"w{i}", jid),
                )
        now = time.time()
        campaign = {"gpu_mode": "parked", "parked_codegen_since": now - 3700}
        reason = should_unpark(self.queue, self.cfg, campaign, now=now)
        self.assertEqual(reason, "codegen_batch_flush")

    def test_no_park_when_gpu_policy_always_on(self) -> None:
        now = time.time()
        cfg = BatchParallelConfig(
            park_threshold_s=7200.0,
            long_cosim_park_s=3600.0,
            park_grace_s=1800.0,
            long_cosim_benches=["hlsfactory_fdtd-2d"],
            gpu_policy="always_on",
        )
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 7200)
        campaign = {"gpu_mode": "up", "config": {"gpu_policy": "always_on"}}
        self.assertIsNone(
            evaluate_park_request(self.queue, campaign, cfg, self.root, now=now)
        )

    def test_no_park_when_flash_codegen_demand(self) -> None:
        now = time.time()
        variant = "aav_n"
        self.queue.register_benches(variant, ["hlsfactory_fdtd-2d", "hlsfactory_2mm"])
        # fdtd long cosim running
        self._seed_cosim("hlsfactory_fdtd-2d", claimed_at=now - 4000)
        # 2mm phase_b cosim done, flash codegen pending
        for kind, phase, status in (
            ("codegen", "phase_b", "done"),
            ("synth", "phase_b", "done"),
            ("cosim", "phase_b", "done"),
            ("codegen", "flash", "pending"),
        ):
            jid = self.queue.enqueue(
                variant=variant, bench="hlsfactory_2mm", kind=kind, phase=phase,
                attempt=0, stage=kind,
            )
            with self.queue._conn() as conn:
                conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, jid))
        campaign = {"gpu_mode": "up"}
        self.assertIsNone(
            evaluate_park_request(self.queue, campaign, self.cfg, self.root, now=now)
        )


if __name__ == "__main__":
    unittest.main()
