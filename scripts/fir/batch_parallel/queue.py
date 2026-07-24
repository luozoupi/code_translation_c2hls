"""SQLite queue for Fir batch_parallel (one flash job per bench)."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

TERMINAL = frozenset({"done", "failed"})

RETRYABLE_ERROR_MARKERS = (
    "connection error",
    "connection refused",
    "connection reset",
    "connect timeout",
    "timed out",
    "timeout",
    "endpoint not reachable",
    "failed to establish a new connection",
    "name or service not known",
)


def is_retryable_error(error: str) -> bool:
    msg = (error or "").lower()
    return any(marker in msg for marker in RETRYABLE_ERROR_MARKERS)


@dataclass(frozen=True)
class FirBenchJob:
    id: int
    bench: str
    status: str
    worker_id: str | None = None
    node_index: int | None = None
    worker_slot: int | None = None
    error: str | None = None
    result_path: str | None = None


class FirBatchParallelQueue:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=120.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bench TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    worker_id TEXT,
                    node_index INTEGER,
                    worker_slot INTEGER,
                    error TEXT,
                    result_path TEXT,
                    created_at REAL NOT NULL,
                    claimed_at REAL,
                    finished_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS node_slots (
                    node_index INTEGER NOT NULL,
                    worker_slot INTEGER NOT NULL,
                    active_job_id INTEGER,
                    hostname TEXT,
                    slurm_job_id TEXT,
                    last_heartbeat REAL,
                    PRIMARY KEY(node_index, worker_slot)
                );
                """
            )

    def register_benches(self, benches: list[str]) -> int:
        now = time.time()
        added = 0
        with self._conn() as conn:
            for bench in benches:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO jobs(bench, status, created_at) VALUES (?, 'pending', ?)",
                    (bench, now),
                )
                added += cur.rowcount
        return added

    def pending_flash_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'pending'").fetchone()
            return int(row[0])

    def claimed_flash_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'claimed'").fetchone()
            return int(row[0])

    def claimed_flash_jobs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, bench, claimed_at, worker_id, node_index, worker_slot "
                "FROM jobs WHERE status = 'claimed' ORDER BY claimed_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def young_claimed_count(self, *, grace_s: float) -> int:
        cutoff = time.time() - grace_s
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'claimed' AND claimed_at > ?",
                (cutoff,),
            ).fetchone()
            return int(row[0])

    def pending_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'claimed')"
            ).fetchone()
            return int(row[0])

    def campaign_complete(self) -> bool:
        with self._conn() as conn:
            rows = conn.execute("SELECT status, error FROM jobs").fetchall()
        for row in rows:
            status = str(row["status"])
            if status in ("pending", "claimed"):
                return False
            if status == "failed" and is_retryable_error(str(row["error"] or "")):
                return False
        return True

    def retryable_failed_count(self) -> int:
        with self._conn() as conn:
            rows = conn.execute("SELECT error FROM jobs WHERE status = 'failed'").fetchall()
        return sum(1 for row in rows if is_retryable_error(str(row["error"] or "")))

    def requeue(self, job_id: int, *, error: str | None = None) -> None:
        """Return a job to pending. Clears stored error unless error= is passed explicitly."""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='pending', worker_id=NULL, node_index=NULL, worker_slot=NULL,
                    error=?, result_path=NULL, claimed_at=NULL, finished_at=NULL
                WHERE id=?
                """,
                (error, job_id),
            )
            conn.execute(
                "UPDATE node_slots SET active_job_id=NULL, last_heartbeat=? WHERE active_job_id=?",
                (now, job_id),
            )

    def requeue_retryable_failures(self) -> int:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, error FROM jobs WHERE status = 'failed'").fetchall()
        requeued = 0
        for row in rows:
            if is_retryable_error(str(row["error"] or "")):
                self.requeue(int(row["id"]))
                requeued += 1
        return requeued

    def requeue_benches_matching(self, *, error_substring: str) -> int:
        needle = error_substring.lower()
        with self._conn() as conn:
            rows = conn.execute("SELECT id, error FROM jobs WHERE status = 'failed'").fetchall()
        requeued = 0
        for row in rows:
            if needle in str(row["error"] or "").lower():
                self.requeue(int(row["id"]))
                requeued += 1
        return requeued

    def requeue_benches(
        self,
        benches: list[str],
        *,
        from_statuses: frozenset[str] = TERMINAL,
    ) -> list[str]:
        """Requeue terminal jobs for the given bench names (e.g. after infra retry)."""
        if not benches:
            return []
        wanted = {str(b) for b in benches}
        requeued: list[str] = []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(wanted))
            rows = conn.execute(
                f"SELECT id, bench, status FROM jobs WHERE bench IN ({placeholders})",
                tuple(sorted(wanted)),
            ).fetchall()
        for row in rows:
            bench = str(row["bench"])
            if str(row["status"]) not in from_statuses:
                continue
            self.requeue(int(row["id"]))
            requeued.append(bench)
        return sorted(requeued)

    def claim(
        self,
        *,
        node_index: int,
        worker_slot: int,
        worker_id: str,
    ) -> FirBenchJob | None:
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, bench FROM jobs
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status='claimed', worker_id=?, node_index=?, worker_slot=?,
                    claimed_at=?, error=NULL
                WHERE id=? AND status='pending'
                """,
                (worker_id, node_index, worker_slot, now, row["id"]),
            )
            if conn.total_changes == 0:
                return None
            conn.execute(
                """
                INSERT INTO node_slots(node_index, worker_slot, active_job_id, last_heartbeat)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_index, worker_slot) DO UPDATE SET
                    active_job_id=excluded.active_job_id,
                    last_heartbeat=excluded.last_heartbeat
                """,
                (node_index, worker_slot, row["id"], now),
            )
            return FirBenchJob(
                id=int(row["id"]),
                bench=str(row["bench"]),
                status="claimed",
                worker_id=worker_id,
                node_index=node_index,
                worker_slot=worker_slot,
            )

    def complete(
        self,
        job_id: int,
        *,
        success: bool,
        error: str = "",
        result_path: str = "",
    ) -> None:
        now = time.time()
        status = "done" if success else "failed"
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, error=?, result_path=?, finished_at=?
                WHERE id=?
                """,
                (status, error or None, result_path or None, now, job_id),
            )
            conn.execute(
                "UPDATE node_slots SET active_job_id=NULL, last_heartbeat=? WHERE active_job_id=?",
                (now, job_id),
            )

    def register_node_slot(
        self,
        *,
        node_index: int,
        worker_slot: int,
        hostname: str,
        slurm_job_id: str,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO node_slots(node_index, worker_slot, hostname, slurm_job_id, last_heartbeat)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_index, worker_slot) DO UPDATE SET
                    hostname=excluded.hostname,
                    slurm_job_id=excluded.slurm_job_id,
                    last_heartbeat=excluded.last_heartbeat
                """,
                (node_index, worker_slot, hostname, slurm_job_id, now),
            )

    def heartbeat_node_slot(self, *, node_index: int, worker_slot: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE node_slots SET last_heartbeat=? WHERE node_index=? AND worker_slot=?",
                (time.time(), node_index, worker_slot),
            )

    def snapshot_node_map(self) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT node_index, worker_slot, active_job_id, hostname, slurm_job_id, last_heartbeat "
                "FROM node_slots ORDER BY node_index, worker_slot"
            ).fetchall()
        return {
            "slots": [
                {
                    "node_index": int(r["node_index"]),
                    "worker_slot": int(r["worker_slot"]),
                    "active_job_id": r["active_job_id"],
                    "hostname": r["hostname"],
                    "slurm_job_id": r["slurm_job_id"],
                    "last_heartbeat": r["last_heartbeat"],
                }
                for r in rows
            ]
        }

    def all_jobs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT bench, status, error, result_path, worker_id, node_index, worker_slot, finished_at "
                "FROM jobs ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]
