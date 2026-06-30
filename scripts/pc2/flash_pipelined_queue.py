"""SQLite job queue for pipelined flash (codegen / synth workers)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class PipelinedJob:
    id: int
    variant: str
    bench: str
    kind: str  # codegen | synth
    phase: str  # phase_b | flash
    attempt: int
    stage: str
    meta: dict[str, Any]


class FlashPipelinedQueue:
    """File-backed queue with one in-flight job per (variant, bench)."""

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
                    variant TEXT NOT NULL,
                    bench TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '',
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    worker_id TEXT,
                    created_at REAL NOT NULL,
                    claimed_at REAL,
                    finished_at REAL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_pending
                    ON jobs(status, kind, variant, created_at);
                CREATE TABLE IF NOT EXISTS bench_lock (
                    variant TEXT NOT NULL,
                    bench TEXT NOT NULL,
                    active_job_id INTEGER,
                    bench_status TEXT NOT NULL DEFAULT 'pending',
                    PRIMARY KEY(variant, bench)
                );
                """
            )

    def seed_bench(self, variant: str, bench: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO bench_lock(variant, bench, bench_status)
                VALUES (?, ?, 'pending')
                """,
                (variant, bench),
            )
            row = conn.execute(
                """
                SELECT id FROM jobs
                WHERE variant=? AND bench=? AND status IN ('pending','claimed')
                LIMIT 1
                """,
                (variant, bench),
            ).fetchone()
            if row:
                return
            conn.execute(
                """
                INSERT INTO jobs(variant, bench, kind, phase, attempt, stage, meta_json, status, created_at)
                VALUES (?, ?, 'codegen', 'phase_b', 0, 'translate', '{}', 'pending', ?)
                """,
                (variant, bench, time.time()),
            )

    def enqueue(
        self,
        *,
        variant: str,
        bench: str,
        kind: str,
        phase: str,
        attempt: int,
        stage: str,
        meta: dict[str, Any] | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs(variant, bench, kind, phase, attempt, stage, meta_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    variant,
                    bench,
                    kind,
                    phase,
                    attempt,
                    stage,
                    json.dumps(meta or {}),
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def _row_to_job(self, row: sqlite3.Row) -> PipelinedJob:
        return PipelinedJob(
            id=int(row["id"]),
            variant=row["variant"],
            bench=row["bench"],
            kind=row["kind"],
            phase=row["phase"],
            attempt=int(row["attempt"]),
            stage=row["stage"] or "",
            meta=json.loads(row["meta_json"] or "{}"),
        )

    def claim(self, *, kind: str, variant: str, worker_id: str | None = None) -> PipelinedJob | None:
        wid = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT j.id
                FROM jobs j
                LEFT JOIN bench_lock bl ON bl.variant=j.variant AND bl.bench=j.bench
                WHERE j.status='pending'
                  AND j.kind=?
                  AND j.variant=?
                  AND (bl.active_job_id IS NULL OR bl.active_job_id=0)
                ORDER BY j.created_at ASC, j.id ASC
                LIMIT 1
                """,
                (kind, variant),
            ).fetchone()
            if not row:
                return None
            job_id = int(row["id"])
            updated = conn.execute(
                """
                UPDATE jobs
                SET status='claimed', worker_id=?, claimed_at=?
                WHERE id=? AND status='pending'
                """,
                (wid, now, job_id),
            ).rowcount
            if not updated:
                return None
            job_row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            assert job_row is not None
            conn.execute(
                """
                INSERT INTO bench_lock(variant, bench, active_job_id, bench_status)
                VALUES (?, ?, ?, 'running')
                ON CONFLICT(variant, bench) DO UPDATE SET
                    active_job_id=excluded.active_job_id,
                    bench_status='running'
                """,
                (job_row["variant"], job_row["bench"], job_id),
            )
            return self._row_to_job(job_row)

    def complete(self, job_id: int, *, error: str = "") -> None:
        now = time.time()
        with self._conn() as conn:
            row = conn.execute("SELECT variant, bench FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return
            conn.execute(
                """
                UPDATE jobs
                SET status=?, finished_at=?, error=?
                WHERE id=?
                """,
                ("failed" if error else "done", now, error, job_id),
            )
            conn.execute(
                """
                UPDATE bench_lock
                SET active_job_id=NULL
                WHERE variant=? AND bench=?
                """,
                (row["variant"], row["bench"]),
            )

    def set_bench_status(self, variant: str, bench: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO bench_lock(variant, bench, bench_status)
                VALUES (?, ?, ?)
                ON CONFLICT(variant, bench) DO UPDATE SET bench_status=excluded.bench_status
                """,
                (variant, bench, status),
            )

    def pending_count(self, *, variant: str, kind: str | None = None) -> int:
        with self._conn() as conn:
            if kind:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM jobs
                    WHERE variant=? AND status='pending' AND kind=?
                    """,
                    (variant, kind),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM jobs
                    WHERE variant=? AND status IN ('pending','claimed')
                    """,
                    (variant,),
                ).fetchone()
            return int(row["c"] if row else 0)

    def benches_remaining(self, variant: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM bench_lock
                WHERE variant=? AND bench_status NOT IN ('done','failed')
                """,
                (variant,),
            ).fetchone()
            return int(row["c"] if row else 0)

    def all_benches_terminal(self, variant: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM bench_lock
                WHERE variant=? AND bench_status NOT IN ('done','failed')
                """,
                (variant,),
            ).fetchone()
            return int(row["c"] if row else 0) == 0
