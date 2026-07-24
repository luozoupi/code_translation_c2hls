"""SQLite queue for batch_parallel campaigns (codegen / synth / cosim + node slots)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

TERMINAL_BENCH = frozenset({"done", "failed"})
JOB_KINDS = frozenset({"codegen", "synth", "cosim"})


@dataclass(frozen=True)
class BatchParallelJob:
    id: int
    variant: str
    bench: str
    kind: str
    phase: str
    attempt: int
    stage: str
    meta: dict[str, Any]
    assigned_node: int | None = None
    assigned_slot: int | None = None
    assigned_role: str | None = None


class BatchParallelQueue:
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
                    assigned_role TEXT,
                    assigned_node INTEGER,
                    assigned_slot INTEGER,
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
                    bench_status TEXT NOT NULL DEFAULT 'queued',
                    PRIMARY KEY(variant, bench)
                );

                CREATE TABLE IF NOT EXISTS node_slots (
                    variant TEXT NOT NULL,
                    role TEXT NOT NULL,
                    node_index INTEGER NOT NULL,
                    worker_slot INTEGER NOT NULL,
                    active_job_id INTEGER,
                    hostname TEXT,
                    slurm_job_id TEXT,
                    last_heartbeat REAL,
                    PRIMARY KEY(variant, role, node_index, worker_slot)
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )

    def register_benches(self, variant: str, benches: list[str]) -> None:
        with self._conn() as conn:
            for bench in benches:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO bench_lock(variant, bench, bench_status)
                    VALUES (?, ?, 'queued')
                    """,
                    (variant, bench),
                )

    def seed_bench(
        self,
        variant: str,
        bench: str,
        *,
        initial_kind: str = "codegen",
        initial_phase: str = "phase_b",
        initial_stage: str = "",
    ) -> None:
        if initial_stage:
            stage = initial_stage
        elif initial_phase == "reference":
            stage = "gold_gate"
        elif initial_kind == "codegen" and initial_phase == "phase_b":
            stage = "translate"
        else:
            stage = ""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO bench_lock(variant, bench, bench_status)
                VALUES (?, ?, 'active')
                ON CONFLICT(variant, bench) DO UPDATE SET bench_status='active'
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
                VALUES (?, ?, ?, ?, 0, ?, '{}', 'pending', ?)
                """,
                (variant, bench, initial_kind, initial_phase, stage, time.time()),
            )

    def seed_initial_wave(
        self,
        variant: str,
        benches: list[str],
        *,
        max_inflight: int,
        seed_kwargs: dict[str, str] | None = None,
    ) -> list[str]:
        kw = seed_kwargs or {}
        seeded: list[str] = []
        for bench in benches[:max_inflight]:
            self.seed_bench(variant, bench, **kw)
            seeded.append(bench)
        for bench in benches[max_inflight:]:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO bench_lock(variant, bench, bench_status)
                    VALUES (?, ?, 'queued')
                    """,
                    (variant, bench),
                )
        return seeded

    def count_benches(self, *, variant: str, status: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM bench_lock WHERE variant=? AND bench_status=?",
                (variant, status),
            ).fetchone()
            return int(row["c"] if row else 0)

    def count_in_flight_benches(self, variant: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM bench_lock
                WHERE variant=? AND bench_status NOT IN ('done','failed','queued')
                """,
                (variant,),
            ).fetchone()
            return int(row["c"] if row else 0)

    def next_queued_bench(self, variant: str, benches_order: list[str]) -> str | None:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT bench FROM bench_lock
                WHERE variant=? AND bench_status='queued'
                """,
                (variant,),
            ).fetchall()
        queued = {r["bench"] for r in rows}
        for bench in benches_order:
            if bench in queued:
                return bench
        return None

    def maybe_seed_next_bench(
        self,
        variant: str,
        benches_order: list[str],
        *,
        max_inflight: int,
        seed_kwargs: dict[str, str] | None = None,
    ) -> str | None:
        if self.count_benches(variant=variant, status="queued") == 0:
            return None
        if self.count_in_flight_benches(variant) >= max_inflight:
            return None
        bench = self.next_queued_bench(variant, benches_order)
        if not bench:
            return None
        self.seed_bench(variant, bench, **(seed_kwargs or {}))
        return bench

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

    def _row_to_job(self, row: sqlite3.Row) -> BatchParallelJob:
        return BatchParallelJob(
            id=int(row["id"]),
            variant=row["variant"],
            bench=row["bench"],
            kind=row["kind"],
            phase=row["phase"],
            attempt=int(row["attempt"]),
            stage=row["stage"] or "",
            meta=json.loads(row["meta_json"] or "{}"),
            assigned_node=int(row["assigned_node"]) if row["assigned_node"] is not None else None,
            assigned_slot=int(row["assigned_slot"]) if row["assigned_slot"] is not None else None,
            assigned_role=row["assigned_role"],
        )

    def register_node_slot(
        self,
        *,
        variant: str,
        role: str,
        node_index: int,
        worker_slot: int,
        hostname: str = "",
        slurm_job_id: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO node_slots(variant, role, node_index, worker_slot, hostname, slurm_job_id, last_heartbeat)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(variant, role, node_index, worker_slot) DO UPDATE SET
                    hostname=excluded.hostname,
                    slurm_job_id=excluded.slurm_job_id,
                    last_heartbeat=excluded.last_heartbeat
                """,
                (variant, role, node_index, worker_slot, hostname, slurm_job_id, time.time()),
            )

    def heartbeat_node_slot(
        self,
        *,
        variant: str,
        role: str,
        node_index: int,
        worker_slot: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE node_slots SET last_heartbeat=?
                WHERE variant=? AND role=? AND node_index=? AND worker_slot=?
                """,
                (time.time(), variant, role, node_index, worker_slot),
            )

    def claim(
        self,
        *,
        kind: str | None = None,
        kinds: tuple[str, ...] | None = None,
        variant: str | None = None,
        role: str | None = None,
        node_index: int | None = None,
        worker_slot: int | None = None,
        worker_id: str | None = None,
    ) -> BatchParallelJob | None:
        kind_list = list(kinds) if kinds else ([kind] if kind else [])
        if not kind_list or not all(kind_list):
            raise ValueError("claim() requires either kind or kinds")
        wid = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        now = time.time()
        with self._conn() as conn:
            if role is not None and node_index is not None and worker_slot is not None:
                slot = conn.execute(
                    """
                    SELECT active_job_id FROM node_slots
                    WHERE variant=? AND role=? AND node_index=? AND worker_slot=?
                    """,
                    (variant, role, node_index, worker_slot),
                ).fetchone()
                if slot and slot["active_job_id"]:
                    return None

            kind_placeholders = ",".join("?" for _ in kind_list)
            kind_order_cases = " ".join(
                f"WHEN ? THEN {i}" for i in range(len(kind_list))
            )
            params: list[Any] = list(kind_list)
            variant_clause = ""
            if variant:
                variant_clause = "AND j.variant=?"
                params.append(variant)
            params.extend(kind_list)

            row = conn.execute(
                f"""
                SELECT j.id
                FROM jobs j
                LEFT JOIN bench_lock bl ON bl.variant=j.variant AND bl.bench=j.bench
                WHERE j.status='pending'
                  AND j.kind IN ({kind_placeholders})
                  {variant_clause}
                  AND (bl.active_job_id IS NULL OR bl.active_job_id=0)
                ORDER BY CASE j.kind {kind_order_cases} END ASC, j.created_at ASC, j.id ASC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if not row:
                return None
            job_id = int(row["id"])
            updated = conn.execute(
                """
                UPDATE jobs
                SET status='claimed', worker_id=?, claimed_at=?,
                    assigned_role=?, assigned_node=?, assigned_slot=?
                WHERE id=? AND status='pending'
                """,
                (wid, now, role, node_index, worker_slot, job_id),
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
            if role is not None and node_index is not None and worker_slot is not None:
                conn.execute(
                    """
                    UPDATE node_slots SET active_job_id=?
                    WHERE variant=? AND role=? AND node_index=? AND worker_slot=?
                    """,
                    (job_id, variant, role, node_index, worker_slot),
                )
            return self._row_to_job(job_row)

    def requeue(self, job_id: int) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return False
            if row["status"] not in ("claimed", "failed"):
                return False
            updated = conn.execute(
                """
                UPDATE jobs
                SET status='pending', worker_id=NULL, claimed_at=NULL,
                    assigned_role=NULL, assigned_node=NULL, assigned_slot=NULL,
                    finished_at=NULL, error=''
                WHERE id=? AND status IN ('claimed', 'failed')
                """,
                (job_id,),
            ).rowcount
            if not updated:
                return False
            conn.execute(
                """
                UPDATE bench_lock SET active_job_id=NULL
                WHERE variant=? AND bench=? AND active_job_id=?
                """,
                (row["variant"], row["bench"], job_id),
            )
            if row["assigned_role"] and row["assigned_node"] is not None and row["assigned_slot"] is not None:
                conn.execute(
                    """
                    UPDATE node_slots SET active_job_id=NULL
                    WHERE variant=? AND role=? AND node_index=? AND worker_slot=?
                    """,
                    (row["variant"], row["assigned_role"], row["assigned_node"], row["assigned_slot"]),
                )
            return True

    def requeue_orphaned_claimed(self, *, kinds: tuple[str, ...] = ("cosim", "synth", "codegen")) -> list[int]:
        """Reset claimed jobs back to pending (e.g. after Slurm TIMEOUT)."""
        placeholders = ",".join("?" for _ in kinds)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id FROM jobs
                WHERE kind IN ({placeholders}) AND status='claimed'
                ORDER BY id
                """,
                kinds,
            ).fetchall()
        requeued: list[int] = []
        for row in rows:
            if self.requeue(int(row["id"])):
                requeued.append(int(row["id"]))
        return requeued

    def requeue_stale_claimed(
        self,
        *,
        max_age_s: float,
        kinds: tuple[str, ...] = ("cosim", "synth", "codegen"),
        now: float | None = None,
    ) -> list[int]:
        """Requeue claimed jobs whose worker heartbeat (or claim age) exceeded max_age_s.

        Prefer node_slots.last_heartbeat for the assigned slot. If the slot has no
        heartbeat row, fall back to jobs.claimed_at.
        """
        ts = time.time() if now is None else float(now)
        cutoff = ts - float(max_age_s)
        placeholders = ",".join("?" for _ in kinds)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT j.id, j.claimed_at, j.assigned_role, j.assigned_node, j.assigned_slot,
                       j.variant, ns.last_heartbeat
                FROM jobs j
                LEFT JOIN node_slots ns
                  ON ns.variant = j.variant
                 AND ns.role = j.assigned_role
                 AND ns.node_index = j.assigned_node
                 AND ns.worker_slot = j.assigned_slot
                WHERE j.kind IN ({placeholders}) AND j.status='claimed'
                ORDER BY j.id
                """,
                kinds,
            ).fetchall()
        requeued: list[int] = []
        for row in rows:
            heartbeat = row["last_heartbeat"]
            claimed_at = row["claimed_at"]
            age_anchor = heartbeat if heartbeat is not None else claimed_at
            if age_anchor is None:
                continue
            if float(age_anchor) > cutoff:
                continue
            if self.requeue(int(row["id"])):
                requeued.append(int(row["id"]))
        return requeued

    def fail_benches(self, benches: list[str], *, error: str, variant: str | None = None) -> list[int]:
        """Mark pending/claimed jobs for benches as failed and unlock the bench."""
        failed: list[int] = []
        now = time.time()
        with self._conn() as conn:
            for bench in benches:
                params: list[Any] = [bench]
                variant_clause = ""
                if variant:
                    variant_clause = "AND variant=?"
                    params.append(variant)
                rows = conn.execute(
                    f"""
                    SELECT id, variant FROM jobs
                    WHERE bench=? {variant_clause}
                      AND status IN ('pending', 'claimed')
                    ORDER BY id
                    """,
                    tuple(params),
                ).fetchall()
                for row in rows:
                    job_id = int(row["id"])
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status='failed', finished_at=?, error=?,
                            worker_id=NULL, claimed_at=NULL,
                            assigned_role=NULL, assigned_node=NULL, assigned_slot=NULL
                        WHERE id=? AND status IN ('pending', 'claimed')
                        """,
                        (now, error, job_id),
                    )
                    conn.execute(
                        """
                        UPDATE bench_lock
                        SET active_job_id=NULL, bench_status='failed'
                        WHERE variant=? AND bench=?
                        """,
                        (row["variant"], bench),
                    )
                    failed.append(job_id)
                if not rows:
                    # Ensure lock shows failed even if no open jobs (e.g. gemm_blocked).
                    if variant:
                        conn.execute(
                            """
                            INSERT INTO bench_lock(variant, bench, active_job_id, bench_status)
                            VALUES (?, ?, NULL, 'failed')
                            ON CONFLICT(variant, bench) DO UPDATE SET
                                active_job_id=NULL,
                                bench_status='failed'
                            """,
                            (variant, bench),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE bench_lock
                            SET active_job_id=NULL, bench_status='failed'
                            WHERE bench=?
                            """,
                            (bench,),
                        )
        return failed

    def clear_node_slot_assignments(self, *, role: str | None = None) -> int:
        with self._conn() as conn:
            if role:
                return conn.execute(
                    "UPDATE node_slots SET active_job_id=NULL WHERE role=?",
                    (role,),
                ).rowcount
            return conn.execute("UPDATE node_slots SET active_job_id=NULL").rowcount

    def complete(self, job_id: int, *, error: str = "") -> None:
        now = time.time()
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
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
                UPDATE bench_lock SET active_job_id=NULL
                WHERE variant=? AND bench=?
                """,
                (row["variant"], row["bench"]),
            )
            if row["assigned_role"] and row["assigned_node"] is not None and row["assigned_slot"] is not None:
                conn.execute(
                    """
                    UPDATE node_slots SET active_job_id=NULL
                    WHERE variant=? AND role=? AND node_index=? AND worker_slot=?
                    """,
                    (row["variant"], row["assigned_role"], row["assigned_node"], row["assigned_slot"]),
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

    def pending_count(self, *, variant: str | None = None, kind: str | None = None) -> int:
        with self._conn() as conn:
            clauses = ["status IN ('pending','claimed')"]
            params: list[Any] = []
            if variant:
                clauses.append("variant=?")
                params.append(variant)
            if kind:
                clauses.append("kind=?")
                params.append(kind)
            where = " AND ".join(clauses)
            row = conn.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {where}", tuple(params)).fetchone()
            return int(row["c"] if row else 0)

    def pending_codegen(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE kind='codegen' AND status='pending'"
            ).fetchone()
            return int(row["c"] if row else 0)

    def claimed_codegen(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE kind='codegen' AND status='claimed'"
            ).fetchone()
            return int(row["c"] if row else 0)

    def claimed_cosim_jobs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, variant, bench, phase, claimed_at
                FROM jobs
                WHERE kind='cosim' AND status='claimed'
                ORDER BY claimed_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def codegen_demand_count(self) -> int:
        """Benches that still need GPU codegen (queued or not yet enqueued after phase_b)."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT b.bench) AS c
                FROM bench_lock b
                WHERE b.bench_status NOT IN ('done', 'failed', 'queued')
                  AND (
                    EXISTS (
                      SELECT 1 FROM jobs j
                      WHERE j.bench = b.bench AND j.kind = 'codegen'
                        AND j.status IN ('pending', 'claimed')
                    )
                    OR (
                      EXISTS (
                        SELECT 1 FROM jobs j
                        WHERE j.bench = b.bench AND j.phase = 'phase_b'
                          AND j.kind = 'cosim' AND j.status = 'done'
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM jobs j
                        WHERE j.bench = b.bench AND j.phase = 'flash'
                          AND j.kind = 'codegen' AND j.status = 'done'
                      )
                    )
                    OR (
                      EXISTS (
                        SELECT 1 FROM jobs j
                        WHERE j.bench = b.bench AND j.phase = 'phase_b'
                          AND j.kind = 'synth' AND j.status = 'done'
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM jobs j
                        WHERE j.bench = b.bench AND j.phase = 'flash'
                      )
                    )
                  )
                """
            ).fetchone()
        return int(row["c"] if row else 0)

    def pending_or_claimed_count(self, *, kinds: tuple[str, ...]) -> int:
        placeholders = ",".join("?" for _ in kinds)
        with self._conn() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS c FROM jobs
                WHERE kind IN ({placeholders}) AND status IN ('pending','claimed')
                """,
                kinds,
            ).fetchone()
            return int(row["c"] if row else 0)

    def pending_count_global(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status IN ('pending','claimed')"
            ).fetchone()
            return int(row["c"] if row else 0)

    def claimed_count_global(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status='claimed'"
            ).fetchone()
            return int(row["c"] if row else 0)

    def benches_non_terminal(self, variant: str | None = None) -> int:
        with self._conn() as conn:
            if variant:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM bench_lock
                    WHERE variant=? AND bench_status NOT IN ('done','failed')
                    """,
                    (variant,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM bench_lock
                    WHERE bench_status NOT IN ('done','failed')
                    """
                ).fetchone()
            return int(row["c"] if row else 0)

    def all_benches_terminal(self, variant: str) -> bool:
        return self.benches_non_terminal(variant) == 0

    def campaign_complete(self, active_variants: list[str]) -> bool:
        for variant in active_variants:
            if not self.all_benches_terminal(variant):
                return False
        if self.pending_count_global() > 0 or self.claimed_count_global() > 0:
            return False
        return self.codegen_demand_count() == 0

    def snapshot_node_map(self) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT ns.*, j.bench, j.phase, j.kind
                FROM node_slots ns
                LEFT JOIN jobs j ON j.id = ns.active_job_id
                ORDER BY ns.variant, ns.role, ns.node_index, ns.worker_slot
                """
            ).fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            variant = row["variant"]
            role = row["role"]
            node_key = f"node_{row['node_index']}"
            slot_key = f"slot_{row['worker_slot']}"
            out.setdefault(variant, {}).setdefault(role, {}).setdefault(node_key, {})
            if row["active_job_id"]:
                out[variant][role][node_key][slot_key] = f"{row['bench']}/{row['phase']}"
            else:
                out[variant][role][node_key][slot_key] = None
        return out

    def set_meta(self, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO meta(key, value_json) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
                """,
                (key, json.dumps(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute("SELECT value_json FROM meta WHERE key=?", (key,)).fetchone()
            if not row:
                return default
            return json.loads(row["value_json"])
