#!/usr/bin/env python3
"""Discrete-event simulation for shared-GPU flash + async cosim campaigns.

Uses post-batch cosim/csynth timings from csynth_cosim_time_profile.csv.
Explores bench ordering, GPU park/unpark policies, and tail repair thrash.

GPU policies:
  always_on       — GPU never parked
  naive_park      — unpark on each cosim repair (bad at tail)
  supervisor_park — unpark on every codegen enqueue while parked (many unparks)
  batch_park      — accumulate codegen while parked; unpark when batch full (recommended)

Worker layout (matches Slurm):
  synth_nodes_per_variant × synth_workers_per_node — csynth+csim per variant (default 2×4=8)
  cosim_nodes_per_variant × cosim_workers_per_node — max 2 cosims per cosim node (default 4×2=8/variant)
  cosim_mode pooled     — shared farm across all variant cosim nodes
  cosim_mode per_variant — each variant's cosim nodes serve only that variant

Example::

    .venv/bin/python scripts/pc2/simulate_flash_campaign.py --scenario short_first_batch_parallel
    .venv/bin/python scripts/pc2/simulate_flash_campaign.py --cosim-mode per_variant
    .venv/bin/python scripts/pc2/simulate_flash_campaign.py --synth-nodes-per-variant 2 --cosim-nodes-per-variant 4
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = (
    REPO
    / "artifacts/pc2/analysis/20260628_fixed_cosim_flash_r2_pipelined/csynth_cosim_time_profile.csv"
)
VARIANTS = ("aav_n", "aav_o", "nav_n", "nav_o", "noskills")

# Slurm layout defaults (prior: 1 synth node × 4 workers; 1 cosim node uncapped)
DEFAULT_SYNTH_NODES_PER_VARIANT = 2
DEFAULT_SYNTH_WORKERS_PER_NODE = 4
DEFAULT_COSIM_NODES_PER_VARIANT = 4
DEFAULT_COSIM_WORKERS_PER_NODE = 2
DEFAULT_GPU_QUEUE_S = 2 * 3600  # Slurm pending before parked GPU node is RUNNING

# ---------------------------------------------------------------------------
# Duration model (seconds) — from profile CSV + simple LLM overhead
# ---------------------------------------------------------------------------

CODEGEN_S = 90.0  # one LLM translate / repair call
REPAIR_ROUNDS_MAX = 4


@dataclass(frozen=True)
class PhaseProfile:
    csynth_s: float
    cosim_s: float
    cosim_passed: bool


@dataclass
class BenchProfile:
    bench: str
    phase_b: PhaseProfile
    flash: PhaseProfile

    def sort_key(self) -> float:
        """Short-first ordering: defer benches with longer cosim tail."""
        return max(self.phase_b.cosim_s, self.flash.cosim_s)


def load_profiles(path: Path) -> dict[str, dict[str, BenchProfile]]:
    """variant -> bench -> BenchProfile."""
    raw: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["variant"], row["bench"], row["phase"])
            raw[key] = row

    out: dict[str, dict[str, BenchProfile]] = {v: {} for v in VARIANTS}
    benches = sorted({k[1] for k in raw})
    for variant in VARIANTS:
        for bench in benches:
            def phase(name: str) -> PhaseProfile:
                row = raw.get((variant, bench, name))
                if not row:
                    return PhaseProfile(30.0, 60.0, True)
                cosim_s = float(row["cosim_s"] or 0) if row.get("cosim_s") else 60.0
                if row.get("cosim_status") == "timeout":
                    cosim_s = max(cosim_s, 12 * 3600)
                passed = str(row.get("cosim_passed", "")).lower() in ("true", "1", "yes")
                return PhaseProfile(
                    csynth_s=float(row["csynth_final_s"] or 30),
                    cosim_s=cosim_s,
                    cosim_passed=passed,
                )

            out[variant][bench] = BenchProfile(bench, phase("phase_b"), phase("flash"))
    return out


# ---------------------------------------------------------------------------
# Discrete-event simulation
# ---------------------------------------------------------------------------

Event = tuple[float, int, str, dict[str, Any]]  # time, seq, kind, payload


@dataclass
class GpuState:
    mode: str = "up"  # up | parked | pending_unpark
    busy_until: float = 0.0
    unpark_ready_at: float = 0.0
    total_up_s: float = 0.0
    last_up_start: float | None = None
    park_events: int = 0
    unpark_requests: int = 0
    unpark_completions: int = 0


@dataclass
class SimConfig:
    bench_order: str  # lexical | short_first
    gpu_policy: str  # always_on | naive_park | supervisor_park | batch_park
    park_threshold_s: float
    gpu_queue_s: float  # Slurm pending before GPU RUNNING
    repair_cooldown_s: float  # stay up after last codegen before park
    codegen_batch_threshold: int  # batch_park: unpark when this many codegen jobs wait while parked
    codegen_batch_flush_s: float  # batch_park: also unpark if oldest wait exceeds (0=off)
    synth_nodes_per_variant: int
    synth_workers_per_node: int
    cosim_mode: str  # pooled | per_variant
    cosim_nodes_per_variant: int
    cosim_workers_per_node: int  # max concurrent cosims on one cosim node
    variants: tuple[str, ...]
    simulate_repairs: bool  # retry cosim up to REPAIR_ROUNDS_MAX on historical fail
    variant_stagger_benches: int  # offset sort order per variant index (spread tails)
    bench_serial: bool  # one bench at a time per variant (closer to ordered campaign)
    gpu_nodes: int = 1  # shared codegen pool; park/unpark applies to all nodes together

    @property
    def synth_workers_per_variant(self) -> int:
        return self.synth_nodes_per_variant * self.synth_workers_per_node

    @property
    def cosim_slots_per_variant(self) -> int:
        return self.cosim_nodes_per_variant * self.cosim_workers_per_node

    @property
    def cosim_slots_pooled(self) -> int:
        return len(self.variants) * self.cosim_slots_per_variant


@dataclass
class SimMetrics:
    wall_s: float
    gpu_hours: float
    park_events: int
    unpark_requests: int
    unpark_completions: int
    pending_codegen_max: int
    pending_cosim_max: int
    gpu_queue_wait_s: float
    repair_codegen_jobs: int
    worker_layout: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


class CampaignSimulator:
    def __init__(
        self,
        profiles: dict[str, dict[str, BenchProfile]],
        config: SimConfig,
    ) -> None:
        self.profiles = profiles
        self.cfg = config
        self.t = 0.0
        self._seq = 0
        self._queue: list[Event] = []
        self.gpu = GpuState()
        self.gpu_busy_until: list[float] = [0.0] * max(1, config.gpu_nodes)
        self.pending_codegen = 0
        self.pending_cosim = 0
        self.pending_codegen_max = 0
        self.pending_cosim_max = 0
        self.gpu_queue_wait_s = 0.0
        self.repair_codegen_jobs = 0
        self.events: list[dict[str, Any]] = []
        self._parked_codegen_since: float | None = None  # batch_park: first enqueue while parked

        self.synth_busy: dict[str, int] = {v: 0 for v in config.variants}
        self.synth_q: dict[str, list[tuple[str, str, int]]] = {
            v: [] for v in config.variants
        }  # bench, phase, attempt

        self.cosim_node_busy_var: dict[str, list[int]] = {
            v: [0] * config.cosim_nodes_per_variant for v in config.variants
        }
        self.cosim_node_busy_pooled: list[int] = [
            0] * (len(config.variants) * config.cosim_nodes_per_variant)
        self.cosim_q_pooled: list[tuple[str, str, str, int]] = []  # variant,bench,phase,attempt
        self.cosim_q: dict[str, list[tuple[str, str, str, int]]] = {
            v: [] for v in config.variants
        }

        # bench state: variant -> bench -> dict
        self.bench_state: dict[str, dict[str, dict[str, Any]]] = {}
        self._bench_queues: dict[str, list[str]] = {}
        self._active_bench: dict[str, str | None] = {v: None for v in config.variants}

    def _log(self, kind: str, **payload: Any) -> None:
        self.events.append({"t": self.t, "kind": kind, **payload})

    def _schedule(self, delay: float, kind: str, **payload: Any) -> None:
        self._seq += 1
        heapq.heappush(self._queue, (self.t + delay, self._seq, kind, payload))

    def _bench_order(self, variant: str) -> list[str]:
        prof = self.profiles[variant]
        benches = list(prof.keys())
        if self.cfg.bench_order == "lexical":
            return sorted(benches)
        ordered = sorted(benches, key=lambda b: prof[b].sort_key())
        if self.cfg.variant_stagger_benches:
            idx = self.cfg.variants.index(variant)
            offset = idx * self.cfg.variant_stagger_benches
            ordered = ordered[offset:] + ordered[:offset]
        return ordered

    def _account_gpu_up(self) -> None:
        if self.gpu.last_up_start is not None:
            elapsed = self.t - self.gpu.last_up_start
            self.gpu.total_up_s += elapsed * self.cfg.gpu_nodes
        self.gpu.last_up_start = self.t

    def _account_gpu_down(self) -> None:
        if self.gpu.last_up_start is not None:
            elapsed = self.t - self.gpu.last_up_start
            self.gpu.total_up_s += elapsed * self.cfg.gpu_nodes
            self.gpu.last_up_start = None

    def _gpu_has_free_slot(self) -> bool:
        return any(self.t >= until for until in self.gpu_busy_until)

    def _gpu_free_slot(self) -> int | None:
        for i, until in enumerate(self.gpu_busy_until):
            if self.t >= until:
                return i
        return None

    def _gpu_request_up(self, reason: str) -> None:
        if self.cfg.gpu_policy == "always_on":
            return
        if self.gpu.mode == "up":
            return
        if self.gpu.mode == "pending_unpark":
            return
        self.gpu.unpark_requests += 1
        self._log("gpu_unpark_request", reason=reason, policy=self.cfg.gpu_policy)
        self.gpu.mode = "pending_unpark"
        self.gpu.unpark_ready_at = self.t + self.cfg.gpu_queue_s
        self.gpu_queue_wait_s += self.cfg.gpu_queue_s
        self._schedule(self.cfg.gpu_queue_s, "gpu_unpark_ready", reason=reason)

    def _gpu_unpark_ready(self, reason: str) -> None:
        if self.gpu.mode != "pending_unpark":
            return
        self.gpu.mode = "up"
        self.gpu.unpark_completions += 1
        self._account_gpu_up()
        self._log("gpu_up", reason=reason)
        self._clear_parked_codegen_clock()
        self._drain_codegen()

    def _gpu_try_park(self, reason: str) -> None:
        if self.cfg.gpu_policy == "always_on":
            return
        if self.gpu.mode == "parked":
            return
        if self.pending_codegen > 0:
            return
        if self.gpu.mode == "pending_unpark":
            return
        self._account_gpu_down()
        self.gpu.mode = "parked"
        self.gpu.park_events += 1
        self._log("gpu_parked", reason=reason)
        self._clear_parked_codegen_clock()

    def _needs_park_for_cosim(self, cosim_s: float) -> bool:
        return (
            self.cfg.gpu_policy != "always_on"
            and cosim_s >= self.cfg.park_threshold_s
        )

    def _maybe_batch_unpark(self, reason: str = "batch_threshold") -> None:
        """Unpark only when enough codegen work accumulated while GPU is parked."""
        if self.cfg.gpu_policy != "batch_park":
            return
        if self.gpu.mode != "parked":
            return
        if self.pending_codegen <= 0:
            return
        hit_size = self.pending_codegen >= self.cfg.codegen_batch_threshold
        hit_time = False
        if self.cfg.codegen_batch_flush_s > 0 and self._parked_codegen_since is not None:
            hit_time = (self.t - self._parked_codegen_since) >= self.cfg.codegen_batch_flush_s
        if not (hit_size or hit_time):
            return
        self._log(
            "batch_unpark_trigger",
            pending=self.pending_codegen,
            threshold=self.cfg.codegen_batch_threshold,
            reason=reason,
        )
        self._gpu_request_up(f"batch:{reason}:{self.pending_codegen}")

    def _note_parked_codegen(self) -> None:
        if self.cfg.gpu_policy != "batch_park":
            return
        if self.gpu.mode == "parked" and self._parked_codegen_since is None:
            self._parked_codegen_since = self.t

    def _clear_parked_codegen_clock(self) -> None:
        self._parked_codegen_since = None

    def _enqueue_codegen(
        self,
        variant: str,
        bench: str,
        phase: str,
        attempt: int,
        reason: str,
    ) -> None:
        self.pending_codegen += 1
        self.pending_codegen_max = max(self.pending_codegen_max, self.pending_codegen)
        if reason == "cosim_repair":
            self.repair_codegen_jobs += 1
        self._log("codegen_enqueued", variant=variant, bench=bench, phase=phase, attempt=attempt, reason=reason)

        if self.cfg.gpu_policy == "batch_park":
            if self.gpu.mode != "up":
                self._note_parked_codegen()
                self._maybe_batch_unpark(reason)
        elif self.cfg.gpu_policy == "naive_park":
            if reason == "cosim_repair" or self.gpu.mode == "parked":
                self._gpu_request_up(f"codegen:{variant}/{bench}/{phase}")
        elif self.cfg.gpu_policy == "supervisor_park":
            self._gpu_request_up(f"codegen:{variant}/{bench}/{phase}")

        if self.gpu.mode == "up" and self._gpu_has_free_slot():
            self._drain_codegen()

    def _drain_codegen(self) -> None:
        while self.pending_codegen > 0 and self.gpu.mode == "up":
            node = self._gpu_free_slot()
            if node is None:
                break
            job = self._pop_codegen_job()
            if job is None:
                break
            variant, bench, phase, attempt = job
            self.pending_codegen -= 1
            self.gpu_busy_until[node] = self.t + CODEGEN_S
            self._log(
                "codegen_run",
                variant=variant,
                bench=bench,
                phase=phase,
                attempt=attempt,
                dur=CODEGEN_S,
                gpu_node=node,
            )
            self._schedule(
                CODEGEN_S,
                "codegen_done",
                variant=variant,
                bench=bench,
                phase=phase,
                attempt=attempt,
                gpu_node=node,
            )

    def _pop_codegen_job(self) -> tuple[str, str, str, int] | None:
        best: tuple[float, str, str, str, int] | None = None
        for variant, benches in self.bench_state.items():
            for bench, st in benches.items():
                if st.get("codegen_pending"):
                    ts = st["codegen_pending_ts"]
                    payload = st["codegen_pending"]
                    cand = (ts, variant, bench, payload["phase"], payload["attempt"])
                    if best is None or cand < best:
                        best = cand
        if best is None:
            return None
        _, variant, bench, phase, attempt = best
        del self.bench_state[variant][bench]["codegen_pending"]
        del self.bench_state[variant][bench]["codegen_pending_ts"]
        return variant, bench, phase, attempt

    def _mark_codegen_pending(self, variant: str, bench: str, phase: str, attempt: int) -> None:
        st = self.bench_state[variant][bench]
        st["codegen_pending"] = {"phase": phase, "attempt": attempt}
        st["codegen_pending_ts"] = self.t

    def _start_variant_bench(self, variant: str, bench: str) -> None:
        self.bench_state[variant][bench] = {
            "stage": "phase_b",
            "attempt": 0,
            "cosim_attempt": 0,
        }
        self._active_bench[variant] = bench
        self._mark_codegen_pending(variant, bench, "phase_b", 0)
        self._enqueue_codegen(variant, bench, "phase_b", 0, "initial_translate")

    def _maybe_start_next_bench(self, variant: str) -> None:
        if not self.cfg.bench_serial:
            return
        if self._active_bench.get(variant) is not None:
            return
        q = self._bench_queues.get(variant) or []
        while q:
            bench = q.pop(0)
            if self.bench_state[variant].get(bench, {}).get("stage") != "done":
                self._start_variant_bench(variant, bench)
                return

    def _profile(self, variant: str, bench: str, phase: str) -> PhaseProfile:
        bp = self.profiles[variant][bench]
        return bp.phase_b if phase == "phase_b" else bp.flash

    def _worker_layout_label(self) -> str:
        n = len(self.cfg.variants)
        synth = (
            f"synth={n}x{self.cfg.synth_nodes_per_variant}x{self.cfg.synth_workers_per_node}"
            f"({n * self.cfg.synth_workers_per_variant})"
        )
        nodes = self.cfg.cosim_nodes_per_variant
        cap = self.cfg.cosim_workers_per_node
        if self.cfg.cosim_mode == "per_variant":
            cosim = (
                f"cosim={n}x{nodes}x{cap}"
                f"({n * self.cfg.cosim_slots_per_variant})"
            )
        else:
            total_nodes = n * nodes
            cosim = f"cosim=pooled:{total_nodes}x{cap}({self.cfg.cosim_slots_pooled})"
        gpu = f"gpu={self.cfg.gpu_nodes}"
        return f"{synth} {cosim} {gpu}"

    def _alloc_cosim_node_per_variant(self, variant: str) -> int | None:
        cap = self.cfg.cosim_workers_per_node
        for node_i, busy in enumerate(self.cosim_node_busy_var[variant]):
            if busy < cap:
                return node_i
        return None

    def _alloc_cosim_node_pooled(self) -> int | None:
        cap = self.cfg.cosim_workers_per_node
        for node_i, busy in enumerate(self.cosim_node_busy_pooled):
            if busy < cap:
                return node_i
        return None

    def _cosim_busy_per_variant(self, variant: str) -> int:
        return sum(self.cosim_node_busy_var[variant])

    def _cosim_busy_pooled(self) -> int:
        return sum(self.cosim_node_busy_pooled)

    def _schedule_synth(self, variant: str, bench: str, phase: str, attempt: int) -> None:
        self.synth_q[variant].append((bench, phase, attempt))
        self._try_synth()

    def _try_synth(self) -> None:
        for variant in self.cfg.variants:
            limit = self.cfg.synth_workers_per_variant
            while self.synth_busy[variant] < limit and self.synth_q[variant]:
                bench, phase, attempt = self.synth_q[variant].pop(0)
                prof = self._profile(variant, bench, phase)
                dur = prof.csynth_s
                self.synth_busy[variant] += 1
                self._log(
                    "synth_start",
                    variant=variant,
                    bench=bench,
                    phase=phase,
                    attempt=attempt,
                    dur=dur,
                    slot=f"{self.synth_busy[variant]}/{limit}",
                )
                self._schedule(
                    dur,
                    "synth_done",
                    variant=variant,
                    bench=bench,
                    phase=phase,
                    attempt=attempt,
                )

    def _schedule_cosim(self, variant: str, bench: str, phase: str, attempt: int) -> None:
        prof = self._profile(variant, bench, phase)
        if self._needs_park_for_cosim(prof.cosim_s):
            self._gpu_try_park(f"long_cosim:{variant}/{bench}/{phase}")
        self.pending_cosim += 1
        self.pending_cosim_max = max(self.pending_cosim_max, self.pending_cosim)
        job = (variant, bench, phase, attempt)
        if self.cfg.cosim_mode == "per_variant":
            self.cosim_q[variant].append(job)
        else:
            self.cosim_q_pooled.append(job)
        self._try_cosim()

    def _try_cosim(self) -> None:
        if self.cfg.cosim_mode == "per_variant":
            for v in self.cfg.variants:
                while self.cosim_q[v]:
                    node = self._alloc_cosim_node_per_variant(v)
                    if node is None:
                        break
                    variant, bench, phase, attempt = self.cosim_q[v].pop(0)
                    self._start_cosim_job(
                        variant, bench, phase, attempt, node=node, per_variant=True,
                    )
            return

        while self.cosim_q_pooled:
            node = self._alloc_cosim_node_pooled()
            if node is None:
                break
            variant, bench, phase, attempt = self.cosim_q_pooled.pop(0)
            self._start_cosim_job(
                variant, bench, phase, attempt, node=node, per_variant=False,
            )

    def _start_cosim_job(
        self,
        variant: str,
        bench: str,
        phase: str,
        attempt: int,
        *,
        node: int,
        per_variant: bool,
    ) -> None:
        prof = self._profile(variant, bench, phase)
        dur = prof.cosim_s
        cap = self.cfg.cosim_workers_per_node
        if per_variant:
            self.cosim_node_busy_var[variant][node] += 1
            busy = self.cosim_node_busy_var[variant][node]
            pool = f"{variant}:node{node}:{busy}/{cap}"
            total_busy = self._cosim_busy_per_variant(variant)
            total_cap = self.cfg.cosim_slots_per_variant
        else:
            self.cosim_node_busy_pooled[node] += 1
            busy = self.cosim_node_busy_pooled[node]
            pool = f"pooled:node{node}:{busy}/{cap}"
            total_busy = self._cosim_busy_pooled()
            total_cap = self.cfg.cosim_slots_pooled
        self.pending_cosim -= 1
        self._log(
            "cosim_start",
            variant=variant,
            bench=bench,
            phase=phase,
            attempt=attempt,
            dur=dur,
            slot=pool,
            farm=f"{total_busy}/{total_cap}",
        )
        self._schedule(
            dur,
            "cosim_done",
            variant=variant,
            bench=bench,
            phase=phase,
            attempt=attempt,
            node=node,
            per_variant=per_variant,
        )

    def _cosim_should_fail(self, variant: str, bench: str, phase: str, attempt: int) -> bool:
        if not self.cfg.simulate_repairs:
            return False
        prof = self._profile(variant, bench, phase)
        if prof.cosim_passed:
            return False
        # Historical fail: fail on first attempts, pass on last
        return attempt < REPAIR_ROUNDS_MAX - 1

    def _has_pending_work(self) -> bool:
        if self.pending_codegen > 0:
            return True
        if any(self.synth_q.values()):
            return True
        if self.cosim_q_pooled or any(self.cosim_q.values()):
            return True
        if sum(self.synth_busy.values()) > 0:
            return True
        if sum(sum(nodes) for nodes in self.cosim_node_busy_var.values()) > 0:
            return True
        if sum(self.cosim_node_busy_pooled) > 0:
            return True
        return not self._all_benches_complete()

    def _all_benches_complete(self) -> bool:
        for variant in self.cfg.variants:
            for bench in self.profiles[variant]:
                if self.bench_state.get(variant, {}).get(bench, {}).get("stage") != "done":
                    return False
        return True

    def _incomplete_benches(self) -> list[str]:
        missing: list[str] = []
        for variant in self.cfg.variants:
            for bench in self.profiles[variant]:
                st = self.bench_state.get(variant, {}).get(bench, {})
                if st.get("stage") != "done":
                    stage = st.get("stage", "missing")
                    missing.append(f"{variant}/{bench}:{stage}")
        return missing

    def _assert_all_benches_complete(self) -> None:
        missing = self._incomplete_benches()
        if not missing:
            return
        expected = sum(len(self.profiles[v]) for v in self.cfg.variants)
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview += f", ... (+{len(missing) - 10} more)"
        raise RuntimeError(
            f"simulation ended with {len(missing)}/{expected} benches incomplete: {preview}"
        )

    def _dispatch_event(self, kind: str, payload: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{kind}")
        handler(**payload)
        if self.cfg.gpu_policy == "supervisor_park":
            self._supervisor_tick()
        elif self.cfg.gpu_policy == "batch_park":
            self._maybe_batch_unpark("periodic_check")

    def _pump_workers(self) -> None:
        if self.gpu.mode == "up" and self._gpu_has_free_slot():
            self._drain_codegen()
        self._try_synth()
        self._try_cosim()
        if self.cfg.gpu_policy == "supervisor_park":
            self._supervisor_tick()
        elif self.cfg.gpu_policy == "batch_park":
            self._maybe_batch_unpark("pump")

    def _run_event_loop(self) -> None:
        while self._queue:
            self.t, _, kind, payload = heapq.heappop(self._queue)
            self._dispatch_event(kind, payload)

    def run(self) -> SimMetrics:
        # Seed benches: all parallel, or one-at-a-time per variant (ordered campaign)
        for variant in self.cfg.variants:
            self.bench_state[variant] = {}
            order = self._bench_order(variant)
            self._bench_queues[variant] = list(order)
            if self.cfg.bench_serial:
                self._active_bench[variant] = None
                self._maybe_start_next_bench(variant)
            else:
                for bench in order:
                    self._start_variant_bench(variant, bench)

        if self.cfg.gpu_policy != "always_on":
            self.gpu.mode = "up"
            self._account_gpu_up()  # start hot for early fast benches
        else:
            self.gpu.mode = "up"
            self._account_gpu_up()

        self._drain_codegen()

        self._run_event_loop()

        # Drain tail: codegen backlog, batch_park unpark, or stranded synth/cosim queues.
        for _ in range(10_000):
            if not self._has_pending_work():
                break
            if (
                self.cfg.gpu_policy == "batch_park"
                and self.pending_codegen > 0
                and self.gpu.mode == "parked"
            ):
                self._gpu_request_up(f"batch:tail_flush:{self.pending_codegen}")
            elif (
                self.pending_codegen > 0
                and self.gpu.mode == "parked"
                and self.cfg.gpu_policy in ("naive_park", "supervisor_park")
            ):
                self._gpu_request_up(f"tail_flush:{self.pending_codegen}")
            self._pump_workers()
            if self._queue:
                self._run_event_loop()
            elif self.pending_codegen > 0 and self.gpu.mode == "up" and self._gpu_has_free_slot():
                before = self.pending_codegen
                self._drain_codegen()
                if self.pending_codegen == before and not self._queue:
                    raise RuntimeError(
                        f"stalled with pending_codegen={self.pending_codegen} and idle GPU"
                    )

        self._assert_all_benches_complete()

        self._account_gpu_down()

        return SimMetrics(
            wall_s=self.t,
            gpu_hours=self.gpu.total_up_s / 3600,
            park_events=self.gpu.park_events,
            unpark_requests=self.gpu.unpark_requests,
            unpark_completions=self.gpu.unpark_completions,
            pending_codegen_max=self.pending_codegen_max,
            pending_cosim_max=self.pending_cosim_max,
            gpu_queue_wait_s=self.gpu_queue_wait_s,
            repair_codegen_jobs=self.repair_codegen_jobs,
            worker_layout=self._worker_layout_label(),
            events=self.events,
        )

    def _supervisor_tick(self) -> None:
        if self.pending_codegen > 0:
            self._gpu_request_up("supervisor:pending_codegen")
        elif self.gpu.mode == "up" and self.pending_codegen == 0:
            # cooldown before park handled via scheduled event at codegen_done
            pass

    def _on_gpu_unpark_ready(self, reason: str) -> None:
        self._gpu_unpark_ready(reason)

    def _on_codegen_done(
        self,
        variant: str,
        bench: str,
        phase: str,
        attempt: int,
        gpu_node: int = 0,
    ) -> None:
        self._schedule_synth(variant, bench, phase, attempt)
        if self.gpu.mode == "up":
            self._drain_codegen()

    def _on_synth_done(self, variant: str, bench: str, phase: str, attempt: int) -> None:
        self.synth_busy[variant] -= 1
        self._try_synth()
        self._schedule_cosim(variant, bench, phase, attempt)

    def _on_cosim_done(
        self,
        variant: str,
        bench: str,
        phase: str,
        attempt: int,
        node: int,
        per_variant: bool,
    ) -> None:
        if per_variant:
            self.cosim_node_busy_var[variant][node] -= 1
        else:
            self.cosim_node_busy_pooled[node] -= 1
        self._try_cosim()
        st = self.bench_state[variant][bench]
        failed = self._cosim_should_fail(variant, bench, phase, attempt)

        if failed:
            next_attempt = attempt + 1
            st["attempt"] = next_attempt
            self._mark_codegen_pending(variant, bench, phase, next_attempt)
            if self.cfg.gpu_policy == "naive_park":
                self._gpu_request_up(f"cosim_fail:{variant}/{bench}/{phase}")
            self._enqueue_codegen(variant, bench, phase, next_attempt, "cosim_repair")
            if self.gpu.mode == "up" and self._gpu_has_free_slot():
                self._drain_codegen()
            return

        if phase == "phase_b":
            st["stage"] = "flash"
            st["attempt"] = 0
            self._mark_codegen_pending(variant, bench, "flash", 0)
            self._enqueue_codegen(variant, bench, "flash", 0, "flash_start")
            return

        st["stage"] = "done"
        self._log("bench_done", variant=variant, bench=bench)
        if self.cfg.bench_serial and self._active_bench.get(variant) == bench:
            self._active_bench[variant] = None
            self._maybe_start_next_bench(variant)

        if self.cfg.gpu_policy in ("supervisor_park", "batch_park") and self.pending_codegen == 0:
            self._schedule(self.cfg.repair_cooldown_s, "supervisor_maybe_park", reason="post_repair_cooldown")

    def _on_supervisor_maybe_park(self, reason: str) -> None:
        if self.pending_codegen == 0:
            self._gpu_try_park(reason)


SCENARIOS: dict[str, Callable[[], SimConfig]] = {
    "baseline": lambda: SimConfig(
        bench_order="lexical",
        gpu_policy="always_on",
        park_threshold_s=2 * 3600,
        gpu_queue_s=0,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
    "short_first_always_on": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="always_on",
        park_threshold_s=2 * 3600,
        gpu_queue_s=0,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
    "short_first_supervisor": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="supervisor_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
    "short_first_naive_park": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="naive_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=0,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
    "short_first_supervisor_slurm2h": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="supervisor_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
    "short_first_supervisor_serial": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="supervisor_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=True,
    ),
    "short_first_batch_serial": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="batch_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=True,
    ),
    "short_first_batch_serial_b10": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="batch_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=10,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=True,
    ),
    "lexical_supervisor_serial": lambda: SimConfig(
        bench_order="lexical",
        gpu_policy="supervisor_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=True,
    ),
    "lexical_batch_serial": lambda: SimConfig(
        bench_order="lexical",
        gpu_policy="batch_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=True,
    ),
    "short_first_batch_parallel": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="batch_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="pooled",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
    "short_first_batch_parallel_pervar_cosim": lambda: SimConfig(
        bench_order="short_first",
        gpu_policy="batch_park",
        park_threshold_s=2 * 3600,
        gpu_queue_s=DEFAULT_GPU_QUEUE_S,
        repair_cooldown_s=1800,
        codegen_batch_threshold=5,
        codegen_batch_flush_s=0,
        synth_nodes_per_variant=DEFAULT_SYNTH_NODES_PER_VARIANT,
        synth_workers_per_node=DEFAULT_SYNTH_WORKERS_PER_NODE,
        cosim_mode="per_variant",
        cosim_nodes_per_variant=DEFAULT_COSIM_NODES_PER_VARIANT,
        cosim_workers_per_node=DEFAULT_COSIM_WORKERS_PER_NODE,
        variants=VARIANTS,
        simulate_repairs=True,
        variant_stagger_benches=0,
        bench_serial=False,
    ),
}


def format_metrics(name: str, m: SimMetrics, cfg: SimConfig) -> str:
    days = m.wall_s / 86400
    layout = m.worker_layout
    return (
        f"{name:36s}  wall={days:6.2f}d  gpu_h={m.gpu_hours:7.0f}h  "
        f"park={m.park_events:3d}  unpark={m.unpark_completions:3d}  "
        f"gpu_q={m.gpu_queue_wait_s/3600:6.0f}h  max_cg_q={m.pending_codegen_max:2d}  "
        f"[{layout}]"
    )


def write_timeline(path: Path, events: list[dict[str, Any]], limit: int = 500) -> None:
    lines = ["# Campaign simulation timeline (first N events)", ""]
    for ev in events[:limit]:
        t_h = ev["t"] / 3600
        extra = {k: v for k, v in ev.items() if k not in ("t", "kind")}
        lines.append(f"{t_h:8.1f}h  {ev['kind']:20s}  {extra}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="short_first_supervisor")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--gpu-nodes", type=int, default=None, help="parallel GPU codegen nodes")
    parser.add_argument(
        "--gpu-queue-hours",
        type=float,
        default=None,
        help="override Slurm GPU pending (default: 2h for park policies, 0 for always_on)",
    )
    parser.add_argument("--park-threshold-hours", type=float, default=2.0)
    parser.add_argument(
        "--synth-nodes-per-variant", type=int, default=DEFAULT_SYNTH_NODES_PER_VARIANT,
        help="csynth+csim compute nodes per variant (default 2 = doubled)",
    )
    parser.add_argument(
        "--synth-workers-per-node", type=int, default=DEFAULT_SYNTH_WORKERS_PER_NODE,
        help="workers per synth node",
    )
    parser.add_argument("--synth-workers-per-variant", type=int, default=None,
                        help="deprecated: sets 1 node with N workers")
    parser.add_argument("--synth-workers", type=int, default=None,
                        help="deprecated alias for --synth-workers-per-variant")
    parser.add_argument(
        "--cosim-nodes-per-variant", type=int, default=DEFAULT_COSIM_NODES_PER_VARIANT,
        help="cosim nodes per variant (default 4 = quadrupled)",
    )
    parser.add_argument(
        "--cosim-workers-per-node", type=int, default=DEFAULT_COSIM_WORKERS_PER_NODE,
        help="max concurrent cosims per cosim node (default 2)",
    )
    parser.add_argument("--cosim-mode", choices=("pooled", "per_variant"), default=None,
                        help="pooled=shared farm; per_variant=dedicated nodes per variant")
    parser.add_argument("--cosim-workers", type=int, default=None,
                        help="deprecated: total pooled/per-variant cosim slots")
    parser.add_argument("--bench-serial", action="store_true", help="one bench at a time per variant")
    parser.add_argument("--no-repairs", action="store_true", help="assume all cosim pass first try")
    parser.add_argument("--codegen-batch-size", type=int, default=None, help="batch_park unpark threshold")
    parser.add_argument("--codegen-batch-flush-hours", type=float, default=None, help="max wait while parked")
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/pc2/analysis/campaign_sim")
    parser.add_argument("--write-timeline", action="store_true")
    args = parser.parse_args()

    profiles = load_profiles(args.profile_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(SCENARIOS) if args.compare_all else [args.scenario]
    rows: list[dict[str, Any]] = []

    print(f"Profile: {args.profile_csv}")
    print(f"Variants: {len(VARIANTS)}  benches/variant: {len(next(iter(profiles.values())))}")
    print("Worker model: 2× synth nodes × 4 workers/variant; 4× cosim nodes × 2 cap/variant")
    print()

    for name in names:
        cfg = SCENARIOS[name]()
        cfg.park_threshold_s = args.park_threshold_hours * 3600
        cfg.synth_nodes_per_variant = args.synth_nodes_per_variant
        cfg.synth_workers_per_node = args.synth_workers_per_node
        if args.synth_workers is not None:
            cfg.synth_nodes_per_variant = 1
            cfg.synth_workers_per_node = args.synth_workers
        if args.synth_workers_per_variant is not None:
            cfg.synth_nodes_per_variant = 1
            cfg.synth_workers_per_node = args.synth_workers_per_variant
        cfg.cosim_nodes_per_variant = args.cosim_nodes_per_variant
        cfg.cosim_workers_per_node = args.cosim_workers_per_node
        if args.cosim_workers is not None:
            cap = cfg.cosim_workers_per_node
            total_nodes = max(1, (args.cosim_workers + cap - 1) // cap)
            if cfg.cosim_mode == "pooled":
                cfg.cosim_nodes_per_variant = max(
                    1, total_nodes // len(cfg.variants),
                )
            else:
                cfg.cosim_nodes_per_variant = total_nodes
        if args.cosim_mode is not None:
            cfg.cosim_mode = args.cosim_mode
        if args.gpu_nodes is not None:
            cfg.gpu_nodes = max(1, args.gpu_nodes)
        cfg.simulate_repairs = not args.no_repairs
        if args.bench_serial:
            cfg.bench_serial = True
        if args.gpu_queue_hours is not None:
            cfg.gpu_queue_s = args.gpu_queue_hours * 3600
        elif cfg.gpu_policy != "always_on":
            cfg.gpu_queue_s = DEFAULT_GPU_QUEUE_S
        if args.codegen_batch_size is not None:
            cfg.codegen_batch_threshold = args.codegen_batch_size
        if args.codegen_batch_flush_hours is not None:
            cfg.codegen_batch_flush_s = args.codegen_batch_flush_hours * 3600

        sim = CampaignSimulator(profiles, cfg)
        metrics = sim.run()
        print(format_metrics(name, metrics, cfg))
        row = {"scenario": name, "worker_layout": metrics.worker_layout, **metrics.__dict__}
        row.pop("events")
        rows.append(row)

        if args.write_timeline and not args.compare_all:
            tl = args.out_dir / f"timeline_{name}.txt"
            write_timeline(tl, metrics.events)
            print(f"Wrote {tl}")

    out_json = args.out_dir / "comparison.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
