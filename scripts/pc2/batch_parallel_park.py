"""GPU park policy for batch_parallel — queue-driven, not cosim_start-driven."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batch_parallel_config import BatchParallelConfig
    from batch_parallel_queue import BatchParallelQueue

from batch_parallel_gpu_state import gpu_codegen_busy, gpu_must_stay_up

REPO = Path(__file__).resolve().parents[2]
DEFAULT_COSIM_PROFILE_CSV = (
    REPO
    / "artifacts/pc2/analysis/20260628_fixed_cosim_flash_r2_pipelined/csynth_cosim_time_profile.csv"
)

_LONG_BENCH_CACHE: frozenset[str] | None = None


def normalize_bench_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("hlsfactory_"):
        return text
    return f"hlsfactory_{text}"


def bench_short_name(name: str) -> str:
    return normalize_bench_name(name).removeprefix("hlsfactory_")


def load_long_cosim_benches_from_profile(
    csv_path: Path,
    *,
    variant: str = "aav_n",
    min_cosim_s: float = 3600.0,
) -> frozenset[str]:
    """Benches whose profiled cosim (any phase) exceeds min_cosim_s."""
    if not csv_path.is_file():
        return frozenset()
    long: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != variant:
                continue
            try:
                cosim_s = float(row.get("cosim_s") or 0)
            except (TypeError, ValueError):
                continue
            if row.get("cosim_status") == "timeout":
                cosim_s = max(cosim_s, min_cosim_s)
            if cosim_s >= min_cosim_s:
                long.add(normalize_bench_name(str(row["bench"])))
    return frozenset(long)


def resolved_long_cosim_benches(cfg: BatchParallelConfig) -> frozenset[str]:
    global _LONG_BENCH_CACHE
    explicit = {normalize_bench_name(b) for b in (cfg.long_cosim_benches or [])}
    if explicit:
        return frozenset(explicit)
    if _LONG_BENCH_CACHE is not None:
        return _LONG_BENCH_CACHE
    profile = Path(cfg.cosim_profile_csv) if cfg.cosim_profile_csv else DEFAULT_COSIM_PROFILE_CSV
    _LONG_BENCH_CACHE = load_long_cosim_benches_from_profile(
        profile,
        min_cosim_s=float(cfg.long_cosim_profile_min_s),
    )
    return _LONG_BENCH_CACHE


def cosim_park_threshold_s(bench: str, cfg: BatchParallelConfig, long_benches: frozenset[str]) -> float:
    """Seconds of *actual* claimed cosim runtime before GPU park is considered."""
    name = normalize_bench_name(bench)
    if name in long_benches:
        return float(cfg.long_cosim_park_s)
    return float(cfg.park_threshold_s)


def codegen_idle(queue: BatchParallelQueue, campaign_root: Path) -> bool:
    busy, _ = gpu_codegen_busy(queue, campaign_root)
    return not busy


def evaluate_park_request(
    queue: BatchParallelQueue,
    campaign: dict[str, Any],
    cfg: BatchParallelConfig,
    campaign_root: Path,
    *,
    now: float | None = None,
) -> str | None:
    """
    Return a park reason when GPU should be released, else None.

    Never triggers on cosim_start — only when a claimed cosim job has run long
    enough (profile allowlist uses a shorter threshold) and all codegen/LLM
    work is fully idle (queue + in-flight ledger).
    """
    import time

    if str(campaign.get("gpu_mode") or "up") != "up":
        return None
    if gpu_must_stay_up(queue, campaign_root, campaign):
        return None

    long_benches = resolved_long_cosim_benches(cfg)
    ts = time.time() if now is None else now
    best: tuple[float, str] | None = None
    for job in queue.claimed_cosim_jobs():
        claimed_at = float(job.get("claimed_at") or 0)
        if claimed_at <= 0:
            continue
        runtime = ts - claimed_at
        threshold = cosim_park_threshold_s(str(job["bench"]), cfg, long_benches)
        if runtime >= threshold:
            reason = (
                f"long_cosim:{job['bench']}/{job['phase']}:"
                f"{runtime:.0f}s>={threshold:.0f}s"
            )
            if best is None or runtime > best[0]:
                best = (runtime, reason)
    return best[1] if best else None


def can_hard_park(
    queue: BatchParallelQueue,
    campaign: dict[str, Any],
    cfg: BatchParallelConfig,
    campaign_root: Path,
    *,
    now: float | None = None,
) -> tuple[bool, str | None]:
    """All synchronized gates before scancel: idle codegen/LLM, long cosim, grace elapsed."""
    if gpu_must_stay_up(queue, campaign_root, campaign):
        return False, None
    reason = evaluate_park_request(queue, campaign, cfg, campaign_root, now=now)
    if not reason:
        return False, None
    if not park_grace_elapsed(campaign, cfg, now=now):
        return False, reason
    return True, reason


def park_grace_elapsed(campaign: dict[str, Any], cfg: BatchParallelConfig, *, now: float | None = None) -> bool:
    import time

    pending_at = campaign.get("park_pending_at")
    if pending_at is None:
        return False
    ts = time.time() if now is None else now
    return ts - float(pending_at) >= float(cfg.park_grace_s)
