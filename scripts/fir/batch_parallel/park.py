"""GPU park policy for Fir batch_parallel — park during long Vitis phases."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batch_parallel.config import FirBatchParallelConfig
    from batch_parallel.queue import FirBatchParallelQueue

from batch_parallel.gpu_state import gpu_must_stay_up


def normalize_bench_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("hlsfactory_"):
        return text
    return f"hlsfactory_{text}"


def vitis_park_threshold_s(bench: str, cfg: FirBatchParallelConfig) -> float:
    name = normalize_bench_name(bench)
    long_benches = {normalize_bench_name(b) for b in (cfg.long_vitis_benches or [])}
    if name in long_benches:
        return float(cfg.long_vitis_park_s)
    return float(cfg.park_threshold_s)


def evaluate_park_request(
    queue: FirBatchParallelQueue,
    campaign: dict[str, Any],
    cfg: FirBatchParallelConfig,
    campaign_root,
    *,
    now: float | None = None,
) -> str | None:
    """
    Return a park reason when GPU should be released during Vitis-heavy work.

    Never parks while flash benches are pending, LLM is in flight, or claimed
    jobs are still in the initial LLM burst window.
    """
    if str(getattr(cfg, "gpu_policy", "batch_park") or "batch_park") == "always_on":
        return None
    if str((campaign.get("config") or {}).get("gpu_policy") or "") == "always_on":
        return None
    if str(campaign.get("gpu_mode") or "up") != "up":
        return None
    if gpu_must_stay_up(queue, campaign_root, campaign):
        return None
    if queue.pending_flash_count() > 0:
        return None

    ts = time.time() if now is None else now
    best: tuple[float, str] | None = None
    for job in queue.claimed_flash_jobs():
        claimed_at = float(job.get("claimed_at") or 0)
        if claimed_at <= 0:
            continue
        runtime = ts - claimed_at
        threshold = vitis_park_threshold_s(str(job["bench"]), cfg)
        if runtime >= threshold:
            reason = f"long_vitis:{job['bench']}:{runtime:.0f}s>={threshold:.0f}s"
            if best is None or runtime > best[0]:
                best = (runtime, reason)
    return best[1] if best else None


def park_grace_elapsed(campaign: dict[str, Any], cfg: FirBatchParallelConfig, *, now: float | None = None) -> bool:
    pending_at = campaign.get("park_pending_at")
    if pending_at is None:
        return False
    ts = time.time() if now is None else now
    return ts - float(pending_at) >= float(cfg.park_grace_s)


def can_hard_park(
    queue: FirBatchParallelQueue,
    campaign: dict[str, Any],
    cfg: FirBatchParallelConfig,
    campaign_root,
    *,
    now: float | None = None,
) -> tuple[bool, str | None]:
    if gpu_must_stay_up(queue, campaign_root, campaign):
        return False, None
    reason = evaluate_park_request(queue, campaign, cfg, campaign_root, now=now)
    if not reason:
        return False, None
    if not park_grace_elapsed(campaign, cfg, now=now):
        return False, reason
    return True, reason


def vitis_active_count(queue: FirBatchParallelQueue) -> int:
    return queue.claimed_flash_count()


def unpark_flash_min_pending(cfg: FirBatchParallelConfig, queue: FirBatchParallelQueue) -> int:
    batch_threshold = max(1, int(cfg.gpu_batch_threshold))
    if vitis_active_count(queue) < batch_threshold:
        return 1
    return batch_threshold


def should_unpark(
    queue: FirBatchParallelQueue,
    cfg: FirBatchParallelConfig,
    campaign: dict[str, Any],
    *,
    now: float | None = None,
) -> str | None:
    if str(getattr(cfg, "gpu_policy", "batch_park") or "batch_park") == "always_on":
        return None
    if str((campaign.get("config") or {}).get("gpu_policy") or "") == "always_on":
        return None
    if str(campaign.get("gpu_mode") or "up") != "parked":
        return None

    pending = queue.pending_flash_count()
    if pending <= 0:
        return None

    min_pending = unpark_flash_min_pending(cfg, queue)
    if pending >= min_pending:
        if min_pending <= 1:
            return "flash_backlog_low_vitis"
        return f"flash_batch:{min_pending}"

    flush_s = float(getattr(cfg, "gpu_batch_flush_s", 0) or 0)
    if flush_s > 0:
        since = campaign.get("parked_flash_since")
        ts = time.time() if now is None else now
        if since is not None and ts - float(since) >= flush_s:
            return "flash_batch_flush"

    return None
