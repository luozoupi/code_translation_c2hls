"""Central GPU / LLM busy ledger for Fir batch_parallel park decisions."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

RETRIABLE_LLM_MARKERS = (
    "connection error",
    "connection reset",
    "connection refused",
    "endpoint",
    "timed out",
    "timeout",
    "remote end closed",
    "broken pipe",
)


def is_retriable_llm_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in RETRIABLE_LLM_MARKERS)


def _campaign_root_from_env() -> Path | None:
    raw = os.getenv("C2HLS_FIR_BATCH_CAMPAIGN_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).resolve()


def _state_path(campaign_root: Path) -> Path:
    return campaign_root.resolve() / "flow" / "gpu_llm.json"


def _lock_path(campaign_root: Path) -> Path:
    return campaign_root.resolve() / "flow" / "gpu_llm.lock"


def _locked_update(campaign_root: Path, updater: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    state_path = _state_path(campaign_root)
    lock_path = _lock_path(campaign_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            payload: dict[str, Any] = {}
            if state_path.is_file():
                try:
                    payload = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = {}
            payload = updater(payload)
            payload["updated_at"] = time.time()
            state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def read_llm_in_flight(campaign_root: Path) -> dict[str, Any] | None:
    state_path = _state_path(campaign_root)
    if not state_path.is_file():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    inflight = payload.get("in_flight")
    return dict(inflight) if isinstance(inflight, dict) else None


def gpu_llm_busy(campaign_root: Path) -> bool:
    return read_llm_in_flight(campaign_root) is not None


def begin_llm_request(
    campaign_root: Path,
    *,
    job_id: int,
    bench: str,
    worker: str,
) -> None:
    def _upd(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("in_flight"):
            raise RuntimeError(f"GPU LLM slot already held: {payload['in_flight']}")
        payload["in_flight"] = {
            "job_id": int(job_id),
            "bench": bench,
            "worker": worker,
            "started_at": time.time(),
        }
        return payload

    _locked_update(campaign_root, _upd)


def end_llm_request(campaign_root: Path, *, job_id: int) -> None:
    def _upd(payload: dict[str, Any]) -> dict[str, Any]:
        inflight = payload.get("in_flight")
        if isinstance(inflight, dict) and int(inflight.get("job_id", -1)) == int(job_id):
            payload["in_flight"] = None
        return payload

    _locked_update(campaign_root, _upd)


def llm_enter(**_fields: Any) -> None:
    """Hook entry point for c2hls (C2HLS_BATCH_LLM_HOOK_MODULE)."""
    root = _campaign_root_from_env()
    if root is None:
        return
    job_id = int(os.getenv("C2HLS_FIR_BATCH_JOB_ID", "0") or "0")
    bench = os.getenv("C2HLS_FIR_BATCH_BENCH", "")
    worker = os.getenv("C2HLS_FIR_BATCH_WORKER", "")
    if job_id <= 0:
        return
    try:
        begin_llm_request(root, job_id=job_id, bench=bench, worker=worker)
    except RuntimeError:
        pass


def llm_exit(**_fields: Any) -> None:
    root = _campaign_root_from_env()
    if root is None:
        return
    job_id = int(os.getenv("C2HLS_FIR_BATCH_JOB_ID", "0") or "0")
    if job_id <= 0:
        return
    end_llm_request(root, job_id=job_id)


def gpu_hold_reasons(queue, campaign_root: Path, campaign: dict[str, Any] | None = None) -> list[str]:
    blockers: list[str] = []
    pending = queue.pending_flash_count()
    if pending:
        blockers.append(f"pending_flash={pending}")
    inflight = read_llm_in_flight(campaign_root)
    if inflight:
        blockers.append(
            "llm_in_flight="
            f"{inflight.get('bench')} job={inflight.get('job_id')}"
        )
    if campaign is not None and str(campaign.get("gpu_mode") or "up") == "pending_unpark":
        blockers.append("pending_unpark")
    young = queue.young_claimed_count(grace_s=float((campaign or {}).get("config", {}).get("llm_burst_grace_s", 300)))
    if young:
        blockers.append(f"young_claimed={young}")
    return blockers


def gpu_must_stay_up(queue, campaign_root: Path, campaign: dict[str, Any] | None = None) -> bool:
    return bool(gpu_hold_reasons(queue, campaign_root, campaign))


def snapshot_gpu_busy(queue, campaign_root: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    return {
        "busy": gpu_must_stay_up(queue, campaign_root, campaign),
        "blockers": gpu_hold_reasons(queue, campaign_root, campaign),
        "pending_flash": queue.pending_flash_count(),
        "claimed_flash": queue.claimed_flash_count(),
        "llm_in_flight": read_llm_in_flight(campaign_root),
        "gpu_mode": str(campaign.get("gpu_mode") or "up"),
    }
