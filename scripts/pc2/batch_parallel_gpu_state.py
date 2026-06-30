"""Central GPU / LLM busy ledger — single gate for park and scancel decisions."""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any, Callable

from batch_parallel_config import load_campaign

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
    variant: str,
    bench: str,
    phase: str,
    worker: str,
) -> None:
    def _upd(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("in_flight"):
            raise RuntimeError(f"GPU LLM slot already held: {payload['in_flight']}")
        payload["in_flight"] = {
            "job_id": int(job_id),
            "variant": variant,
            "bench": bench,
            "phase": phase,
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


def gpu_codegen_busy(queue, campaign_root: Path) -> tuple[bool, list[str]]:
    """
    True when the GPU must stay up for codegen: queued, claimed, or LLM HTTP in flight.
    """
    blockers: list[str] = []
    pending = queue.pending_codegen()
    claimed = queue.claimed_codegen()
    if pending:
        blockers.append(f"pending_codegen={pending}")
    if claimed:
        blockers.append(f"claimed_codegen={claimed}")
    inflight = read_llm_in_flight(campaign_root)
    if inflight:
        blockers.append(
            "llm_in_flight="
            f"{inflight.get('bench')}/{inflight.get('phase')} job={inflight.get('job_id')}"
        )
    return bool(blockers), blockers


def gpu_hold_reasons(queue, campaign_root: Path, campaign: dict[str, Any] | None = None) -> list[str]:
    """All reasons the GPU must not be cancelled right now."""
    busy, blockers = gpu_codegen_busy(queue, campaign_root)
    if busy:
        return blockers
    if campaign is not None and str(campaign.get("gpu_mode") or "up") == "pending_unpark":
        return ["pending_unpark"]
    return []


def gpu_must_stay_up(queue, campaign_root: Path, campaign: dict[str, Any] | None = None) -> bool:
    return bool(gpu_hold_reasons(queue, campaign_root, campaign))


def snapshot_gpu_busy(queue, campaign_root: Path) -> dict[str, Any]:
    busy, blockers = gpu_codegen_busy(queue, campaign_root)
    campaign = load_campaign(campaign_root)
    return {
        "busy": busy,
        "blockers": blockers,
        "pending_codegen": queue.pending_codegen(),
        "claimed_codegen": queue.claimed_codegen(),
        "llm_in_flight": read_llm_in_flight(campaign_root),
        "gpu_mode": str(campaign.get("gpu_mode") or "up"),
    }
