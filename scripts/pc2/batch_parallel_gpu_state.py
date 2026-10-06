"""Central GPU / LLM busy ledger — single gate for park and scancel decisions."""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any, Callable

from batch_parallel_config import load_campaign

# Only failures that happen before api.deepseek.com accepts the request.
# A timeout, reset, or closed socket means the completion was already billed.
# Requeueing those starts another billed call for the same kernel.
PRE_UPSTREAM_LLM_MARKERS = (
    "connection refused",
    "connection error",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
)

# One extra try when the login proxy is not up yet. A second failure stops.
MAX_CODEGEN_CONNECTION_TRIES = 2

STALE_LLM_SLOT_S = 7200.0


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


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(str(cur))
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return " ".join(parts).lower()


def is_retriable_llm_error(exc: BaseException) -> bool:
    msg = _exception_text(exc)
    if "timed out" in msg or "timeout" in msg:
        return False
    return any(marker in msg for marker in PRE_UPSTREAM_LLM_MARKERS)


def codegen_should_requeue(campaign_root: Path, job_id: int, exc: BaseException) -> bool:
    """Allow one pre-upstream retry. Never retry a call that already reached DeepSeek."""
    if not is_retriable_llm_error(exc):
        return False
    path = campaign_root.resolve() / "flow" / "llm_codegen_attempts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                doc = loaded
        except json.JSONDecodeError:
            doc = {}
    key = str(int(job_id))
    tries = int(doc.get(key) or 0) + 1
    doc[key] = tries
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return tries < MAX_CODEGEN_CONNECTION_TRIES


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
        inflight = payload.get("in_flight")
        if isinstance(inflight, dict) and inflight:
            if int(inflight.get("job_id", -1)) == int(job_id):
                inflight["worker"] = worker
                inflight["started_at"] = time.time()
                payload["in_flight"] = inflight
                return payload
            started = float(inflight.get("started_at") or 0.0)
            age_s = time.time() - started
            if age_s < STALE_LLM_SLOT_S:
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
