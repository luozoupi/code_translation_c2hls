"""Discover and snapshot live batch_parallel campaigns for the explorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

ACTIVE_STATUSES = frozenset({"running", "completing"})
INACTIVE_CAMPAIGN_STATUSES = frozenset({"aborted", "complete", "stopped"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _queue_summary(queue_db: Path) -> dict[str, int]:
    import sqlite3

    summary = {"total": 0, "done": 0, "claimed": 0, "pending": 0, "failed": 0}
    if not queue_db.is_file():
        return summary
    conn = sqlite3.connect(queue_db)
    try:
        rows = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        for status, count in rows:
            st = str(status or "pending")
            summary["total"] += int(count)
            if st in summary:
                summary[st] += int(count)
    finally:
        conn.close()
    return summary


def _is_batch_parallel_campaign(path: Path) -> bool:
    return path.is_dir() and (path / "campaign.json").is_file() and (path / "queue.db").is_file()


def discover_live_campaigns(repo_root: Path | None = None) -> list[dict[str, Any]]:
    root = repo_root or REPO
    campaigns: list[dict[str, Any]] = []
    for site in ("fir", "pc2"):
        site_root = root / "artifacts" / site
        if not site_root.is_dir():
            continue
        for child in sorted(site_root.iterdir()):
            if not _is_batch_parallel_campaign(child):
                continue
            campaign = _read_json(child / "campaign.json")
            summary = _queue_summary(child / "queue.db")
            status = str(campaign.get("campaign_status") or "unknown")
            claimed = int(summary.get("claimed", 0))
            if status in INACTIVE_CAMPAIGN_STATUSES:
                active = False
            else:
                # Active = real in-flight work (claimed benches), not stale queue rows.
                active = claimed > 0
            campaigns.append({
                "id": f"{site}/{child.name}",
                "site": site,
                "label": child.name,
                "path": str(child.resolve()),
                "campaign_status": status,
                "compute_state": campaign.get("compute_state"),
                "gpu_mode": campaign.get("gpu_mode"),
                "active": active,
                "live_supported": site == "fir",
                "summary": summary,
                "mtime": (child / "campaign.json").stat().st_mtime,
            })
    campaigns.sort(key=lambda c: (not c["active"], -c["mtime"]))
    return campaigns


def build_live_snapshot(campaign_path: Path) -> dict[str, Any]:
    """Build Fir batch_parallel dashboard snapshot for a campaign root."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts" / "fir"))
    from batch_parallel.dashboard import build_snapshot

    return build_snapshot(campaign_path.resolve())


def resolve_campaign_path(repo_root: Path, campaign_id: str) -> Path | None:
    if "/" not in campaign_id:
        return None
    site, dirname = campaign_id.split("/", 1)
    path = (repo_root / "artifacts" / site / dirname).resolve()
    if not _is_batch_parallel_campaign(path):
        return None
    return path
