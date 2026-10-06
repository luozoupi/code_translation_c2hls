"""Campaign helper lifecycle: discharge watch/drain/coord when the run is done."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

TERMINAL_CAMPAIGN_STATUSES = frozenset(
    {"complete", "completed", "failed", "aborted"}
)


def campaign_status_is_terminal(status: Any) -> bool:
    return str(status or "").strip().lower() in TERMINAL_CAMPAIGN_STATUSES


def campaign_is_terminal(campaign: Optional[dict[str, Any]]) -> bool:
    return campaign_status_is_terminal((campaign or {}).get("campaign_status"))


def helper_job_ids_to_discharge(
    campaign: Optional[dict[str, Any]],
    *,
    self_job_id: Optional[str] = None,
) -> list[str]:
    helpers = (campaign or {}).get("helper_jobs") or {}
    if not isinstance(helpers, dict):
        return []
    skip = {str(self_job_id)} if self_job_id else set()
    out: list[str] = []
    seen: set[str] = set()
    for jid in helpers.values():
        if not jid or str(jid) in {"None", "null"}:
            continue
        token = str(jid)
        if token in skip or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def campaign_root_is_terminal(campaign_root: Path) -> bool:
    root = Path(campaign_root)
    if (root / "CAMPAIGN_COMPLETE").is_file():
        return True
    camp_path = root / "campaign.json"
    if not camp_path.is_file():
        return False
    import json

    try:
        doc = json.loads(camp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return campaign_is_terminal(doc if isinstance(doc, dict) else {})


def discharge_helper_jobs(
    campaign: Optional[dict[str, Any]],
    *,
    self_job_id: Optional[str] = None,
    scancel: Optional[Any] = None,
) -> list[str]:
    """Cancel leftover helper Slurm jobs. Returns the ids passed to scancel."""
    cancel = scancel or _default_scancel
    discharged: list[str] = []
    for jid in helper_job_ids_to_discharge(campaign, self_job_id=self_job_id):
        cancel(jid)
        discharged.append(jid)
    return discharged


def _default_scancel(job_id: str) -> None:
    import subprocess

    subprocess.run(["scancel", str(job_id)], check=False)
