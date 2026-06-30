"""JSONL flow logger for batch_parallel campaigns."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from batch_parallel_config import campaign_paths, load_campaign

log = logging.getLogger(__name__)

_RENDER_DEBOUNCE_SEC = float(os.environ.get("BATCH_PARALLEL_RENDER_DEBOUNCE_SEC", "5"))


class BatchParallelFlow:
    def __init__(self, campaign_root: Path) -> None:
        self.campaign_root = campaign_root.resolve()
        self.paths = campaign_paths(self.campaign_root)
        self.paths["events"].parent.mkdir(parents=True, exist_ok=True)
        (self.paths["events"].parent / "by_scope").mkdir(parents=True, exist_ok=True)
        (self.paths["events"].parent / "by_scope" / "variants").mkdir(parents=True, exist_ok=True)
        self.paths["status"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["reports"].mkdir(parents=True, exist_ok=True)
        self._last_render_at = 0.0

    def _gpu_mode(self) -> str:
        doc = load_campaign(self.campaign_root)
        return str(doc.get("gpu_mode") or "unknown")

    def _render_reports(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_render_at < _RENDER_DEBOUNCE_SEC:
            return
        try:
            from render_batch_parallel_flow import render_timeline_reports

            if render_timeline_reports(self.campaign_root):
                self._last_render_at = now
        except Exception:
            log.exception("failed to render batch_parallel timeline reports")

    def emit(self, event: str, *, scope: str = "campaign", **fields: Any) -> None:
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "t": time.time(),
            "scope": scope,
            "event": event,
            "gpu_mode": self._gpu_mode(),
            **fields,
        }
        line = json.dumps(row, separators=(",", ":")) + "\n"
        self.paths["events"].open("a", encoding="utf-8").write(line)
        if scope == "gpu":
            self.paths["gpu_events"].open("a", encoding="utf-8").write(line)
        elif scope == "variant" and fields.get("variant"):
            variant_path = (
                self.paths["events"].parent / "by_scope" / "variants" / f"{fields['variant']}.jsonl"
            )
            variant_path.open("a", encoding="utf-8").write(line)
        self._render_reports(force=event in {"campaign_complete"})

    def write_status(self, payload: dict[str, Any]) -> None:
        self.paths["status"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def write_node_map(self, payload: dict[str, Any]) -> None:
        self.paths["node_map"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
