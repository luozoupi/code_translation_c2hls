"""JSONL flow logger for Fir batch_parallel campaigns."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from batch_parallel.config import campaign_paths, load_campaign


class FirBatchParallelFlow:
    def __init__(self, campaign_root: Path) -> None:
        self.campaign_root = campaign_root.resolve()
        self.paths = campaign_paths(self.campaign_root)
        self.paths["events"].parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        doc = load_campaign(self.campaign_root) if self.paths["campaign"].is_file() else {}
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "t": time.time(),
            "event": event,
            "gpu_mode": doc.get("gpu_mode", "unknown"),
            **fields,
        }
        line = json.dumps(row, separators=(",", ":")) + "\n"
        self.paths["events"].open("a", encoding="utf-8").write(line)

    def write_status(self, payload: dict[str, Any]) -> None:
        self.paths["status"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def write_node_map(self, payload: dict[str, Any]) -> None:
        self.paths["node_map"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
