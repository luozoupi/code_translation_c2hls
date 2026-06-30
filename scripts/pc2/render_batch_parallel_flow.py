"""Render batch_parallel flow JSONL to markdown and HTML timelines."""

from __future__ import annotations

import argparse
import fcntl
import html
import json
import os
from collections import defaultdict
from pathlib import Path

from batch_parallel_config import campaign_paths


def _load_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def render_markdown(events: list[dict]) -> str:
    lines = ["# batch_parallel timeline", ""]
    for row in events:
        ts = row.get("ts", "")
        event = row.get("event", "")
        scope = row.get("scope", "")
        extra = {
            k: v
            for k, v in row.items()
            if k not in {"ts", "t", "event", "scope"} and v is not None
        }
        lines.append(f"- `{ts}` **{scope}/{event}** {extra}")
    lines.append("")
    return "\n".join(lines)


def render_html(events: list[dict]) -> str:
    lanes: dict[str, list[dict]] = defaultdict(list)
    for row in events:
        if row.get("scope") == "gpu":
            lanes["GPU"].append(row)
        elif row.get("role") in ("synth", "cosim"):
            key = f"{row.get('variant','?')}/{row.get('role')}/node_{row.get('node_index')}/slot_{row.get('worker_slot')}"
            lanes[key].append(row)
        else:
            lanes["campaign"].append(row)

    body_parts = []
    for lane, items in sorted(lanes.items()):
        body_parts.append(f"<h2>{html.escape(lane)}</h2><ul>")
        for row in items:
            label = html.escape(
                f"{row.get('ts','')} {row.get('event','')} "
                f"{row.get('bench','')} {row.get('phase','')}"
            )
            body_parts.append(f"<li>{label}</li>")
        body_parts.append("</ul>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>batch_parallel timeline</title>
<style>
body {{ font-family: sans-serif; margin: 1rem 2rem; }}
h2 {{ border-bottom: 1px solid #ccc; }}
</style></head><body>
<h1>batch_parallel timeline</h1>
{''.join(body_parts)}
</body></html>
"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def render_timeline_reports(campaign_root: Path, *, events_path: Path | None = None) -> bool:
    """Write flow/reports/timeline.md and timeline.html from events.jsonl."""
    paths = campaign_paths(campaign_root)
    events_file = events_path or paths["events"]
    events = _load_events(events_file)
    md_path = paths["reports"] / "timeline.md"
    html_path = paths["reports"] / "timeline.html"
    lock_path = paths["events"].parent / ".render.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_f:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        _atomic_write(md_path, render_markdown(events))
        _atomic_write(html_path, render_html(events))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Render batch_parallel flow logs")
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    paths = campaign_paths(args.campaign_root)
    if render_timeline_reports(args.campaign_root):
        print(f"wrote {paths['reports'] / 'timeline.md'}")
        print(f"wrote {paths['reports'] / 'timeline.html'}")
    else:
        print("render skipped (another render in progress)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
