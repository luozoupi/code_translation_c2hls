#!/usr/bin/env python3
"""Discover healthy vLLM endpoints from other PC2 sessions/campaigns."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(os.environ.get("C2HLS_ROOT", Path(__file__).resolve().parents[2]))


def _pc2_root() -> Path:
    return _repo_root() / "artifacts" / "pc2"


def endpoint_url_healthy(url: str, *, timeout: float = 10.0) -> bool:
    base = str(url or "").strip().rstrip("/")
    if not base:
        return False
    for suffix in ("/models", "/health"):
        req = urllib.request.Request(f"{base}{suffix}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
    return False


def slurm_job_running(job_id: str | int | None) -> bool:
    if job_id in (None, "", "null", "None"):
        return False
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-t", "RUNNING,COMPLETING"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return bool(proc.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def load_endpoint_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data = dict(data)
    data["_path"] = str(path)
    return data


def discover_endpoint_files(pc2_root: Optional[Path] = None) -> list[Path]:
    root = pc2_root or _pc2_root()
    if not root.is_dir():
        return []
    patterns = (
        "sessions/*/llm_endpoint.json",
        "batch_parallel*/llm_endpoint.json",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(root.glob(pattern)))
    # De-dupe while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _endpoint_sort_key(item: dict[str, Any]) -> tuple:
    path = str(item.get("_path", ""))
    job_running = 1 if slurm_job_running(item.get("job_id")) else 0
    batch_bonus = 1 if "batch_parallel" in path else 0
    started = str(item.get("started_at") or "")
    return (job_running, batch_bonus, started)


def discover_healthy_endpoints(
    *,
    pc2_root: Optional[Path] = None,
    exclude_paths: Optional[set[str]] = None,
    require_job_running: bool = False,
) -> list[dict[str, Any]]:
    exclude = {str(p) for p in (exclude_paths or set())}
    healthy: list[dict[str, Any]] = []
    for path in discover_endpoint_files(pc2_root):
        resolved = str(path.resolve())
        if resolved in exclude:
            continue
        payload = load_endpoint_file(path)
        if not payload:
            continue
        url = str(payload.get("url", "")).strip()
        if not endpoint_url_healthy(url):
            continue
        if require_job_running and not slurm_job_running(payload.get("job_id")):
            continue
        healthy.append(payload)
    healthy.sort(key=_endpoint_sort_key, reverse=True)
    return healthy


def adopt_endpoint(
    *,
    target_endpoint_file: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Copy a discovered endpoint into this session and annotate borrow metadata."""
    payload = {
        "url": source.get("url"),
        "model": source.get("model", ""),
        "host": source.get("host", ""),
        "port": source.get("port"),
        "job_id": source.get("job_id"),
        "partition": source.get("partition", ""),
        "started_at": source.get("started_at"),
        "borrowed": True,
        "borrowed_from": source.get("_path"),
        "borrowed_at": datetime.now(timezone.utc).isoformat(),
    }
    target_endpoint_file.parent.mkdir(parents=True, exist_ok=True)
    target_endpoint_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def cmd_list(args: argparse.Namespace) -> int:
    endpoints = discover_healthy_endpoints(
        exclude_paths=set(args.exclude or []),
        require_job_running=args.require_job_running,
    )
    for ep in endpoints:
        print(
            f"{ep.get('url')}\tjob={ep.get('job_id')}\tfrom={ep.get('_path')}"
        )
    return 0


def cmd_adopt(args: argparse.Namespace) -> int:
    target = Path(args.target_endpoint_file)
    exclude = set(args.exclude or [])
    exclude.add(str(target.resolve()))
    endpoints = discover_healthy_endpoints(
        exclude_paths=exclude,
        require_job_running=args.require_job_running,
    )
    if not endpoints:
        return 1
    chosen = endpoints[0]
    adopt_endpoint(target_endpoint_file=target, source=chosen)
    print(json.dumps({
        "url": chosen.get("url"),
        "job_id": chosen.get("job_id"),
        "borrowed_from": chosen.get("_path"),
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover borrowable PC2 vLLM endpoints")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List healthy endpoints")
    p_list.add_argument("--exclude", action="append", default=[])
    p_list.add_argument("--require-job-running", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_adopt = sub.add_parser("adopt", help="Adopt best endpoint into target file")
    p_adopt.add_argument("target_endpoint_file")
    p_adopt.add_argument("--exclude", action="append", default=[])
    p_adopt.add_argument("--require-job-running", action="store_true")
    p_adopt.set_defaults(func=cmd_adopt)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
