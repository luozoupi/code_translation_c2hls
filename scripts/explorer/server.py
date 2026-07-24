#!/usr/bin/env python3
"""Live HTTP server for the experiment explorer."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "explorer"))

from catalog import compare_experiments
from index_cache import get_index, prewarm_index_cache
from live import build_live_snapshot, discover_live_campaigns, resolve_campaign_path

HTML_PATH = Path(__file__).with_name("experiments_explorer.html")


class ExplorerHandler(BaseHTTPRequestHandler):
    repo_root: Path = REPO
    registry_path: Path = REPO / "experiments_registry.json"
    html_bytes: bytes = b""
    cache_sec: float = 30.0

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[explorer] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, self.html_bytes, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/index":
            qs = urllib.parse.parse_qs(parsed.query)
            force_refresh = qs.get("refresh", ["0"])[0] in ("1", "true", "yes")
            index, cache_meta = get_index(
                self.repo_root,
                self.registry_path,
                cache_sec=self.cache_sec,
                force_refresh=force_refresh,
            )
            payload = json.dumps(index, indent=2).encode("utf-8")
            self._send(
                200,
                payload,
                "application/json; charset=utf-8",
                extra_headers={"X-Explorer-Cache": str(cache_meta.get("source") or "unknown")},
            )
            return

        if parsed.path == "/api/index/refresh":
            _, cache_meta = get_index(
                self.repo_root,
                self.registry_path,
                cache_sec=0.0,
                force_refresh=True,
            )
            payload = json.dumps(cache_meta, indent=2).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return

        if parsed.path == "/api/compare":
            qs = urllib.parse.parse_qs(parsed.query)
            ids = []
            for raw in qs.get("ids", []):
                ids.extend(part.strip() for part in raw.split(",") if part.strip())
            benches = []
            for raw in qs.get("benches", []):
                if raw.strip():
                    benches.extend(part.strip() for part in raw.split(",") if part.strip())
            include_planned = qs.get("include_planned", ["0"])[0] in ("1", "true", "yes")
            index, _cache_meta = get_index(
                self.repo_root,
                self.registry_path,
                cache_sec=self.cache_sec,
            )
            result = compare_experiments(
                index.get("experiments") or [],
                ids=ids,
                bench_filter=benches or None,
                include_planned=include_planned,
            )
            payload = json.dumps(result, indent=2).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return

        if parsed.path == "/api/live/campaigns":
            payload = json.dumps(
                {"campaigns": discover_live_campaigns(self.repo_root)},
                indent=2,
            ).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return

        if parsed.path == "/api/live/snapshot":
            qs = urllib.parse.parse_qs(parsed.query)
            campaign_id = (qs.get("id") or [""])[0].strip()
            if not campaign_id:
                self._send(400, b"missing id\n", "text/plain; charset=utf-8")
                return
            path = resolve_campaign_path(self.repo_root, campaign_id)
            if path is None:
                self._send(404, b"campaign not found\n", "text/plain; charset=utf-8")
                return
            site = campaign_id.split("/", 1)[0]
            if site != "fir":
                body = json.dumps({
                    "error": "live snapshot supported for Fir batch_parallel only (v1)",
                    "campaign_id": campaign_id,
                }).encode("utf-8")
                self._send(501, body, "application/json; charset=utf-8")
                return
            try:
                snapshot = build_live_snapshot(path)
            except Exception as exc:
                body = json.dumps({"error": str(exc), "campaign_id": campaign_id}).encode("utf-8")
                self._send(500, body, "application/json; charset=utf-8")
                return
            payload = json.dumps(snapshot, indent=2).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return

        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment explorer server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--registry", type=Path, default=REPO / "experiments_registry.json")
    parser.add_argument("--cache-sec", type=float, default=300.0)
    parser.add_argument("--prewarm", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    registry_path = args.registry.resolve()

    if args.prewarm:
        meta = prewarm_index_cache(repo_root, registry_path)
        print(
            f"index cache: {meta.get('source')} "
            f"experiments={meta.get('experiment_count')} "
            f"fp={str(meta.get('fingerprint', ''))[:12]}"
        )

    html_bytes = HTML_PATH.read_bytes()
    handler = ExplorerHandler
    handler.repo_root = repo_root
    handler.registry_path = registry_path
    handler.html_bytes = html_bytes
    handler.cache_sec = args.cache_sec

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"explorer: http://{args.host}:{args.port}/")
    print(f"repo: {handler.repo_root}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
