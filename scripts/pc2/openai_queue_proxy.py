#!/usr/bin/env python3
"""Queued OpenAI-compatible proxy to api.openai.com (login node with internet).

Compute nodes have no outbound internet; they hit this proxy on the login host.
Forwards /v1/chat/completions and /v1/models to the hosted OpenAI API.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_UPSTREAM = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-luna"


class QueueState:
    def __init__(self, upstream_base: str, api_key: str, max_workers: int = 1) -> None:
        self.upstream_base = upstream_base.rstrip("/")
        self.api_key = api_key
        self.jobs: queue.Queue[tuple[dict[str, Any], threading.Event, dict[str, Any]]] = queue.Queue()
        self.lock = threading.Lock()
        self.active = 0
        self.max_workers = max_workers
        for _ in range(max_workers):
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        while True:
            payload, done, holder = self.jobs.get()
            with self.lock:
                self.active += 1
            try:
                holder["response"] = self._forward(payload)
                holder["status"] = 200
            except urllib.error.HTTPError as exc:
                holder["status"] = exc.code
                holder["response"] = exc.read().decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                holder["status"] = 502
                holder["response"] = json.dumps({"error": str(exc)})
            finally:
                with self.lock:
                    self.active -= 1
                done.set()
                self.jobs.task_done()

    def _forward(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.upstream_base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return resp.read().decode("utf-8")

    def enqueue(self, payload: dict[str, Any]) -> tuple[int, str]:
        done = threading.Event()
        holder: dict[str, Any] = {}
        self.jobs.put((payload, done, holder))
        done.wait()
        return int(holder.get("status", 500)), str(holder.get("response", ""))

    def stats(self) -> dict[str, Any]:
        return {
            "queue_depth": self.jobs.qsize(),
            "active": self.active,
            "max_workers": self.max_workers,
            "upstream": self.upstream_base,
        }


def make_handler(state: QueueState, *, model_id: str, default_effort: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[openai-proxy] {self.address_string()} {fmt % args}", flush=True)

        def _json(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in {"/health", "/v1/health"}:
                self._json(200, {"ok": True, "model": model_id, **state.stats()})
                return
            if self.path in {"/models", "/v1/models"}:
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model"}],
                    },
                )
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in {"/chat/completions", "/v1/chat/completions"}:
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid json"})
                return
            if isinstance(payload, dict):
                if model_id:
                    payload["model"] = model_id
                if default_effort and "reasoning_effort" not in payload:
                    payload["reasoning_effort"] = default_effort
            code, response = state.enqueue(payload)
            data = response.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _load_api_key(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    for name in ("OPENAI_API_KEY", "OPEN_AI_API", "OPENAI_API"):
        val = (os.environ.get(name) or "").strip()
        if val and val.lower() != "empty":
            return val
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18200)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--session-dir", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--reasoning-effort", default="")
    args = parser.parse_args()

    api_key = _load_api_key(args.api_key)
    if not api_key:
        raise SystemExit("missing OpenAI API key (OPENAI_API_KEY / OPEN_AI_API)")
    if len(api_key) < 20:
        raise SystemExit(f"OpenAI API key looks too short (len={len(api_key)})")

    model_id = (
        (args.model or "").strip()
        or (os.environ.get("OPENAI_PROXY_MODEL") or "").strip()
        or (os.environ.get("C2HLS_MODEL") or "").strip()
        or DEFAULT_MODEL
    )
    default_effort = (
        (args.reasoning_effort or "").strip()
        or (os.environ.get("C2HLS_REASONING_EFFORT") or "").strip()
    )

    workers = max(1, int(args.workers))
    state = QueueState(args.upstream, api_key, max_workers=workers)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, model_id=model_id, default_effort=default_effort),
    )
    print(
        f"[openai-proxy] listening http://{args.host}:{args.port} "
        f"queue_workers={workers} model={model_id} effort={default_effort or '-'}",
        flush=True,
    )

    if args.session_dir:
        session = Path(args.session_dir)
        session.mkdir(parents=True, exist_ok=True)
        host = args.host if args.host not in {"0.0.0.0", ""} else socket.gethostname()
        endpoint = {
            "url": f"http://{host}:{args.port}/v1",
            "host": host,
            "port": args.port,
            "model": model_id,
            "provider": "openai",
            "queued": True,
            "workers": workers,
            "reasoning_effort": default_effort or None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (session / "openai_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
        (session / "llm_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\n")
        print(f"[openai-proxy] wrote {session / 'llm_endpoint.json'}", flush=True)

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
