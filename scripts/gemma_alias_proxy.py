#!/usr/bin/env python3
"""Proxy one OpenAI-compatible model alias and disable reasoning output."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class ProxyHandler(BaseHTTPRequestHandler):
    upstream: str
    alias: str
    upstream_model: str
    timeout: float

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, method: str, body: bytes | None = None) -> None:
        url = f"{self.upstream}{self.path}"
        headers = {"Content-Type": "application/json"}
        authorization = self.headers.get("Authorization")
        if authorization:
            headers["Authorization"] = authorization
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                content_type = response.headers.get(
                    "Content-Type", "application/json"
                )
                self._send(response.status, payload, content_type)
        except urllib.error.HTTPError as exc:
            self._send(
                exc.code,
                exc.read(),
                exc.headers.get("Content-Type", "application/json"),
            )
        except (OSError, urllib.error.URLError) as exc:
            payload = json.dumps(
                {"error": {"message": f"upstream connection failed: {exc}"}}
            ).encode("utf-8")
            self._send(502, payload, "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            payload = {
                "object": "list",
                "data": [
                    {
                        "id": self.alias,
                        "object": "model",
                        "owned_by": "gemma-alias-proxy",
                        "root": self.upstream_model,
                        "max_model_len": 262144,
                    }
                ],
            }
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
            return
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send(400, b'{"error":{"message":"invalid JSON"}}', "application/json")
            return
        if not isinstance(payload, dict):
            self._send(
                400,
                b'{"error":{"message":"JSON body must be an object"}}',
                "application/json",
            )
            return
        if self.path.rstrip("/") == "/v1/chat/completions":
            payload["model"] = self.upstream_model
            template_kwargs = payload.setdefault("chat_template_kwargs", {})
            if not isinstance(template_kwargs, dict):
                template_kwargs = {}
                payload["chat_template_kwargs"] = template_kwargs
            template_kwargs["enable_thinking"] = False
        self._forward(
            "POST",
            (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30113)
    parser.add_argument(
        "--upstream",
        default="http://cs-u-converge.cs.umn.edu:30001",
    )
    parser.add_argument("--alias", default="gemma-4-31b-it")
    parser.add_argument("--upstream-model", default="google/gemma-4-31B-it")
    parser.add_argument("--timeout", type=float, default=1900.0)
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file or None,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ProxyHandler.upstream = args.upstream.rstrip("/")
    ProxyHandler.alias = args.alias
    ProxyHandler.upstream_model = args.upstream_model
    ProxyHandler.timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    logging.info(
        "proxy start host=%s port=%s alias=%s upstream_model=%s upstream=%s",
        args.host,
        args.port,
        args.alias,
        args.upstream_model,
        args.upstream,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
