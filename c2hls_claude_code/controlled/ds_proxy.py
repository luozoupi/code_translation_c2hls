#!/usr/bin/env python3
"""Minimal OpenAI-compatible forwarding proxy for the DeepSeek API, for the controlled DeepSeek-vs-Claude rerun.

Compute nodes have no internet; this runs on a login node and forwards POST /v1/chat/completions to
api.deepseek.com. Differences from the ChatHLS queue proxy used before:
  * the API key is read from the environment inside this process (DEEPSEEK_API_KEY, else DeepSeek_API),
    never from argv, and is never logged;
  * upstream timeout is configurable (default 3600 s, was 600 s);
  * every call is logged as one JSON line (status, elapsed, model, max_tokens, finish_reason, usage) to
    --log; request/response bodies are not logged here (the c2hls-side controlled layer logs prompt hashes).
No retries here: retries are done by the client-side controlled layer, identically for both models.

  ds_proxy.py --port 18094 --workers 8 --log controlled/ds_proxy_calls.jsonl
"""
import argparse, json, os, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://api.deepseek.com/v1/chat/completions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18094)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DeepSeek_API") or ""
    if not key:
        raise SystemExit("set DEEPSEEK_API_KEY (or DeepSeek_API) in the environment")
    sem = threading.Semaphore(max(1, args.workers))
    log_lock = threading.Lock()
    stats = {"calls": 0, "errors": 0, "active": 0}

    def log(rec: dict) -> None:
        with log_lock, open(args.log, "a") as fp:
            fp.write(json.dumps(rec) + "\n")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet stderr
            pass

        def _send(self, code: int, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("health"):
                return self._send(200, json.dumps({"ok": True, **stats}).encode())
            self._send(404, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                req = json.loads(raw)
            except ValueError:
                return self._send(400, b'{"error":"bad json"}')
            t0 = time.time()
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": req.get("model"),
                   "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens"),
                   "extra": {k: req[k] for k in ("thinking", "reasoning_effort") if k in req}}
            with sem:
                stats["active"] += 1
                try:
                    up = urllib.request.Request(UPSTREAM, data=raw, method="POST", headers={
                        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                    with urllib.request.urlopen(up, timeout=args.timeout) as r:
                        body, code = r.read(), r.status
                except urllib.error.HTTPError as e:
                    body, code = e.read(), e.code
                except Exception as e:  # noqa: BLE001  (timeouts, resets)
                    body, code = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), 502
                finally:
                    stats["active"] -= 1
            stats["calls"] += 1
            rec.update(status=code, elapsed_s=round(time.time() - t0, 1))
            try:
                doc = json.loads(body)
                ch = (doc.get("choices") or [{}])[0]
                rec["finish_reason"] = ch.get("finish_reason")
                msg = ch.get("message") or {}
                rec["content_len"] = len(msg.get("content") or "")
                rec["reasoning_len"] = len(msg.get("reasoning_content") or "")
                rec["usage"] = doc.get("usage")
                if code != 200:
                    rec["error"] = str(doc.get("error"))[:300]
            except ValueError:
                rec["error"] = body[:300].decode("utf-8", "replace")
            if code != 200:
                stats["errors"] += 1
            log(rec)
            self._send(code, body)

    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"[ds_proxy] listening http://{args.host}:{args.port}/v1 workers={args.workers} timeout={args.timeout}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
