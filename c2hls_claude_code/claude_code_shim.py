#!/usr/bin/env python3
"""OpenAI-compatible /v1/chat/completions server that answers with Claude Code (`claude -p`).

c2hls sends every LLM call through an OpenAI client when OPENAI_BASE_URL is set, so pointing
OPENAI_BASE_URL at this server runs the unchanged flash / DSE flow (its prompts, skills, repair
loops, gates and artifacts) with Claude Code as the model.

Each request runs one headless, stateless `claude -p`:
  * system messages -> --system-prompt-file (replaces Claude Code's own system prompt)
  * the remaining messages -> stdin (a single user message as is; a multi-turn history is
    rendered as a transcript that ends with the turn to answer)
  * --tools "" (no tools), --disable-slash-commands (no skills), --strict-mcp-config (no MCP),
    --setting-sources "" (no user/project/local settings), --no-session-persistence,
    cwd = an empty per-call directory (no CLAUDE.md, no project memory)
  * max_tokens -> CLAUDE_CODE_MAX_OUTPUT_TOKENS; model -> --model (non-Claude names map to
    --default-model)

Every call is logged under --log-dir: calls.jsonl (one line per call) and call_<n>/ with the
system prompt, prompt, reply and the CLI's JSON result.

Run on a login node (internet + Claude Code login); compute nodes reach http://<host>:<port>/v1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ARGS: argparse.Namespace
SEM: threading.Semaphore
COUNTER_LOCK = threading.Lock()
COUNTER = 0
STATS = {"calls": 0, "errors": 0, "busy": 0}


def _text(content) -> str:
    """OpenAI message content can be a string or a list of {type: text} parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def render(messages: list[dict]) -> tuple[str, str]:
    system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    turns = [m for m in messages if m.get("role") not in ("system", "developer")]
    if len(turns) == 1 and turns[0].get("role") == "user":
        return system, _text(turns[0].get("content"))
    parts = ["The conversation so far is below. Reply as the assistant to the final user turn, "
             "exactly as you would in a normal chat. Do not mention this transcript format.\n"]
    for m in turns:
        role = "Assistant" if m.get("role") == "assistant" else "User"
        parts.append(f"<<<{role}>>>\n{_text(m.get('content'))}\n<<<end {role}>>>\n")
    if turns and turns[-1].get("role") == "assistant":
        parts.append("Continue the assistant reply above from exactly where it stops.\n")
    return system, "\n".join(parts)


def run_claude(model: str, system: str, prompt: str, max_tokens: int | None, call_dir: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix="cc_shim_", dir=ARGS.tmp_dir))
    try:
        sys_file = call_dir / "system.txt"
        sys_file.write_text(system or "You are a helpful assistant.")
        cmd = [ARGS.claude_bin, "-p", "--output-format", "json", "--model", model,
               "--system-prompt-file", str(sys_file), "--tools", "", "--disable-slash-commands",
               "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence"]
        env = dict(os.environ)
        cap = ARGS.max_output_tokens
        if max_tokens:
            cap = min(int(max_tokens), cap)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(cap)
        t0 = time.time()
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=work, env=env,
                              timeout=ARGS.call_timeout)
        elapsed = time.time() - t0
        (call_dir / "cli_stdout.json").write_text(proc.stdout)
        if proc.stderr:
            (call_dir / "cli_stderr.txt").write_text(proc.stderr)
        try:
            doc = json.loads(proc.stdout)
        except json.JSONDecodeError:
            doc = {"is_error": True, "result": "", "error": f"exit {proc.returncode}: {proc.stderr[-2000:]}"}
        doc["_elapsed_s"] = round(elapsed, 1)
        doc["_returncode"] = proc.returncode
        return doc
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- agent mode
AGENT_DIR = Path(__file__).resolve().parent / "agent"
FENCE_RE = re.compile(r"```(?:cpp|c\+\+|c|hls)?[^\n]*\n(.*?)```", re.S)

AGENT_APPENDIX = """

# Working environment (agent mode)

You are running with tools in a workspace directory that holds:
- `kernel.cpp`: the current HLS code from the request (your starting point; edit it in place)
- `{header}`: the header (do not change it)
- `testbench.cpp`: the testbench the flow checks correctness with (do not change it)
- `./hls_check.sh compile|csim|csynth`: checks `kernel.cpp` with Vitis HLS on a compute node
  (part {part}, clock {clock} ns, top `{top}`) and prints a summary. It blocks until done;
  csynth of a large design can take a long time. You have {max_checks} checks in total.

You may read and edit files in the workspace and run `./hls_check.sh`. No other commands are available.
Use the checks to verify that your design compiles, passes csim and synthesizes, and to compare
alternatives. Keep `kernel.cpp` holding your best verified design.

When you are done, answer the request exactly as it asks: your final reply must contain the complete
final kernel (the content of `kernel.cpp`) in one ```cpp fence, as the request's output format requires.
"""


def agent_match(prompt: str) -> bool:
    return bool(ARGS.agent_match) and re.search(ARGS.agent_match, prompt, re.M) is not None


def _current_kernel(prompt: str) -> str:
    """The request's current HLS code: the last fenced block that defines the top function."""
    top = re.escape(ARGS.agent_top)
    blocks = [b for b in FENCE_RE.findall(prompt) if re.search(rf"\b{top}\s*\([^;]*\)\s*\{{", b)]
    return blocks[-1] if blocks else ""


def run_agent(model: str, system: str, prompt: str, call_dir: Path) -> dict:
    ws = call_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    bench = Path(ARGS.agent_bench_dir)
    (ws / ARGS.agent_header).write_text((bench / ARGS.agent_header).read_text())
    (ws / "testbench.cpp").write_text((bench / "testbench.cpp").read_text())
    kernel = _current_kernel(prompt)
    (ws / "kernel.cpp").write_text(kernel or (bench / "plain.cpp").read_text())
    check = (AGENT_DIR / "hls_check.sh.template").read_text()
    for k, v in {"__MAX_CHECKS__": ARGS.agent_max_checks, "__CSIM_TIME__": ARGS.agent_csim_time,
                 "__CSYNTH_TIME__": ARGS.agent_csynth_time, "__ACCOUNT__": ARGS.agent_account,
                 "__PART__": ARGS.agent_part, "__CLOCK__": ARGS.agent_clock_ns, "__TOP__": ARGS.agent_top, "__HEADER__": ARGS.agent_header,
                 "__CHECK_PY__": AGENT_DIR / "hls_check.py"}.items():
        check = check.replace(k, str(v))
    (ws / "hls_check.sh").write_text(check)
    (ws / "hls_check.sh").chmod(0o750)
    sys_file = call_dir / "system.txt"
    sys_file.write_text((system or "") + AGENT_APPENDIX.format(part=ARGS.agent_part, clock=ARGS.agent_clock_ns,
                                                                top=ARGS.agent_top, max_checks=ARGS.agent_max_checks,
                                                                header=ARGS.agent_header))
    cmd = [ARGS.claude_bin, "-p", "--output-format", "stream-json", "--verbose", "--model", model,
           "--system-prompt-file", str(sys_file), "--restricted",
           "--tools", "Read,Write,Edit,Glob,Grep,Bash", "--allowedTools", "Read,Write,Edit,Glob,Grep,Bash(./hls_check.sh:*)",
           "--permission-mode", "acceptEdits", "--disable-slash-commands", "--strict-mcp-config",
           "--no-session-persistence"]
    env = dict(os.environ)
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(ARGS.max_output_tokens)
    # A csynth check blocks for a long time; let the Bash tool wait for it.
    env["BASH_DEFAULT_TIMEOUT_MS"] = env["BASH_MAX_TIMEOUT_MS"] = str(int(ARGS.agent_timeout * 1000))
    t0 = time.time()
    with open(call_dir / "agent_transcript.jsonl", "w") as out, open(call_dir / "cli_stderr.txt", "w") as err:
        try:
            proc = subprocess.run(cmd, input=prompt, stdout=out, stderr=err, text=True, cwd=ws, env=env,
                                  timeout=ARGS.agent_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    doc = {"is_error": True, "result": "", "error": f"agent produced no result (exit {rc})"}
    tool_calls = []
    for line in (call_dir / "agent_transcript.jsonl").read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            doc = ev
        elif ev.get("type") == "assistant":
            for part in (ev.get("message") or {}).get("content") or []:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_calls.append({"tool": part.get("name"), "input": json.dumps(part.get("input"))[:200]})
    text = str(doc.get("result") or "")
    final_kernel = (ws / "kernel.cpp").read_text()
    if not FENCE_RE.search(text) and final_kernel.strip():
        # Keep c2hls's output contract: the reply carries the kernel in one ```cpp fence.
        text = (text + "\n\n" if text else "") + f"```cpp\n{final_kernel.rstrip()}\n```\n"
        doc["_fence_from_workspace"] = True
    doc["result"] = text
    doc["_elapsed_s"] = round(time.time() - t0, 1)
    doc["_agent"] = {"tool_calls": len(tool_calls), "checks_used": int((ws / ".checks_used").read_text())
                     if (ws / ".checks_used").is_file() else 0, "started_from_request_kernel": bool(kernel)}
    (call_dir / "agent_tool_calls.json").write_text(json.dumps(tool_calls, indent=1))
    return doc


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": ARGS.default_model, "object": "model", "owned_by": "claude-code"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._send(200, {"ok": True, **STATS})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        global COUNTER
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "only /v1/chat/completions is served"}})
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if req.get("stream"):
            self._send(400, {"error": {"message": "stream=true is not supported by this shim"}})
            return
        model = str(req.get("model") or ARGS.default_model)
        if not model.startswith("claude"):
            model = ARGS.default_model
        with COUNTER_LOCK:
            COUNTER += 1
            n = COUNTER
        call_dir = Path(ARGS.log_dir) / f"call_{n:05d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        system, prompt = render(req.get("messages") or [])
        (call_dir / "request.json").write_text(json.dumps(req, indent=1))
        (call_dir / "prompt.txt").write_text(prompt)
        if not SEM.acquire(timeout=ARGS.queue_timeout):
            STATS["busy"] += 1
            self._send(429, {"error": {"message": "claude code shim busy"}})
            return
        agent = agent_match(prompt)
        try:
            if agent:
                doc = run_agent(model, system, prompt, call_dir)
            else:
                doc = run_claude(model, system, prompt, req.get("max_tokens") or req.get("max_completion_tokens"), call_dir)
        except subprocess.TimeoutExpired:
            doc = {"is_error": True, "result": "", "error": f"claude -p timed out after {ARGS.call_timeout}s"}
        finally:
            SEM.release()
        STATS["calls"] += 1
        text = str(doc.get("result") or "")
        usage = doc.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0) \
            + int(usage.get("cache_creation_input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        stop = str(doc.get("stop_reason") or "")
        finish = "length" if stop in ("max_tokens", "length") else "stop"
        (call_dir / "reply.txt").write_text(text)
        with open(Path(ARGS.log_dir) / "calls.jsonl", "a") as fh:
            fh.write(json.dumps({"n": n, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                                 "system_chars": len(system), "prompt_chars": len(prompt), "reply_chars": len(text),
                                 "max_tokens": req.get("max_tokens"), "in_tokens": in_tok, "out_tokens": out_tok,
                                 "finish": finish, "elapsed_s": doc.get("_elapsed_s"),
                                 "cost_usd": doc.get("total_cost_usd"), "is_error": bool(doc.get("is_error")),
                                 "mode": "agent" if agent else "plain", "agent": doc.get("_agent"),
                                 "fence_from_workspace": bool(doc.get("_fence_from_workspace")),
                                 "error": doc.get("error") or ""}) + "\n")
        if doc.get("is_error"):
            # The CLI can report an error with text in "result" (e.g. "You've hit your session limit ...");
            # that text is not a model answer, so never return it as content.
            STATS["errors"] += 1
            msg = doc.get("error") or doc.get("api_error") or text[:300] or doc.get("subtype")
            self._send(502, {"error": {"message": f"claude code call failed: {msg}"}})
            return
        self._send(200, {
            "id": f"chatcmpl-cc-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok},
        })


def main() -> None:
    global ARGS, SEM
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18300)
    ap.add_argument("--default-model", default="claude-opus-5-5")
    ap.add_argument("--workers", type=int, default=2, help="concurrent claude -p calls")
    ap.add_argument("--queue-timeout", type=float, default=7200, help="seconds a request waits for a worker")
    ap.add_argument("--call-timeout", type=float, default=3600)
    ap.add_argument("--max-output-tokens", type=int, default=64000)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--tmp-dir", default="/scratch/hpc-prf-llmfpga/asa582/projects/tmp/claude_code_shim")
    ap.add_argument("--claude-bin", default=shutil.which("claude") or "claude")
    ap.add_argument("--agent-match", default="",
                    help="regex (multiline) on the user prompt; matching requests run as a tool-using agent")
    ap.add_argument("--agent-header", default="kernel.h", help="bench header file name the kernel includes (agent mode)")
    ap.add_argument("--agent-bench-dir", default="", help="bench dir with kernel.h, testbench.cpp (agent mode)")
    ap.add_argument("--agent-top", default="autosa_mm")
    ap.add_argument("--agent-part", default="xcu280-fsvh2892-2L-e")
    ap.add_argument("--agent-clock-ns", default="3.33")
    ap.add_argument("--agent-max-checks", type=int, default=12)
    ap.add_argument("--agent-csim-time", default="02:00:00")
    ap.add_argument("--agent-csynth-time", default="08:00:00")
    ap.add_argument("--agent-account", default="hpc-prf-llmfpga")
    ap.add_argument("--agent-timeout", type=float, default=12 * 3600, help="seconds per agent session")
    ARGS = ap.parse_args()
    if ARGS.agent_match and not (Path(ARGS.agent_bench_dir) / "testbench.cpp").is_file():  # and the header named by --agent-header
        ap.error("--agent-match needs --agent-bench-dir with kernel.h and testbench.cpp")
    Path(ARGS.log_dir).mkdir(parents=True, exist_ok=True)
    Path(ARGS.tmp_dir).mkdir(parents=True, exist_ok=True)
    SEM = threading.Semaphore(ARGS.workers)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"claude code shim on http://{ARGS.host}:{ARGS.port}/v1 model={ARGS.default_model} "
          f"workers={ARGS.workers} logs={ARGS.log_dir} agent_match={ARGS.agent_match!r}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
