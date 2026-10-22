"""Controlled LLM call layer for the DeepSeek-vs-Claude rerun (applied at runtime; c2hls stays read-only).

apply() replaces C2HLSOrchestrator._call_llm_with_model with one code path used identically for both
models (both go through the OpenAI-compatible client: Claude via claude_code_shim.py, DeepSeek via
controlled/ds_proxy.py):

  1. Same prompt format: every call is flattened with claude_code_shim.render() into exactly
     [system, one user message]. The shim passes a single system+user message through unchanged, so
     both models receive byte-identical text (multi-turn repairs become the same transcript for both).
  2. Answer only: the reply is message.content. Unlike c2hls's _message_text(), reasoning_content is
     never used as the answer.
  3. Truncation / empty answer: finish_reason == "length" or an empty answer is retried (up to
     CTRL_LEN_RETRIES, default 2). The last reply is returned if all tries fail, so c2hls's own repair
     loop sees the same thing for both models.
  4. Transport errors (HTTP 5xx, resets, timeouts) are retried up to CTRL_TRANSPORT_RETRIES (default 3)
     with backoff, for both models.
  5. Output budget: every call gets max(c2hls budget, CTRL_MIN_MAX_TOKENS), for both models.
  6. Backslash-continued preprocessor lines are joined before c2hls's line-based pragma preflight (which
     otherwise splits a continued INTERFACE pragma into stray C++). Logged as event joined_pragma_continuations.
  7. Every try is logged to CTRL_LLM_LOG (JSON lines): agent, model, try, outcome, finish_reason,
     token usage, answer length, and a sha256 of the exact prompt sent, so first-prompt identity can
     be checked across models.
Token usage of every completed try is still recorded through c2hls's _record_llm_usage.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # claude_code_shim.py lives one level up


def _log(rec: dict) -> None:
    path = os.environ.get("CTRL_LLM_LOG")
    if path:
        with open(path, "a") as fp:
            fp.write(json.dumps(rec) + "\n")


def apply(log=print) -> dict:
    import c2hls
    from claude_code_shim import render

    len_retries = int(os.environ.get("CTRL_LEN_RETRIES", "2"))
    transport_retries = int(os.environ.get("CTRL_TRANSPORT_RETRIES", "3"))

    def controlled_call(self, messages: list, model: str = None, max_tokens: int = None,
                        agent_name: str = "orchestrator") -> str:
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        # One output budget policy for every call and both models (c2hls gives repairs/continuations a
        # smaller budget than flash). The Claude shim still caps at its CLI limit (64000).
        floor = int(os.environ.get("CTRL_MIN_MAX_TOKENS", "0") or 0)
        max_tokens = max(int(max_tokens or 0), floor) or None
        if not model:
            model = self.gpt_model
        system, user = render(messages)
        flat = [{"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": user}]
        prompt_sha = hashlib.sha256(json.dumps(flat, ensure_ascii=False).encode()).hexdigest()[:16]
        kind, client = self._client_for_model(model)
        kwargs = {"model": model, "messages": flat}
        if c2hls._uses_openai_compat_reasoning(model):
            kwargs["max_completion_tokens"] = max_tokens
            effort = c2hls._openai_reasoning_effort()
            if effort:
                kwargs["reasoning_effort"] = effort
        else:
            kwargs["max_tokens"] = max_tokens
        extra = c2hls._openai_compat_extra_body(model)
        if extra:
            kwargs["extra_body"] = extra
        c2hls._notify_batch_llm_hook("llm_enter", model=model, agent_name=agent_name)
        content, tries_len, tries_tx = "", 0, 0
        try:
            while True:
                t0 = time.time()
                rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": agent_name, "model": model,
                       "prompt_sha": prompt_sha, "prompt_chars": len(system) + len(user),
                       "max_tokens": max_tokens, "len_try": tries_len, "tx_try": tries_tx}
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    rec.update(outcome="transport_error", error=f"{type(exc).__name__}: {str(exc)[:300]}",
                               elapsed_s=round(time.time() - t0, 1))
                    _log(rec)
                    if tries_tx >= transport_retries:
                        raise
                    tries_tx += 1
                    time.sleep(min(300, 30 * 2 ** (tries_tx - 1)))
                    continue
                choice = response.choices[0]
                finish = getattr(choice, "finish_reason", None)
                content = self._normalize_llm_content(getattr(choice.message, "content", None))
                usage = getattr(response, "usage", None)
                details = getattr(usage, "completion_tokens_details", None) if usage else None
                rec.update(finish_reason=finish, answer_chars=len(content),
                           prompt_tokens=getattr(usage, "prompt_tokens", None),
                           completion_tokens=getattr(usage, "completion_tokens", None),
                           reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
                           elapsed_s=round(time.time() - t0, 1))
                self._record_llm_usage(provider="openai", model=model, agent_name=agent_name, usage=usage,
                                       messages=flat, max_tokens=max_tokens, finish_reason=finish,
                                       content_len=len(content), content_empty=not content.strip())
                bad = finish == "length" or not content.strip()
                rec["outcome"] = ("truncated" if finish == "length" else "empty") if bad else "ok"
                _log(rec)
                if bad and tries_len < len_retries:
                    tries_len += 1
                    continue
                return content
        finally:
            c2hls._notify_batch_llm_hook("llm_exit", model=model, agent_name=agent_name)

    c2hls.C2HLSOrchestrator._call_llm_with_model = controlled_call

    # c2hls's preflight rewrites INTERFACE pragmas line by line (it inserts an s_axilite line after each m_axi
    # line). A pragma continued with a trailing backslash is then split and its tail becomes stray C++
    # ("'depth' was not declared"). Join backslash-continued preprocessor lines first, as the C preprocessor
    # itself does (translation phase 2), so a model's line-wrapping style is not penalised. Same for both models.
    original_preflight = c2hls.C2HLSOrchestrator._preflight_generated_hls_code

    def join_continuations(code: str) -> str:
        out, buf = [], None
        for line in (code or "").split("\n"):
            if buf is not None:
                buf = buf[:-1].rstrip() + " " + line.strip()
            elif line.lstrip().startswith("#") and line.rstrip().endswith("\\"):
                buf = line.rstrip()
            else:
                out.append(line)
                continue
            if buf.rstrip().endswith("\\"):
                buf = buf.rstrip()
                continue
            out.append(buf)
            buf = None
        if buf is not None:
            out.append(buf.rstrip("\\"))
        return "\n".join(out)

    def preflight(self, code: str, context: str) -> str:
        joined = join_continuations(code)
        if joined != code:
            _log({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "joined_pragma_continuations", "context": context})
        return original_preflight(self, joined, context)

    c2hls.C2HLSOrchestrator._preflight_generated_hls_code = preflight
    info = {"controlled": True, "len_retries": len_retries, "transport_retries": transport_retries,
            "format": "claude_code_shim.render -> [system, user]", "answer": "message.content only",
            "log": os.environ.get("CTRL_LLM_LOG", "")}
    log(f"ctrl_llm applied: {json.dumps(info)}")
    return info
