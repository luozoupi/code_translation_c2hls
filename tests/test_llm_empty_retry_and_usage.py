"""Tests for empty-reply rejection and finish_reason/usage logging."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2hls import C2HLSOrchestrator


def test_record_llm_usage_includes_finish_reason_and_empty_flag():
    orch = C2HLSOrchestrator.__new__(C2HLSOrchestrator)
    orch.llm_usage_events = []
    orch._record_llm_usage(
        provider="openai",
        model="claude-sonnet-5",
        agent_name="orchestrator",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        messages=[{"role": "user", "content": "x"}],
        max_tokens=1024,
        finish_reason="stop",
        content_len=0,
        content_empty=True,
    )
    assert len(orch.llm_usage_events) == 1
    event = orch.llm_usage_events[0]
    assert event["finish_reason"] == "stop"
    assert event["content_len"] == 0
    assert event["content_empty"] is True
    assert event["usage_available"] is True


def test_normalize_llm_content_none_becomes_empty_string():
    assert C2HLSOrchestrator._normalize_llm_content(None) == ""
    assert C2HLSOrchestrator._normalize_llm_content("  hi  ") == "  hi  "


def test_openai_compat_extra_body_deepseek_thinking_toggle(monkeypatch):
    import c2hls as c2

    monkeypatch.delenv("C2HLS_THINKING", raising=False)
    assert c2._openai_compat_extra_body("deepseek-v4-flash") == {}
    monkeypatch.setenv("C2HLS_THINKING", "disabled")
    assert c2._openai_compat_extra_body("deepseek-v4-flash") == {
        "thinking": {"type": "disabled"}
    }
    monkeypatch.setenv("C2HLS_THINKING", "enabled")
    assert c2._openai_compat_extra_body("deepseek-v4-flash") == {
        "thinking": {"type": "enabled"}
    }
    monkeypatch.setenv("C2HLS_THINKING", "api_default")
    assert c2._openai_compat_extra_body("deepseek-v4-flash") == {}
    monkeypatch.setenv("C2HLS_THINKING", "disabled")
    assert c2._openai_compat_extra_body("gpt-4o") == {}
    qwen = c2._openai_compat_extra_body("Qwen3-32B")
    assert qwen["chat_template_kwargs"]["enable_thinking"] is False
    assert "thinking" not in qwen


def test_message_text_falls_back_to_reasoning_content():
    empty = SimpleNamespace(content="", reasoning_content="```kernel\nint x;\n```")
    assert "int x;" in C2HLSOrchestrator._message_text(empty)
    prefer = SimpleNamespace(content="hello", reasoning_content="hidden")
    assert C2HLSOrchestrator._message_text(prefer) == "hello"
    none = SimpleNamespace(content=None)
    assert C2HLSOrchestrator._message_text(none) == ""


if __name__ == "__main__":
    test_record_llm_usage_includes_finish_reason_and_empty_flag()
    test_normalize_llm_content_none_becomes_empty_string()
    print("test_llm_empty_retry_and_usage: ok")
