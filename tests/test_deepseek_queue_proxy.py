"""The DeepSeek login proxy must not call upstream after the client is gone."""
from __future__ import annotations

import importlib.util
import socket
import threading
from pathlib import Path

import pytest

_PROXY = (
    Path(__file__).resolve().parents[2]
    / "test-chathls"
    / "ChatHLS-ACL-26"
    / "scripts"
    / "pc2"
    / "deepseek_queue_proxy.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("deepseek_queue_proxy", _PROXY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def proxy():
    return _load()


def test_closed_peer_is_disconnected(proxy):
    left, right = socket.socketpair()
    right.close()
    assert proxy.client_disconnected(left)
    left.close()


def test_open_peer_is_connected(proxy):
    left, right = socket.socketpair()
    try:
        assert proxy.client_disconnected(left) is False
    finally:
        left.close()
        right.close()


def test_disconnected_client_is_not_forwarded(proxy):
    state = proxy.QueueState("https://example.invalid/v1", "test-key", max_workers=1, max_queue=1)
    calls = []

    def _forward(payload, client_sock):
        calls.append(payload)
        return 200, "{}"

    state._forward = _forward
    left, right = socket.socketpair()
    right.close()
    try:
        status, body = state.enqueue({"model": "x"}, left)
    finally:
        left.close()
    assert status == 499
    assert calls == []
    assert "disconnected" in body


def test_devstral_id_is_not_forwarded(proxy):
    assert (
        proxy.coerce_deepseek_model("mistralai/Devstral-2-123B-Instruct-2512")
        == "deepseek-v4-flash"
    )
    assert proxy.coerce_deepseek_model("deepseek-v4-flash") == "deepseek-v4-flash"
    assert proxy.coerce_deepseek_model("deepseek-v4-pro") == "deepseek-v4-pro"
    assert proxy.coerce_deepseek_model("") == "deepseek-v4-flash"


def test_busy_proxy_rejects_without_queueing(proxy):
    state = proxy.QueueState("https://example.invalid/v1", "test-key", max_workers=1, max_queue=1)
    started = threading.Event()
    release = threading.Event()

    def _forward(payload, client_sock):
        started.set()
        release.wait(5)
        return 200, "{}"

    state._forward = _forward
    left, right = socket.socketpair()
    holder = {}

    def _first():
        holder["result"] = state.enqueue({"n": 1}, left)

    thread = threading.Thread(target=_first)
    thread.start()
    assert started.wait(2)
    try:
        status, body = state.enqueue({"n": 2}, right)
        assert status == 429
        assert "busy" in body
    finally:
        release.set()
        thread.join(5)
        left.close()
        right.close()


def test_upstream_timeout_waits_for_a_slow_completion(proxy, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_PROXY_UPSTREAM_TIMEOUT", raising=False)
    assert proxy.upstream_timeout_s() >= 3600
    monkeypatch.setenv("DEEPSEEK_PROXY_UPSTREAM_TIMEOUT", "90")
    assert proxy.upstream_timeout_s() == 90
