"""Tests for the DeepSeek Beijing-peak codegen pause gate in the GPU drain loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

import batch_parallel_gpu_drain as drain


def test_peak_pause_inactive_without_gate(monkeypatch):
    monkeypatch.delenv("C2HLS_DEEPSEEK_PEAK_PAUSE", raising=False)
    monkeypatch.setattr(drain, "is_beijing_peak", lambda: True)
    assert drain._peak_pause_active({}) is False


def test_peak_pause_active_for_external_llm_campaign(monkeypatch):
    monkeypatch.delenv("C2HLS_DEEPSEEK_PEAK_PAUSE", raising=False)
    monkeypatch.setattr(drain, "is_beijing_peak", lambda: True)
    assert drain._peak_pause_active({"external_llm": True}) is True


def test_peak_pause_active_for_env_flag(monkeypatch):
    monkeypatch.setenv("C2HLS_DEEPSEEK_PEAK_PAUSE", "1")
    monkeypatch.setattr(drain, "is_beijing_peak", lambda: True)
    assert drain._peak_pause_active({}) is True


def test_peak_pause_inactive_when_off_peak(monkeypatch):
    monkeypatch.setenv("C2HLS_DEEPSEEK_PEAK_PAUSE", "1")
    monkeypatch.setattr(drain, "is_beijing_peak", lambda: False)
    assert drain._peak_pause_active({"external_llm": True}) is False


def test_peak_pause_skip_env_overrides_external_llm(monkeypatch):
    monkeypatch.setenv("C2HLS_DEEPSEEK_SKIP_PEAK", "1")
    monkeypatch.setenv("C2HLS_DEEPSEEK_PEAK_PAUSE", "1")
    monkeypatch.setattr(drain, "is_beijing_peak", lambda: True)
    assert drain._peak_pause_active({"external_llm": True}) is False


def test_peak_pause_skip_campaign_flag_overrides_external_llm(monkeypatch):
    monkeypatch.delenv("C2HLS_DEEPSEEK_SKIP_PEAK", raising=False)
    monkeypatch.setattr(drain, "is_beijing_peak", lambda: True)
    assert drain._peak_pause_active({"external_llm": True, "skip_peak_pause": True}) is False
