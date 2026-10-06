"""flash_dsp_redo: fill U280 DSP (not a random floor), stay under 100%, pick latency."""
from __future__ import annotations

import os

import autosa_flow_gates as g
import prompt_c2hls as prompts
from c2hls import (
    flash_dsp_ceiling_reject_error,
    flash_dsp_fill_reject_error,
    flash_dsp_floor_reject_error,
    flash_resource_cap_reject_error,
)

U280 = "xcu280-fsvh2892-2L-e"


def _under(**kwargs):
    report = {
        "bram": 10,
        "ff": 1000,
        "lut": 1000,
        "uram": 0,
        "latency_cycles": 5000,
        "latency_cycles_worst": 5000,
    }
    report.update(kwargs)
    return report


def test_flash_dsp_redo_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_DSP_REDO", raising=False)
    assert g.flash_dsp_redo() is False
    monkeypatch.setenv("C2HLS_FLASH_DSP_REDO", "1")
    assert g.flash_dsp_redo() is True


def test_leftover_still_fails_floor():
    leftover = {"dsp": 10, "latency_cycles": 139484}
    assert g.flash_dsp_floor_ok(leftover, min_dsp=300) is False


def test_mid_dsp_2000_and_5000_fail_fill_when_headroom_remains():
    """Floor-only would accept both; redo must reject until DSP is filled."""
    low_2000 = _under(dsp=2000, latency_cycles=8000)
    low_5000 = _under(dsp=5000, latency_cycles=4000)
    assert g.flash_dsp_fill_ok(low_2000, part=U280) is False
    assert g.flash_dsp_fill_ok(low_5000, part=U280) is False
    err = g.flash_dsp_fill_error(low_2000, part=U280)
    assert err is not None
    low = err.lower()
    assert "2000" in err
    assert "rejected" in low
    assert "headroom" in low or "more" in low
    assert g.flash_dsp_fill_error(_under(dsp=8500), part=U280) is None


def test_iso_compute_320_does_not_count_as_filled():
    assert g.flash_dsp_fill_ok(_under(dsp=320), part=U280) is False


def test_fill_accepts_when_dsp_near_device_cap():
    assert g.flash_dsp_fill_ok(_under(dsp=8500), part=U280) is True


def test_fill_accepts_when_other_resource_blocks_more_dsp():
    blocked = _under(dsp=5000, lut=1_200_000)
    assert g.flash_dsp_fill_ok(blocked, part=U280) is True
    assert g.flash_resource_cap_ok(blocked, part=U280) is True


def test_resource_cap_rejects_at_or_above_100_percent():
    over_lut = _under(dsp=4000, lut=1_303_680)
    assert g.flash_resource_cap_ok(over_lut, part=U280) is False
    err = g.flash_resource_cap_error(over_lut, part=U280)
    assert err is not None
    assert "lut" in err.lower()
    assert "rejected" in err.lower()
    under = _under(dsp=4000, lut=1000)
    assert g.flash_resource_cap_ok(under, part=U280) is True
    assert g.flash_resource_cap_error(under, part=U280) is None


def test_dsp_ceiling_still_rejects_over_9024():
    over = {"dsp": 11520, "latency_cycles": 1886}
    assert g.flash_dsp_ceiling_ok(over, max_dsp=9024) is False


def test_select_winner_lowest_latency_among_filled():
    attempts = [
        {
            "success": True,
            "candidate_index": 0,
            "report": _under(dsp=8500, latency_cycles=4000, latency_cycles_worst=4000),
        },
        {
            "success": True,
            "candidate_index": 1,
            "report": _under(dsp=8200, latency_cycles=2000, latency_cycles_worst=2100),
        },
        {
            "success": True,
            "candidate_index": 2,
            "report": _under(dsp=2000, latency_cycles=500, latency_cycles_worst=500),
        },
        {
            "success": False,
            "candidate_index": 3,
            "report": _under(dsp=8600, latency_cycles=100),
        },
    ]
    winner = g.select_flash_dsp_redo_winner(attempts, part=U280)
    assert winner is not None
    assert winner["candidate_index"] == 1
    assert winner["report"]["latency_cycles"] == 2000


def test_select_winner_none_when_nothing_filled():
    attempts = [
        {"success": True, "report": _under(dsp=2000, latency_cycles=100)},
        {"success": True, "report": _under(dsp=5000, latency_cycles=200)},
    ]
    assert g.select_flash_dsp_redo_winner(attempts, part=U280) is None


def test_redo_guidance_asks_to_fill_dsp_not_stop_at_320():
    text = g.flash_dsp_redo_initial_guidance(min_dsp=300, max_dsp=9024)
    low = text.lower()
    assert "9024" in text
    assert "100" in text
    assert "dsp" in low
    assert "latency" in low
    assert "320 is the goal" not in low
    assert "16×4" not in text and "16x4" not in low


def test_redo_floor_fix_does_not_anchor_at_320():
    prompt = prompts.hls_flash_dsp_redo_floor_fix.format(
        min_dsp=300,
        max_dsp=9024,
        dsp=10,
        latency_cycles=139484,
        reject_reason="REJECTED leftover",
        hls_code="extern \"C\" void autosa_mm() {}\n",
        header_code="// kernel.h",
        attempt_history="",
    )
    low = prompt.lower()
    assert "300" in prompt
    assert "9024" in prompt
    assert "16×4" not in prompt and "16x4 → ~320" not in prompt
    assert "```cpp" in prompt
    assert "dsp" in low


def test_fill_and_cap_fix_prompts_include_report():
    fill = prompts.hls_flash_dsp_fill_fix.format(
        max_dsp=9024,
        fill_pct=90,
        dsp=2000,
        latency_cycles=8000,
        reject_reason="REJECTED: headroom",
        hls_code="extern \"C\" void autosa_mm() {}\n",
        header_code="// kernel.h",
        attempt_history="",
    )
    cap = prompts.hls_flash_resource_cap_fix.format(
        max_dsp=9024,
        dsp=4000,
        latency_cycles=3000,
        reject_reason="REJECTED: lut over cap",
        hls_code="extern \"C\" void autosa_mm() {}\n",
        header_code="// kernel.h",
        attempt_history="",
    )
    assert "2000" in fill and "9024" in fill and "```cpp" in fill
    assert "rejected" in cap.lower() and "```cpp" in cap


def test_c2hls_helpers_only_gate_flash_when_redo_on(monkeypatch):
    monkeypatch.setenv("C2HLS_FLASH_DSP_REDO", "1")
    monkeypatch.setenv("C2HLS_FLASH_MIN_DSP", "300")
    monkeypatch.setenv("C2HLS_FLASH_MAX_DSP", "9024")
    leftover = {"dsp": 10, "latency_cycles": 139484}
    mid = _under(dsp=2000, latency_cycles=8000)
    over = _under(dsp=4000, lut=1_303_680)
    assert flash_dsp_floor_reject_error("flash", leftover)
    assert flash_dsp_floor_reject_error("tiling", leftover) is None
    assert flash_dsp_fill_reject_error("flash", mid)
    assert flash_dsp_fill_reject_error("tiling", mid) is None
    assert flash_resource_cap_reject_error("flash", over)
    assert flash_resource_cap_reject_error("tiling", over) is None
    assert flash_dsp_ceiling_reject_error("flash", {"dsp": 11520})
    monkeypatch.delenv("C2HLS_FLASH_DSP_REDO", raising=False)
    assert flash_dsp_fill_reject_error("flash", mid) is None
    assert flash_resource_cap_reject_error("flash", over) is None


def test_campaign_restores_redo_knobs(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_DSP_REDO", raising=False)
    monkeypatch.delenv("C2HLS_FLASH_MIN_DSP", raising=False)
    monkeypatch.delenv("C2HLS_FLASH_MAX_DSP", raising=False)
    monkeypatch.delenv("C2HLS_FLASH_DSP_FILL_PCT", raising=False)
    monkeypatch.delenv("C2HLS_CANDIDATES_PER_STEP", raising=False)
    apply_autosa_flow_from_campaign(
        {
            "flash_dsp_redo": 1,
            "flash_min_dsp": 300,
            "flash_max_dsp": 9024,
            "flash_dsp_fill_pct": 90,
            "candidates_per_step": '{"flash": 3}',
        }
    )
    assert os.environ["C2HLS_FLASH_DSP_REDO"] == "1"
    assert os.environ["C2HLS_FLASH_MIN_DSP"] == "300"
    assert os.environ["C2HLS_FLASH_MAX_DSP"] == "9024"
    assert os.environ["C2HLS_FLASH_DSP_FILL_PCT"] == "90"
    assert "flash" in os.environ["C2HLS_CANDIDATES_PER_STEP"]
