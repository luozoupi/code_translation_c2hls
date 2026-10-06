"""Flash DSP floor: reject csynth DSP below cutoff and tell the LLM why."""
from __future__ import annotations

import os

import autosa_flow_gates as g
import prompt_c2hls as prompts
from c2hls import flash_dsp_floor_reject_error


def test_flash_min_dsp_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_MIN_DSP", raising=False)
    assert g.flash_min_dsp() is None
    monkeypatch.setenv("C2HLS_FLASH_MIN_DSP", "300")
    assert g.flash_min_dsp() == 300
    monkeypatch.setenv("C2HLS_FLASH_MIN_DSP", "nope")
    assert g.flash_min_dsp() is None


def test_flash_dsp_floor_rejects_flash_leftover():
    leftover = {"dsp": 10, "latency_cycles": 139484}
    assert g.flash_dsp_floor_ok(leftover, min_dsp=300) is False
    err = g.flash_dsp_floor_error(leftover, min_dsp=300)
    assert err is not None
    low = err.lower()
    assert "300" in err
    assert "10" in err
    assert "rejected" in low
    assert "dsp" in low
    assert g.flash_dsp_floor_ok({"dsp": 320}, min_dsp=300) is True
    assert g.flash_dsp_floor_error({"dsp": 320}, min_dsp=300) is None
    assert g.flash_dsp_floor_ok({}, min_dsp=300) is False


def test_flash_dsp_floor_repair_prompt_leads_with_cutoff_and_reason():
    report = {"dsp": 10, "latency_cycles": 139484}
    err = g.flash_dsp_floor_error(report, min_dsp=300)
    prompt = prompts.hls_flash_dsp_floor_fix.format(
        min_dsp=300,
        dsp=10,
        latency_cycles=139484,
        reject_reason=err,
        hls_code="extern \"C\" void autosa_mm() { /* leftover */ }\n",
        header_code="// kernel.h",
        attempt_history="",
    )
    head = prompt[:800].lower()
    assert head.index("dsp") < 80
    assert "must be" in head or "cutoff" in head or "hard reject" in head
    assert "300" in prompt
    assert "10" in prompt
    assert "why" in prompt.lower() or "not accepted" in prompt.lower() or "rejected" in prompt.lower()
    assert "leftover" in prompt
    assert "```cpp" in prompt


def test_flash_dsp_floor_initial_guidance_mentions_cutoff():
    text = g.flash_dsp_floor_initial_guidance(300)
    low = text.lower()
    assert "300" in text
    assert "dsp" in low
    assert "reject" in low or "will not accept" in low


def test_c2hls_helper_only_gates_flash_step(monkeypatch):
    monkeypatch.setenv("C2HLS_FLASH_MIN_DSP", "300")
    report = {"dsp": 10, "latency_cycles": 139484}
    assert flash_dsp_floor_reject_error("flash", report)
    assert flash_dsp_floor_reject_error("tiling", report) is None
    monkeypatch.delenv("C2HLS_FLASH_MIN_DSP", raising=False)
    assert flash_dsp_floor_reject_error("flash", report) is None


def test_flash_dsp_ceiling_rejects_over_device(monkeypatch):
    over = {"dsp": 11520, "latency_cycles": 1886}
    assert g.flash_dsp_ceiling_ok(over, max_dsp=9024) is False
    err = g.flash_dsp_ceiling_error(over, max_dsp=9024)
    assert err is not None
    low = err.lower()
    assert "11520" in err
    assert "9024" in err
    assert "rejected" in low
    assert g.flash_dsp_ceiling_ok({"dsp": 5344}, max_dsp=9024) is True
    assert g.flash_dsp_ceiling_error({"dsp": 5344}, max_dsp=9024) is None
    assert g.flash_dsp_ceiling_ok({}, max_dsp=9024) is False


def test_flash_dsp_ceiling_repair_prompt_leads_with_cap():
    report = {"dsp": 11520, "latency_cycles": 1886}
    err = g.flash_dsp_ceiling_error(report, max_dsp=9024)
    prompt = prompts.hls_flash_dsp_ceiling_fix.format(
        max_dsp=9024,
        dsp=11520,
        latency_cycles=1886,
        reject_reason=err,
        hls_code="extern \"C\" void autosa_cnn() { /* overshoot */ }\n",
        header_code="// kernel.h",
        attempt_history="",
    )
    head = prompt[:800].lower()
    assert "9024" in prompt
    assert "11520" in prompt
    assert "rejected" in head or "hard reject" in head
    assert "```cpp" in prompt


def test_c2hls_helper_ceiling_only_gates_flash_step(monkeypatch):
    from c2hls import flash_dsp_ceiling_reject_error

    monkeypatch.setenv("C2HLS_FLASH_MAX_DSP", "9024")
    report = {"dsp": 11520, "latency_cycles": 1886}
    assert flash_dsp_ceiling_reject_error("flash", report)
    assert flash_dsp_ceiling_reject_error("tiling", report) is None
    monkeypatch.delenv("C2HLS_FLASH_MAX_DSP", raising=False)
    assert flash_dsp_ceiling_reject_error("flash", report) is None


def test_campaign_restores_flash_min_dsp(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_MIN_DSP", raising=False)
    apply_autosa_flow_from_campaign({"flash_min_dsp": 300})
    assert os.environ["C2HLS_FLASH_MIN_DSP"] == "300"


def test_campaign_restores_flash_max_dsp(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_MAX_DSP", raising=False)
    apply_autosa_flow_from_campaign({"flash_max_dsp": 9024})
    assert os.environ["C2HLS_FLASH_MAX_DSP"] == "9024"
