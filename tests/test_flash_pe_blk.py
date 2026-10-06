"""Flash PE_BLK prompt knob: 16 / 32 / 64 on the full U280 DSP budget."""
from __future__ import annotations

import os

import autosa_flow_gates as g
from prompt_c2hls import Instruction_c2hls_flash, q_optimize_flash


def test_flash_pe_blk_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_PE_BLK", raising=False)
    assert g.flash_pe_blk() is None
    monkeypatch.setenv("C2HLS_FLASH_PE_BLK", "32")
    assert g.flash_pe_blk() == 32
    monkeypatch.setenv("C2HLS_FLASH_PE_BLK", "nope")
    assert g.flash_pe_blk() is None


def test_flash_pe_blk_guidance_pins_width_and_full_chip_dsp():
    text = g.flash_pe_blk_initial_guidance(32)
    low = text.lower()
    assert "PE_BLK=32" in text
    assert "pe_blk=8" in low
    assert "9024" in text
    assert "mandatory" in low
    assert "slr" not in low


def test_default_flash_prompt_does_not_pin_pe_blk():
    blob = (q_optimize_flash + "\n" + Instruction_c2hls_flash).lower()
    assert "pe_blk must be 16, 32, or 64" not in blob
    assert "pe_blk=16" not in blob
    assert "slr" not in blob


def test_campaign_restores_flash_pe_blk(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_PE_BLK", raising=False)
    apply_autosa_flow_from_campaign({"flash_pe_blk": 64})
    assert os.environ["C2HLS_FLASH_PE_BLK"] == "64"


def test_pe_blk_launcher_rejects_eight():
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pc2"
        / "start_autosa_mm_flash_pe_blk.sh"
    ).read_text(encoding="utf-8")
    assert "C2HLS_FLASH_PE_BLK" in text
    assert "16|32|64" in text
    assert "C2HLS_FLASH_ONLY=1" in text
    assert "16|32|64" in text
    assert "mmpe" in text
