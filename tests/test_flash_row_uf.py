"""Flash ROW_UF prompt knob: tell the model to cover I in one tile."""
from __future__ import annotations

import os

import autosa_flow_gates as g


def test_flash_row_uf_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_ROW_UF", raising=False)
    assert g.flash_row_uf() is None
    monkeypatch.setenv("C2HLS_FLASH_ROW_UF", "64")
    assert g.flash_row_uf() == 64
    monkeypatch.setenv("C2HLS_FLASH_ROW_UF", "nope")
    assert g.flash_row_uf() is None


def test_flash_row_uf_initial_guidance_unrolls_rows():
    text = g.flash_row_uf_initial_guidance(8)
    low = text.lower()
    assert "ROW_UF=8" in text or "ROW_UF = 8" in text
    assert "1024" in text
    assert "9024" in text
    assert "unroll" in low
    assert "mandatory" in low


def test_campaign_restores_flash_row_uf(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_ROW_UF", raising=False)
    apply_autosa_flow_from_campaign({"flash_row_uf": 64})
    assert os.environ["C2HLS_FLASH_ROW_UF"] == "64"


def test_row_uf_launcher_sets_env_and_flash_only():
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pc2"
        / "start_autosa_mm_flash_row_uf.sh"
    ).read_text(encoding="utf-8")
    assert "C2HLS_FLASH_ROW_UF" in text
    assert "64" in text
    assert "C2HLS_FLASH_ONLY=1" in text
    assert "mmuf64" in text
