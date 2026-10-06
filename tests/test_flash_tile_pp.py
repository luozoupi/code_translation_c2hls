"""Flash in-GEMM tile ping-pong knob (C2HLS_FLASH_TILE_PP)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import autosa_flow_gates as g
from prompt_c2hls import Instruction_c2hls_flash, q_optimize_flash

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
_PACKS = (
    PKG / "skills_ii_target_miss_solutions_added(90skills).json",
    PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json",
    PKG / "skills_ii_target_miss_solutions_added(90skills)_no_RMW_m_axi.json",
)


def test_flash_tile_pp_off_by_default(monkeypatch):
    monkeypatch.delenv("C2HLS_FLASH_TILE_PP", raising=False)
    assert g.flash_tile_pp() is False
    monkeypatch.setenv("C2HLS_FLASH_TILE_PP", "1")
    assert g.flash_tile_pp() is True
    monkeypatch.setenv("C2HLS_FLASH_TILE_PP", "yes")
    assert g.flash_tile_pp() is True
    monkeypatch.setenv("C2HLS_FLASH_TILE_PP", "0")
    assert g.flash_tile_pp() is False


def test_flash_tile_pp_guidance_rejects_bulk_dataflow():
    text = g.flash_tile_pp_initial_guidance()
    low = text.lower()
    assert "mandatory" in low
    assert "tile" in low
    assert "buf[2]" in text or "t & 1" in text
    assert "load(t+1)" in text
    assert "full-matrix" in low
    assert "64 one-row" in low or "tiny" in low
    assert "pe_blk=16" in low
    assert "slr" not in low


def test_default_flash_prompt_does_not_mandate_tile_ping_pong():
    blob = (q_optimize_flash + "\n" + Instruction_c2hls_flash).lower()
    assert "ping-pong of **tiles**" not in q_optimize_flash.lower()
    assert "2 or 4 large tiles" not in blob
    assert "pe_blk=16" not in blob


def test_default_lcst_does_not_require_tile_loop():
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "hls-baseline-load-compute-store-gate")
        steps = " ".join(sk["required_steps"]).lower()
        assert "ping-pong tiles" not in steps
        assert "tile-loop dataflow" not in steps
        tmpl = sk["template"]
        assert "tile_loop" not in tmpl
        assert "NT = 4" not in tmpl
        assert "LANES" not in tmpl


def test_campaign_restores_flash_tile_pp(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "pc2"))
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_FLASH_TILE_PP", raising=False)
    apply_autosa_flow_from_campaign({"flash_tile_pp": 1})
    assert os.environ["C2HLS_FLASH_TILE_PP"] == "1"


def test_tile_pp_launcher_forces_pe16_and_prefix():
    text = (
        REPO / "scripts" / "pc2" / "start_autosa_mm_flash_tile_pp.sh"
    ).read_text(encoding="utf-8")
    assert "C2HLS_FLASH_TILE_PP=1" in text
    assert "C2HLS_FLASH_PE_BLK=16" in text
    assert "mmpe16pp" in text
    assert "batch_parallel_autosa_mm_flash_dsp500_pe16_tilepp" in text
    assert "C2HLS_FLASH_ONLY=1" in text
    assert "unset BATCH_PARALLEL_ARTIFACT_PREFIX" in text
