"""Per-kernel PE recipes for AutoSA 64^3 GEMM DSE/stream."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_flash_pe_recipe import format_recipe_prompt, known_benches, recipe_for


def test_mm_default_is_16x4_float(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    rec = recipe_for("autosa_mm")
    assert rec.pe == 16
    assert rec.simd == 4
    assert rec.pack_bits == 128
    assert rec.min_dsp == 200
    assert rec.expected_dsp == 320
    assert rec.tile_loop == "inside_tasks"


def test_pe_recipe_mesh_fields_default_to_chain(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    rec = recipe_for("autosa_mm")
    assert rec.layout == "chain"
    assert rec.pe_i == 0
    assert rec.pe_j == 1


def test_mm_32x8_is_separate_recipe(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    rec = recipe_for("autosa_mm_32x8")
    assert rec.bench == "autosa_mm_32x8"
    assert rec.pe == 32
    assert rec.simd == 8
    assert rec.pack_bits == 256
    assert rec.pe_kj == 512
    assert rec.i_tiles == 2
    assert rec.tile_loop == "around_dataflow"
    assert rec.expected_dsp == 1280
    assert rec.min_dsp == 800
    prompt = format_recipe_prompt("autosa_mm_32x8", step="stream")
    assert "PE_NUM=32" in prompt
    assert "SIMD=8" in prompt
    assert "ap_uint<256>" in prompt
    assert "around_dataflow" in prompt
    assert "wraps DATAFLOW" in prompt
    assert "do not copy pe_num 16" in prompt.lower()
    assert "autosa_mm_32x8" in known_benches()


def test_pe_recipe_env_overrides_autosa_mm_bench(monkeypatch):
    monkeypatch.setenv("C2HLS_PE_RECIPE", "autosa_mm_32x8")
    rec = recipe_for("autosa_mm")
    assert rec.bench == "autosa_mm_32x8"
    assert rec.pe == 32
    assert rec.simd == 8
    assert rec.pack_bits == 256
    prompt = format_recipe_prompt("autosa_mm", step="dse")
    assert "PE_NUM=32" in prompt
    assert "SIMD=8" in prompt


def test_unknown_pe_recipe_env_falls_back_to_bench(monkeypatch):
    monkeypatch.setenv("C2HLS_PE_RECIPE", "does_not_exist")
    rec = recipe_for("autosa_mm")
    assert rec.bench == "autosa_mm"
    assert rec.pe == 16


def test_getting_started_and_intel_need_32x4():
    for bench in ("autosa_mm_getting_started", "autosa_mm_intel"):
        rec = recipe_for(bench)
        assert rec.pe == 32
        assert rec.simd == 4
        assert rec.pack_bits == 128
        assert rec.min_dsp == 400
        assert rec.expected_dsp == 640
        assert rec.i_tiles == 2
        assert rec.tile_loop == "around_dataflow"
    prompt = format_recipe_prompt("autosa_mm_getting_started", step="stream")
    assert "around_dataflow" in prompt
    assert "wraps DATAFLOW" in prompt


def test_int16_is_32x2_uint16_not_float_pack():
    rec = recipe_for("autosa_mm_int16")
    assert rec.pe == 32
    assert rec.simd == 2
    assert rec.data_kind == "uint16"
    assert rec.pack_bits == 32
    assert rec.min_dsp == 40
    assert rec.expected_dsp == 64
    prompt = format_recipe_prompt("autosa_mm_int16", step="stream")
    assert "unsigned short" in prompt or "uint16" in prompt
    assert "float union" in prompt.lower() or "no float union" in prompt.lower()


def test_catapult_uses_i_p_and_pe8():
    rec = recipe_for("autosa_mm_catapult")
    assert rec.pe == 8
    assert rec.simd == 4
    assert rec.i_macro == "I_P"
    assert rec.data_kind == "uint32"
    prompt = format_recipe_prompt("autosa_mm_catapult")
    assert "I_P" in prompt
    assert "PE_NUM=8" in prompt


def test_hcl_is_32x2_float():
    rec = recipe_for("autosa_mm_hcl")
    assert rec.pe == 32
    assert rec.simd == 2
    assert rec.pack_bits == 64
    assert rec.expected_dsp == 320


def test_unknown_bench_falls_back_to_mm(monkeypatch):
    monkeypatch.delenv("C2HLS_PE_RECIPE", raising=False)
    rec = recipe_for("autosa_cnn")
    assert rec.bench == "autosa_mm"
    assert rec.pe == 16


def test_flow_launcher_accepts_32x8_pe_recipe():
    text = (Path(__file__).resolve().parents[1] / "scripts/pc2/start_autosa_mm_flow.sh").read_text(
        encoding="utf-8"
    )
    assert "--pe-recipe" in text
    assert "autosa_mm_32x8" in text
    assert "mm32x8" in text
