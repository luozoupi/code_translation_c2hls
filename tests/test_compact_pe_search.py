# tests/test_compact_pe_search.py
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_search import enumerate_mm_recipes, candidate_id


def test_mm_grid_includes_locked_16x4_and_32x8():
    recs = {candidate_id(r): r for r in enumerate_mm_recipes()}
    assert "pe16_simd4" in recs
    r = recs["pe16_simd4"]
    assert r.pe == 16 and r.simd == 4 and r.pack_bits == 128
    assert r.pe_kj == 1024 and r.i_tiles == 1
    assert r.tile_loop == "inside_tasks"
    assert recs["pe32_simd8"].pack_bits == 256
    assert recs["pe32_simd8"].pe_kj == 512
    assert recs["pe32_simd8"].tile_loop == "around_dataflow"
    assert recs["pe32_simd8"].expected_dsp == 1280


def test_mm_grid_pe_and_simd_divide_64():
    for r in enumerate_mm_recipes():
        assert 64 % r.pe == 0
        assert 64 % r.simd == 0


def test_grid_is_small():
    recs = enumerate_mm_recipes()
    assert 8 <= len(recs) <= 16
