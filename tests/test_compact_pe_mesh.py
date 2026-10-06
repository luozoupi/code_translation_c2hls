from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_instantiate import instantiate_mm
from compact_pe_search import (
    candidate_id,
    candidate_id_mesh,
    enumerate_mm_mesh_recipes,
    enumerate_mm_recipes,
    search_candidate_id,
)


def test_mesh_grid_includes_autosa_analogs():
    recs = {candidate_id_mesh(r): r for r in enumerate_mm_mesh_recipes()}
    r = recs["mesh8x4_simd8"]
    assert r.layout == "mesh"
    assert r.pe_i == 8 and r.pe_j == 4 and r.simd == 8
    assert r.pe == 32 and r.pack_bits == 256
    assert r.expected_dsp == 1280
    assert r.pe_kj == (64 // 8) * (64 // 4)  # 16 * 16 = 256
    assert r.tile_loop == "inside_tasks"
    r5 = recs["mesh16x8_simd8"]
    assert r5.pe == 128 and r5.expected_dsp == 5120
    assert r5.pe_i == 16 and r5.pe_j == 8


def test_mesh_caps_drop_oversize():
    recs = enumerate_mm_mesh_recipes()
    for r in recs:
        assert r.pe_i * r.pe_j <= 128
        assert r.expected_dsp <= 0.85 * 9024
        assert 64 % r.pe_i == 0 and 64 % r.pe_j == 0 and 64 % r.simd == 0
        assert r.layout == "mesh"
    ids = {candidate_id_mesh(r) for r in recs}
    assert "mesh32x16_simd2" not in ids
    assert 20 <= len(recs) <= 40


def test_search_candidate_id_dispatches():
    chain = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe16_simd4")
    mesh = next(r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8")
    assert search_candidate_id(chain) == "pe16_simd4"
    assert search_candidate_id(mesh) == "mesh8x4_simd8"


def test_mesh_8x4_has_32_mesh_pe_and_256bit():
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )
    code = instantiate_mm(rec)
    assert "#define PE_I 8" in code
    assert "#define PE_J 4" in code
    assert "ap_uint<256>" in code
    assert code.count("mesh_pe(") == 33  # 1 definition + 32 calls
    assert "mm_pe(" not in code
    assert "#pragma HLS DATAFLOW" in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_I)" not in code
    assert "for (int i0 = 0; i0 < I; i0 += PE_NUM)" not in code


def test_mesh_16x8_has_128_mesh_pe():
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh16x8_simd8"
    )
    code = instantiate_mm(rec)
    assert code.count("mesh_pe(") == 129
    assert "#define PE_I 16" in code
    assert "#define PE_J 8" in code


def test_instantiated_mesh_8x4_passes_architecture_ok_for_recipe():
    import post_flash_stream as pfs
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )
    code = instantiate_mm(rec)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok_for_recipe(code, report, rec) is True


def test_mesh_architecture_ok_rejects_wrong_pe_count():
    import post_flash_stream as pfs
    rec = next(
        r for r in enumerate_mm_mesh_recipes() if candidate_id_mesh(r) == "mesh8x4_simd8"
    )
    code = instantiate_mm(rec).replace("mesh_pe(", "mm_pe(", 1)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok_for_recipe(code, report, rec) is False
