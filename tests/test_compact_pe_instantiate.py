# tests/test_compact_pe_instantiate.py
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_instantiate import instantiate_mm
from compact_pe_search import enumerate_mm_recipes, candidate_id


def test_16x4_has_16_mm_pe_and_128bit():
    rec = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe16_simd4")
    code = instantiate_mm(rec)
    assert "#define PE_NUM 16" in code
    assert "#define SIMD 4" in code or "#define SIMD   4" in code
    assert "ap_uint<128>" in code
    assert code.count("mm_pe(") == 17  # 1 definition + 16 calls
    assert "for (int i0 = 0; i0 < I; i0 += PE_NUM)" not in code


def test_32x8_wraps_dataflow_and_256bit():
    rec = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe32_simd8")
    code = instantiate_mm(rec)
    assert "ap_uint<256>" in code
    assert code.count("mm_pe(") == 33
    assert "i0 += PE_NUM" in code
    assert "#pragma HLS DATAFLOW" in code


def test_instantiated_32x8_passes_architecture_ok():
    import post_flash_stream as pfs
    rec = next(r for r in enumerate_mm_recipes() if candidate_id(r) == "pe32_simd8")
    code = instantiate_mm(rec)
    report = {
        "dsp": 1280,
        "latency_cycles": 1200,
        "interval": 1100,
        "feedback": {"scopes": [{"name": "pe_kj", "pipeline_ii": 1}]},
    }
    assert pfs.architecture_ok(code, report, "autosa_mm_32x8") is True
