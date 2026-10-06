from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_pe_rank import rank_candidates


def test_rank_drops_csim_fail_and_dsp_miss():
    rows = [
        {"cand_id": "pe16_simd4", "hls_csim_pass": True, "hls_csynth_pass": True,
         "architecture_ok": True, "csynth_latency": 4292, "csynth_dsp": 320},
        {"cand_id": "pe32_simd8", "hls_csim_pass": True, "hls_csynth_pass": True,
         "architecture_ok": True, "csynth_latency": 4583, "csynth_dsp": 1280},
        {"cand_id": "pe8_simd2", "hls_csim_pass": False, "hls_csynth_pass": True,
         "architecture_ok": True, "csynth_latency": 9000, "csynth_dsp": 80},
        {"cand_id": "pe64_simd8", "hls_csim_pass": True, "hls_csynth_pass": True,
         "architecture_ok": False, "csynth_latency": 2000, "csynth_dsp": 50},
    ]
    ranked = rank_candidates(rows)
    assert [r["cand_id"] for r in ranked] == ["pe16_simd4", "pe32_simd8"]
    assert ranked[0]["queue_rank"] == 1


def test_rank_latency_tie_breaks_on_dsp_desc():
    rows = [
        {"cand_id": "pe_a", "hls_csim_pass": True, "hls_csynth_pass": True,
         "architecture_ok": True, "csynth_latency": 1000, "csynth_dsp": 320},
        {"cand_id": "pe_b", "hls_csim_pass": True, "hls_csynth_pass": True,
         "architecture_ok": True, "csynth_latency": 1000, "csynth_dsp": 640},
        {"cand_id": "pe_c", "hls_csim_pass": True, "hls_csynth_pass": True,
         "architecture_ok": True, "csynth_latency": 1000},
    ]
    ranked = rank_candidates(rows)
    assert [r["cand_id"] for r in ranked] == ["pe_b", "pe_a", "pe_c"]
    assert [r["queue_rank"] for r in ranked] == [1, 2, 3]
