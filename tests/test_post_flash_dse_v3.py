"""DSE v3 prompt rules and the optional harness. V2 stays put when both flags are unset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_dse_v2 as v2
import post_flash_dse_v3 as v3
from c2hls_paths import (
    POST_FLASH_DSE_SKILL_ENTRIES_JSON,
    POST_FLASH_DSE_V3_SKILL_ENTRIES_JSON,
)
from hls_eval import _config_compile_jobs_tcl, csynth_pre_commands


HEADER_64 = """
typedef float data_t;
#define I 64
#define J 64
#define K 64
"""

PARTIAL_TILE = """
#define TI 64
#define TJ 64
#define TK 64
#define PE 32
#define SIMD 32
for (int i0 = 0; i0 < TI; i0 += PE) {
  for (int k = 0; k < TK; k += SIMD) {
    for (int j = 0; j < TJ; ++j) {}
  }
}
"""

FULL_K = """
for (int i = 0; i < I; ++i) {
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {}
  }
}
"""

FULL_K_OUTER = """
#define TK 64
for (int k0 = 0; k0 < K; k0 += TK) {
  for (int k = 0; k < TK; ++k) {}
}
"""


def _clear_v3(monkeypatch) -> None:
    monkeypatch.delenv("C2HLS_DSE_V3", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V3_HARNESS", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V2_EXTRA_USER", raising=False)


def _trial_32x32() -> v2.DseV2Trial:
    return v2.DseV2Trial(
        pe=32,
        simd=32,
        i=64,
        j=64,
        k=64,
        data_kind="float",
        expected_dsp=5120,
        min_dsp=300,
        max_dsp=9024,
        pe_kj=(64 // 32) * 64,
        i_tiles=64 // 32,
        pack_bits=32 * 32,
    )


def _user(monkeypatch, trial: v2.DseV2Trial | None = None) -> tuple[str, dict]:
    trial = trial or _trial_32x32()
    skills_block, meta = v2.build_dse_v2_skills_prompt_block(trial)
    user = v2.format_dse_v2_initial_user(
        trial=trial,
        skills_block=skills_block,
        header_name="kernel.h",
        header_code=HEADER_64,
        kernel_code='extern "C" void autosa_mm() {}',
        synth_summary="- latency_cycles: 1443073\n- DSP: 2592\n",
        bench="autosa_mm",
    )
    return user, meta


def _row(trial_id: str, latency: int, coverage: str, *, dsp: int = 5184, lut: int = 10) -> dict:
    return {
        "legal": True,
        "latency_cycles": latency,
        "dsp": dsp,
        "lut": lut,
        "trial_id": trial_id,
        "coverage": coverage,
        "mac_count": 64 * 64 * 64,
    }


def _load_report(trip: int) -> dict:
    return {
        "feedback": {
            "scopes": [
                {
                    "kind": "loop",
                    "name": "load_B",
                    "trip_count": trip,
                    "pipelined": "yes",
                }
            ]
        }
    }


KERNEL_512 = """
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512
"""


def test_v3_skill_file_keeps_four_v2_entries(monkeypatch):
    _clear_v3(monkeypatch)
    v2_data = json.loads(POST_FLASH_DSE_SKILL_ENTRIES_JSON.read_text(encoding="utf-8"))
    v3_data = json.loads(POST_FLASH_DSE_V3_SKILL_ENTRIES_JSON.read_text(encoding="utf-8"))
    v2_by = {entry["id"]: entry for entry in v2_data["skills"]}
    v3_ids = [entry["id"] for entry in v3_data["skills"]]
    assert v3_ids == [*v2.V2_SKILL_IDS, v3.V3_SKILL_ID]
    assert "hls-dse-pick-pe-simd-for-kernel" not in v3_ids
    for sid in v2.V2_SKILL_IDS:
        assert v3_data["skills"][v3_ids.index(sid)] == v2_by[sid]
    new = v3_data["skills"][-1]
    assert new["id"] == "hls-dse-wide-512-load-store-hoist-b"
    for line in v3.V3_RULE_LINES:
        assert line in new["strategy"]


def test_v2_prompt_omits_v3_rules_when_unset(monkeypatch):
    _clear_v3(monkeypatch)
    user, meta = _user(monkeypatch)
    assert meta["skill_ids"] == list(v2.V2_SKILL_IDS)
    assert v3.V3_SKILL_ID not in user
    assert "512-bit beats" not in user
    assert "B[j0][k] is outside i0" not in user
    assert "Do not add a K loop the seed does not have" not in user
    assert "Do not raise PE or SIMD to shorten a scalar load" not in user
    assert v2.dse_v2_system_prompt() == v2._SYSTEM_V2
    assert v2.dse_v3_enabled() is False
    assert v2.dse_v3_harness_enabled() is False


def test_v3_prompt_has_v2_skills_and_new_rules(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3", "1")
    user, meta = _user(monkeypatch)
    assert meta["skill_ids"][:4] == list(v2.V2_SKILL_IDS)
    assert meta["skill_ids"][-1] == v3.V3_SKILL_ID
    assert "hls-dse-pick-pe-simd-for-kernel" not in user
    for sid in v2.V2_SKILL_IDS:
        assert sid in user
    assert v3.V3_SKILL_ID in user
    assert "512-bit beats (16 floats)" in user
    assert "B[j0][k] is outside i0" in user
    assert "Do not add a K loop the seed does not have in order to cut latency." in user
    assert "Do not raise PE or SIMD to shorten a scalar load." in user
    assert "Reuse a B tile across i0." in user
    assert "Do not stream B once per PE-row group across all of J and K." in user
    assert "A dataflow region of the three tile stages is allowed." in user
    assert "Do not emit AutoSA kernel0 or a C for-loop of mm_pe inside DATAFLOW." in user
    assert "max_widen_bitwidth=512" in user
    # avoid-dse-k-recurrence stays: pipeline j, unroll the SIMD adder tree.
    assert "Move PIPELINE to independent j" in user
    assert "unrolled SIMD adder tree" in user
    assert "512-bit beats" in v2.dse_v2_system_prompt()
    assert v2.dse_v3_harness_enabled() is False


def test_v3_extra_user_still_appends_after_skills(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3", "1")
    extra = "Aim for about 5184 DSP."
    monkeypatch.setenv("C2HLS_DSE_V2_EXTRA_USER", extra)
    user, _meta = _user(monkeypatch)
    assert "## Additional goals" in user
    assert extra in user
    for sid in (*v2.V2_SKILL_IDS, v3.V3_SKILL_ID):
        assert user.index(sid) < user.index("## Additional goals")
    assert user.index("Do not add a K loop the seed does not have") < user.index(
        "## Additional goals"
    )


def test_harness_implies_v3_prompt_and_csynth_tcl(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3_HARNESS", "1")
    monkeypatch.setenv("C2HLS_VITIS_JOBS", "8")
    assert v3.dse_v3_enabled() is True
    assert v3.dse_v3_harness_enabled() is True
    user, meta = _user(monkeypatch)
    assert v3.V3_SKILL_ID in meta["skill_ids"]
    assert "B[j0][k] is outside i0" in user
    assert "Do not add a K loop the seed does not have" in user
    pre = v3.harness_csynth_preamble()
    assert "config_array_partition -complete_threshold 0" in pre
    assert "config_compile -jobs" not in pre
    assert v3.harness_csynth_kwargs()["allow_compile_jobs"] is False


def test_v3_without_harness_keeps_jobs_and_does_not_reject(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3", "1")
    monkeypatch.setenv("C2HLS_VITIS_JOBS", "8")
    assert v3.dse_v3_harness_enabled() is False
    assert v3.harness_csynth_kwargs() == {}
    assert csynth_pre_commands() == "config_compile -jobs 8\n"
    assert "complete_threshold" not in csynth_pre_commands()
    assert csynth_pre_commands() == _config_compile_jobs_tcl()
    err = v3.harness_post_csynth_error(
        kernel_code=KERNEL_512,
        report=_load_report(4096),
        tile_rows=64,
        tile_cols=64,
    )
    assert err == ""


def test_scalar_load_b_trip_on_64x64_tile(monkeypatch):
    elements = 64 * 64
    assert v3.scalar_element_trip_illegal(interface_bits=512, trip=4096, elements=elements)
    assert not v3.scalar_element_trip_illegal(interface_bits=512, trip=256, elements=elements)
    assert not v3.scalar_element_trip_illegal(interface_bits=32, trip=4096, elements=elements)
    err = v3.scalar_element_trip_error(
        interface_bits=512, trip=4096, elements=elements, loop_name="load_B"
    )
    assert "illegal scalar load_B" in err
    assert "4096" in err
    assert "256" in err
    assert (
        v3.scalar_element_trip_error(
            interface_bits=512, trip=256, elements=elements, loop_name="load_B"
        )
        == ""
    )

    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3_HARNESS", "1")
    rejected = v3.harness_post_csynth_error(
        kernel_code=KERNEL_512,
        report=_load_report(4096),
        tile_rows=64,
        tile_cols=64,
    )
    assert "load_B" in rejected
    assert "4096" in rejected
    legal = v3.harness_post_csynth_error(
        kernel_code=KERNEL_512,
        report=_load_report(256),
        tile_rows=64,
        tile_cols=64,
    )
    assert legal == ""


def test_coverage_tag_partial_vs_full_k():
    partial = v3.assess_k_coverage(PARTIAL_TILE, i=256, j=256, k=256)
    assert partial["coverage"] == "partial_k"
    assert partial["full_gemm"] is False
    assert partial["mac_count"] == 64 * 64 * 64

    full = v3.assess_k_coverage(FULL_K, i=64, j=64, k=64)
    assert full["coverage"] == "full_k"
    assert full["full_gemm"] is True
    assert full["mac_count"] == 64 * 64 * 64

    assert v3.classify_k_coverage(FULL_K_OUTER) == "full_k"
    unknown = v3.assess_k_coverage("for (int j = 0; j < J; ++j) {}", i=64, j=64, k=64)
    assert unknown["coverage"] == "unknown"
    assert unknown["mac_count"] is None
    assert unknown["full_gemm"] is False


def test_winner_grouping_does_not_mix_coverage_classes():
    partial = _row("partial", 236497, "partial_k")
    full_slower = _row("full_slow", 1459105, "full_k")
    out = v2.select_harness_winner([partial, full_slower])
    assert out["winner"]["trial_id"] == "partial"
    assert out["winner"]["coverage"] == "partial_k"
    assert out["winner"]["full_gemm"] is False
    assert out["by_coverage"]["full_k"]["trial_id"] == "full_slow"
    assert out["by_coverage"]["full_k"]["latency_cycles"] > out["winner"]["latency_cycles"]

    full_faster = _row("full_fast", 1000, "full_k")
    mixed = v2.select_harness_winner([partial, full_faster])
    assert v2.select_winner([partial, full_faster])["trial_id"] == "full_fast"
    assert mixed["winner"]["trial_id"] == "partial"
    assert mixed["winner"]["full_gemm"] is False
    assert mixed["by_coverage"]["full_k"]["trial_id"] == "full_fast"
    assert mixed["by_coverage"]["partial_k"]["full_gemm"] is False

    seeded = v2.select_harness_winner(
        [partial, full_faster], seed_coverage="full_k"
    )
    assert seeded["winner"]["trial_id"] == "full_fast"
    assert seeded["winner"]["full_gemm"] is True
    assert seeded["by_coverage"]["partial_k"]["full_gemm"] is False


def test_best_latency_inside_one_coverage_class():
    rows = [
        _row("p_slow", 300, "partial_k", dsp=100),
        _row("p_fast_hi_dsp", 200, "partial_k", dsp=500),
        _row("p_tie", 200, "partial_k", dsp=100, lut=9),
        _row("full", 50, "full_k"),
        {"legal": False, "latency_cycles": 1, "dsp": 1, "lut": 1, "trial_id": "bad", "coverage": "partial_k"},
    ]
    out = v2.select_harness_winner(rows, seed_coverage="partial_k")
    assert out["winner"]["trial_id"] == "p_tie"
    assert out["winner"]["full_gemm"] is False
    assert out["by_coverage"]["full_k"]["trial_id"] == "full"


def test_float_pair_floor_is_nine_tenths_of_expected_dsp():
    # floor(5 * PE * SIMD * 0.9) via (5 * pe * simd * 9) // 10.
    # 5*1*1*0.9 = 4.5 truncates toward zero to 4.
    assert v3.float_pair_min_dsp(64, 4) == 1152
    assert v3.float_pair_min_dsp(16, 32) == 2304
    assert v3.float_pair_min_dsp(32, 32) == 4608
    assert v3.float_pair_min_dsp(128, 8) == 4608
    assert v3.float_pair_min_dsp(8, 8) == 288
    assert v3.float_pair_min_dsp(1, 1) == 4


def test_v3_trial_floor_follows_pe_simd_and_keeps_grid_band(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3", "1")
    detailed = v2.expand_dse_v2_trials_detailed(i=128, j=128, k=128, data_kind="float")
    by_id = {(t.pe, t.simd): t for t in detailed}
    assert by_id[(64, 4)].legal_for_prompt
    assert by_id[(64, 4)].expected_dsp == 1280
    assert by_id[(64, 4)].min_dsp == 1152
    assert by_id[(64, 4)].max_dsp == 9024
    assert by_id[(16, 32)].min_dsp == 2304
    assert by_id[(128, 8)].legal_for_prompt
    assert by_id[(128, 8)].expected_dsp == 5120
    assert by_id[(128, 8)].min_dsp == 4608
    # Membership band still uses expected DSP against the grid 300 / 9024.
    assert by_id[(8, 4)].expected_dsp == 160
    assert "min_dsp=300" in by_id[(8, 4)].skip_reason
    assert by_id[(128, 32)].expected_dsp == 20480
    assert "max_dsp=9024" in by_id[(128, 32)].skip_reason
    i64 = v2.expand_dse_v2_trials(i=64, j=64, k=64, data_kind="float")
    assert (64, 4) in {(t.pe, t.simd) for t in i64}
    assert next(t for t in i64 if (t.pe, t.simd) == (64, 4)).min_dsp == 1152
    assert v2.architecture_ok_v2({"dsp": 1151}, by_id[(64, 4)]) is False
    assert v2.architecture_ok_v2({"dsp": 1152}, by_id[(64, 4)]) is True
    assert v2.architecture_ok_v2({"dsp": 2303}, by_id[(16, 32)]) is False
    assert v2.architecture_ok_v2({"dsp": 4608}, by_id[(128, 8)]) is True


def test_v3_harness_alone_uses_the_same_pair_floor(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3_HARNESS", "1")
    trial = next(
        t
        for t in v2.expand_dse_v2_trials(i=64, j=64, k=64, data_kind="float")
        if t.pe == 64 and t.simd == 4
    )
    assert trial.min_dsp == 1152
    assert trial.max_dsp == 9024


def test_v3_prompt_states_64x4_floor_not_300(monkeypatch):
    _clear_v3(monkeypatch)
    monkeypatch.setenv("C2HLS_DSE_V3", "1")
    trial = next(
        t
        for t in v2.expand_dse_v2_trials(i=64, j=64, k=64, data_kind="float")
        if t.pe == 64 and t.simd == 4
    )
    assert trial.min_dsp == 1152
    user, _meta = _user(monkeypatch, trial)
    assert "reject csynth if DSP < 1152" in user
    assert "reject if DSP < 1152" in user
    assert "DSP < 300" not in user


def test_v2_prompt_keeps_grid_floor_when_v3_unset(monkeypatch):
    _clear_v3(monkeypatch)
    trial = next(
        t
        for t in v2.expand_dse_v2_trials(i=64, j=64, k=64, data_kind="float")
        if t.pe == 64 and t.simd == 4
    )
    assert trial.min_dsp == 300
    assert trial.expected_dsp == 1280
    user, _meta = _user(monkeypatch, trial)
    assert "reject csynth if DSP < 300" in user
    assert "1152" not in user
    assert v2.architecture_ok_v2({"dsp": 299}, trial) is False
    assert v2.architecture_ok_v2({"dsp": 300}, trial) is True
