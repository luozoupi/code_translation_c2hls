"""Tests for DSE 2.0 PE×SIMD grid sweep (no locked 16×4 recipe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_dse as pfd
import post_flash_dse_v2 as v2
from post_flash_pe_recipe import format_recipe_prompt


HEADER_64 = """
typedef float data_t;
#define I 64
#define J 64
#define K 64
"""


def test_grid_file_exists_and_defaults():
    path = v2.resolve_dse_v2_grid_path()
    assert path.is_file(), path
    grid = v2.load_dse_v2_grid()
    assert grid["pe"]["min"] == 8
    assert grid["pe"]["max"] == 128
    assert grid["simd"]["min"] == 1
    assert grid["simd"]["max"] == 32
    assert grid["min_dsp"] == 300
    assert grid["max_dsp"] == 9024


def test_expand_17_promptable_trials_for_ijk64():
    trials = v2.expand_dse_v2_trials(i=64, j=64, k=64, data_kind="float")
    ids = {(t.pe, t.simd) for t in trials}
    assert len(trials) == 17, sorted(ids)
    # Skipped by plan
    assert (128, 4) not in ids
    assert (8, 4) not in ids  # DSP 160 < 300
    assert (8, 1) not in ids  # DSP 40 < 300
    assert (8, 2) not in ids  # DSP 80 < 300
    assert (16, 1) not in ids
    assert (16, 2) not in ids
    assert (32, 1) not in ids
    assert (64, 32) not in ids  # DSP 10240 > 9024
    # Present
    assert (8, 8) in ids
    assert (16, 4) in ids
    assert (32, 2) in ids
    assert (64, 1) in ids
    assert (64, 2) in ids
    expected = {
        (8, 8),
        (8, 16),
        (8, 32),
        (16, 4),
        (16, 8),
        (16, 16),
        (16, 32),
        (32, 2),
        (32, 4),
        (32, 8),
        (32, 16),
        (32, 32),
        (64, 1),
        (64, 2),
        (64, 4),
        (64, 8),
        (64, 16),
    }
    assert ids == expected


def test_detailed_records_skip_reasons():
    detailed = v2.expand_dse_v2_trials_detailed(i=64, j=64, k=64, data_kind="float")
    by_id = {(t.pe, t.simd): t for t in detailed}
    assert "does not divide I" in by_id[(128, 4)].skip_reason
    assert "min_dsp" in by_id[(8, 4)].skip_reason
    assert "max_dsp" in by_id[(64, 32)].skip_reason
    assert by_id[(16, 4)].legal_for_prompt
    assert by_id[(16, 4)].expected_dsp == 320
    assert by_id[(32, 8)].expected_dsp == 1280
    assert by_id[(64, 1)].legal_for_prompt
    assert by_id[(64, 1)].expected_dsp == 320
    assert by_id[(32, 2)].expected_dsp == 320
    assert "min_dsp" in by_id[(8, 1)].skip_reason
    assert "min_dsp" in by_id[(32, 1)].skip_reason


def test_only_simd_env_keeps_legal_simd_1_and_2(monkeypatch):
    monkeypatch.setenv("C2HLS_DSE_V2_ONLY_SIMD", "1,2")
    trials = v2.expand_dse_v2_trials(i=64, j=64, k=64, data_kind="float")
    ids = {(t.pe, t.simd) for t in trials}
    assert ids == {(32, 2), (64, 1), (64, 2)}
    detailed = v2.expand_dse_v2_trials_detailed(i=64, j=64, k=64, data_kind="float")
    by_id = {(t.pe, t.simd): t for t in detailed}
    assert by_id[(64, 1)].legal_for_prompt
    assert by_id[(32, 2)].legal_for_prompt
    assert "C2HLS_DSE_V2_ONLY_SIMD" in by_id[(64, 4)].skip_reason


def test_parse_ijk_and_data_kind():
    assert v2.parse_ijk_from_header(HEADER_64) == (64, 64, 64)
    assert v2.detect_data_kind(HEADER_64) == "float"
    cat = "#define I_P 64\n#define J_P 64\n#define K_P 64\ntypedef unsigned int data_t;\n"
    assert v2.parse_ijk_from_header(cat) == (64, 64, 64)
    assert v2.detect_data_kind(cat) == "uint32"


def test_prompt_32x8_has_trial_numbers_no_locked_leak(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_V3", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V3_HARNESS", raising=False)
    trial = next(
        t
        for t in v2.expand_dse_v2_trials(i=64, j=64, k=64)
        if t.pe == 32 and t.simd == 8
    )
    skills_block, meta = v2.build_dse_v2_skills_prompt_block(trial)
    assert "hls-dse-pick-pe-simd-for-kernel" not in meta["skill_ids"]
    assert set(v2.V2_SKILL_IDS) <= set(meta["skill_ids"])
    user = v2.format_dse_v2_initial_user(
        trial=trial,
        skills_block=skills_block,
        header_name="kernel.h",
        header_code=HEADER_64,
        kernel_code='extern "C" void autosa_mm() {}',
        synth_summary="- latency_cycles: 95500\n- DSP: 12\n",
        bench="autosa_mm",
    )
    assert "PE_NUM=32" in user
    assert "SIMD=8" in user
    assert "expected DSP ≈ 1280" in user
    assert "#define PE 32" in user
    assert "#define SIMD 8" in user
    assert "instantiate 32 explicit mm_pe()" in user
    assert "PE_NUM × SIMD × 5" in user or "PE_NUM × SIMD × 5" in user.replace("×", "x")
    # Formulas present
    assert "reject csynth if DSP < 300" in user
    # Leak guards
    low = user.lower()
    assert "locked mm: 16x4" not in low
    assert "getting_started" not in low
    assert "keep pe=16 simd=4" not in low
    assert "autosa_mm_catapult" not in low
    # System similarly: no 320-at-16x4 claim
    assert "320 at 16x4" not in v2._SYSTEM_V2.lower()
    assert "yields 320" not in v2._SYSTEM_V2.lower()


def test_prompt_16x4_trial_may_say_16_and_4_but_has_formulas():
    trial = next(
        t
        for t in v2.expand_dse_v2_trials(i=64, j=64, k=64)
        if t.pe == 16 and t.simd == 4
    )
    skills_block, _ = v2.build_dse_v2_skills_prompt_block(trial)
    user = v2.format_dse_v2_initial_user(
        trial=trial,
        skills_block=skills_block,
        header_name="kernel.h",
        header_code=HEADER_64,
        kernel_code="// seed",
        synth_summary="- latency_cycles: 1\n",
        bench="autosa_mm",
    )
    assert "PE_NUM=16" in user
    assert "SIMD=4" in user
    assert "There is **no** locked PE×SIMD" in user
    assert "Formulas (same on every trial)" in user
    assert "locked mm: 16x4" not in user.lower()


def test_select_winner_lowest_latency_then_dsp_lut():
    results = [
        {
            "legal": True,
            "latency_cycles": 9000,
            "dsp": 320,
            "lut": 100,
            "trial_id": "a",
        },
        {
            "legal": True,
            "latency_cycles": 5000,
            "dsp": 640,
            "lut": 200,
            "trial_id": "b",
        },
        {
            "legal": False,
            "latency_cycles": 1000,
            "dsp": 320,
            "lut": 50,
            "trial_id": "bad",
        },
        {
            "legal": True,
            "latency_cycles": 5000,
            "dsp": 320,
            "lut": 90,
            "trial_id": "c",
        },
    ]
    winner = v2.select_winner(results)
    assert winner is not None
    assert winner["trial_id"] == "c"  # same lat as b, lower DSP


def test_select_winner_excludes_low_dsp_and_failed_csim():
    results = [
        {"legal": False, "latency_cycles": 100, "dsp": 10, "lut": 1, "trial_id": "fail"},
        {"legal": True, "latency_cycles": 8000, "dsp": 320, "lut": 10, "trial_id": "ok"},
    ]
    assert v2.select_winner(results)["trial_id"] == "ok"
    assert v2.select_winner([{"legal": False, "latency_cycles": 1}]) is None


def test_architecture_ok_v2_uses_min_dsp_300(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_V3", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V3_HARNESS", raising=False)
    trial = next(t for t in v2.expand_dse_v2_trials(i=64, j=64, k=64) if t.pe == 16)
    assert trial.min_dsp == 300
    assert v2.architecture_ok_v2({"dsp": 299}, trial) is False
    assert v2.architecture_ok_v2({"dsp": 300}, trial) is True
    assert v2.architecture_ok_v2({"dsp": 320}, trial) is True


def test_v1_recipe_still_locked_16x4():
    prompt = format_recipe_prompt("autosa_mm", step="dse")
    assert "PE_NUM=16" in prompt
    assert "SIMD=4" in prompt
    assert "locked mm: 16x4" in prompt.lower() or "DSP~320" in prompt or "320" in prompt


def test_v1_maybe_chain_skipped_when_dse_v2(monkeypatch):
    monkeypatch.setenv("C2HLS_DSE_V2", "1")
    monkeypatch.setenv("C2HLS_POST_FLASH_DSE", "1")
    monkeypatch.setenv("C2HLS_DSE_CHAIN_FLASH", "1")
    called = {"n": 0}

    def boom(**_kwargs):
        called["n"] += 1
        raise AssertionError("v1 must not run")

    monkeypatch.setattr(pfd, "run_dse_for_cell", boom)
    out = pfd.maybe_chain_dse(
        bench="autosa_mm",
        bench_dir=Path("/tmp/bench"),
        cell_dir=Path("/tmp/cell"),
        orchestrator=object(),
    )
    assert out is None
    assert called["n"] == 0


def test_dse_v2_enabled_flags(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_V2", raising=False)
    monkeypatch.delenv("C2HLS_DSE_V2_CHAIN_FLASH", raising=False)
    assert v2.dse_v2_enabled() is False
    assert v2.chain_after_flash_v2() is False
    monkeypatch.setenv("C2HLS_DSE_V2", "1")
    assert v2.dse_v2_enabled() is True
    assert v2.chain_after_flash_v2() is True
    monkeypatch.setenv("C2HLS_DSE_V2_CHAIN_FLASH", "0")
    assert v2.chain_after_flash_v2() is False


def test_dropped_skill_not_in_v2_skills():
    skills = v2.load_dse_v2_skills()
    ids = {sk.id for sk in skills}
    assert "hls-dse-pick-pe-simd-for-kernel" not in ids
    assert "hls-dse-gemm-multi-pe-latency-hiding" in ids


def _trial_128x8() -> v2.DseV2Trial:
    return v2.DseV2Trial(
        pe=128,
        simd=8,
        i=1024,
        j=1024,
        k=1024,
        data_kind="float",
        expected_dsp=5120,
        min_dsp=5000,
        max_dsp=9024,
        pe_kj=(1024 // 8) * 1024,
        i_tiles=1024 // 128,
        pack_bits=8 * 32,
    )


def _user_for(trial: v2.DseV2Trial) -> str:
    skills_block, meta = v2.build_dse_v2_skills_prompt_block(trial)
    assert meta["skill_ids"] == list(v2.V2_SKILL_IDS)
    return v2.format_dse_v2_initial_user(
        trial=trial,
        skills_block=skills_block,
        header_name="kernel.h",
        header_code=HEADER_64,
        kernel_code='extern "C" void autosa_mm() {}',
        synth_summary="- latency_cycles: 1443073\n- DSP: 2592\n",
        bench="autosa_mm",
    )


def test_extra_user_defaults_off_and_keeps_prompt(monkeypatch):
    monkeypatch.delenv("C2HLS_DSE_V2_EXTRA_USER", raising=False)
    raw = "Repair the **DSE** kernel\n"
    assert v2.append_dse_v2_extra_user(raw) is raw
    assert v2.dse_v2_extra_user_text() == ""
    user = _user_for(_trial_128x8())
    assert "## Additional goals" not in user
    for sid in v2.V2_SKILL_IDS:
        assert sid in user
    assert "hls-dse-pick-pe-simd-for-kernel" not in user
    assert "PE_NUM=128" in user
    assert "SIMD=8" in user


def test_extra_user_appends_without_dropping_skills_or_adding_a_gate(monkeypatch):
    extra = (
        "Current design: 1,443,073 cycles, DSP 2592, PE 16, SIMD 32. "
        "Aim for about 5184 DSP and at least 5000 measured DSP. "
        "Latency at most 865844 cycles."
    )
    monkeypatch.setenv("C2HLS_DSE_V2_EXTRA_USER", extra)
    trial = _trial_128x8()
    user = _user_for(trial)
    assert "## Additional goals" in user
    assert extra in user
    for sid in v2.V2_SKILL_IDS:
        assert sid in user
        assert user.index(sid) < user.index("## Additional goals")
    assert "PE_NUM=128" in user
    repaired = v2.append_dse_v2_extra_user("Repair body\n")
    assert repaired.startswith("Repair body")
    assert extra in repaired
    assert v2.architecture_ok_v2({"dsp": 5000, "latency_cycles": 2000000}, trial) is True
    assert v2.architecture_ok_v2({"dsp": 4999, "latency_cycles": 1}, trial) is False
