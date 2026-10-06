"""Default flash 90-skills and FLASH MODE must stay kernel-independent.

AutoSA mm recipes (LANES=16 / PE_BLK / 64x64 ping-pong) belong in
gemm_family, systolic_io, onchip, or C2HLS_FLASH_* knobs — not in the
generic dump or q_optimize_flash. Frozen mmflow flash 139484/10 used
these payloads.
"""
from __future__ import annotations

import json
from pathlib import Path

from prompt_c2hls import Instruction_c2hls_flash, q_optimize_flash

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
_PACKS = (
    PKG / "skills_ii_target_miss_solutions_added(90skills).json",
    PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json",
    PKG / "skills_ii_target_miss_solutions_added(90skills)_no_RMW_m_axi.json",
)
_CONTAM = (
    "PE_BLK",
    "64x64",
    "64 x 64",
    "LANES=16 so 64x64",
    "const int LANES = 16",
    "const int PE = 16",
    "j += LANES",
    "9024 DSP",
)


def test_flash_prompt_is_modest_generic_hls():
    blob = q_optimize_flash + "\n" + Instruction_c2hls_flash
    assert "usually 2, 4, 8, 16, 32, 64, or 128" in q_optimize_flash
    assert "DATAFLOW only when buffering stays simple" in Instruction_c2hls_flash
    assert "often beat complex tiling/dataflow" in q_optimize_flash
    assert "only when the added buffering" in q_optimize_flash
    low = blob.lower()
    assert "pe_blk must be 16" not in low
    assert "ping-pong of **tiles**" not in q_optimize_flash.lower()
    assert "g >> k" not in q_optimize_flash
    assert "old + dot" not in q_optimize_flash


def test_baseline_lcst_is_scalar_then_advanced():
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "hls-baseline-load-compute-store-gate")
        strat = sk["strategy"].lower()
        assert "before any tiling, unrolling, coalescing, or dataflow" in strat
        blob = json.dumps(sk)
        for needle in _CONTAM:
            assert needle not in blob, f"{path.name} baseline still has {needle!r}"
        tmpl = sk["template"]
        assert "LANES" not in tmpl
        assert "tile_loop" not in tmpl
        assert "local_in[i][j]=in[i][j]" in tmpl.replace(" ", "")


def test_mandatory_load_store_is_pipeline_only():
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "hls-mandatory-pipeline-load-store-loops")
        assert "LANES" not in (sk.get("template") or "")
        assert "4096" not in (sk.get("strategy") or "")
        assert "PE_BLK" not in json.dumps(sk)


def test_autosa_bins_stay_out_of_90_dump():
    for path in _PACKS:
        ids = {s["id"] for s in json.loads(path.read_text(encoding="utf-8"))["skills"]}
        assert "hls-sysio-dataflow-pe-tasks" not in ids
        assert "hls-enforcement-keep-flash-wide-load" not in ids
        assert "hls-enforcement-wrap-flash-with-tile-dataflow" not in ids
