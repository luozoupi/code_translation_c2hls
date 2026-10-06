"""Wide AXI is opt-in. Default 90-skills stay generic LCST."""
from __future__ import annotations

import json
from pathlib import Path

from prompt_c2hls import Instruction_c2hls_flash, q_optimize_flash
from skill_library import _BASELINE_FIRST_SKILL_IDS

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
_PACKS = (
    PKG / "skills_ii_target_miss_solutions_added(90skills).json",
    PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json",
    PKG / "skills_ii_target_miss_solutions_added(90skills)_no_RMW_m_axi.json",
)


def test_default_flash_prompt_does_not_mandate_lanes_64x64():
    blob = (q_optimize_flash + "\n" + Instruction_c2hls_flash).lower()
    assert "usually 2, 4, 8, 16, 32, 64, or 128" in q_optimize_flash
    assert "4096" not in q_optimize_flash
    assert "max_widen_bitwidth=512" not in q_optimize_flash
    assert "scalar ii=1" not in blob
    assert "cut compute ii to 1" not in blob
    assert "g >> k" not in q_optimize_flash
    assert "old + dot" not in q_optimize_flash
    assert "j += pe" not in blob


def test_lcst_template_is_scalar_copy_not_lanes():
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "hls-baseline-load-compute-store-gate")
        tmpl = sk["template"]
        assert "LANES" not in tmpl
        assert "j += LANES" not in tmpl
        assert "const int PE = 16" not in tmpl
        assert "local_in[i][j]" in tmpl
        steps = " ".join(sk["required_steps"]).lower()
        assert "pe_blk" not in steps
        assert "4096" not in steps
        assert "lanes" not in sk["strategy"].lower()


def test_mandatory_load_store_pipeline_is_pragma_only():
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "hls-mandatory-pipeline-load-store-loops")
        assert "LANES" not in (sk.get("template") or "")
        assert "4096" not in (sk.get("strategy") or "")


def test_coalescing_skill_keeps_narrow_abi_and_compute_lanes():
    """HEAD coalescing already mentions LANES for compute; not 64x64 GEMM I/O."""
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "axi-burst-coalescing-narrow-safe")
        low = sk["strategy"].lower()
        assert "max_widen_bitwidth=512" in low or "512" in low
        assert "scalar pointer" in low or "public scalar" in low
        assert "64x64" not in json.dumps(sk)
        assert "PE_BLK" not in json.dumps(sk)


def test_wide_io_coalescing_is_baseline_first():
    assert "axi-burst-coalescing-narrow-safe" in _BASELINE_FIRST_SKILL_IDS
    assert _BASELINE_FIRST_SKILL_IDS.index(
        "axi-burst-coalescing-narrow-safe"
    ) < _BASELINE_FIRST_SKILL_IDS.index("hls-prefer-full-workspace-staging-when-fits")
