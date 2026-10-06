"""Generic flash may use DATAFLOW when it is simple; it must not mandate GEMM ping-pong."""
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


def test_flash_prompt_keeps_dataflow_optional():
    blob = (q_optimize_flash + "\n" + Instruction_c2hls_flash).lower()
    assert "often beat complex tiling/dataflow" in blob
    assert "dataflow only when buffering stays simple" in blob
    assert "only when the added buffering" in blob
    assert "must overlap" not in q_optimize_flash.lower()
    assert "inline-off load_tile" not in blob


def test_lcst_skill_establishes_baseline_before_dataflow():
    for path in _PACKS:
        skills = json.loads(path.read_text(encoding="utf-8"))["skills"]
        sk = next(s for s in skills if s["id"] == "hls-baseline-load-compute-store-gate")
        strat = sk["strategy"].lower()
        assert "before any tiling, unrolling, coalescing, or dataflow" in strat
        assert "PE_BLK" not in json.dumps(sk)


def test_dataflow_stage_split_is_baseline_first():
    assert "hls-doublebuffer-dataflow-stage-split" in _BASELINE_FIRST_SKILL_IDS
    assert _BASELINE_FIRST_SKILL_IDS.index(
        "hls-doublebuffer-dataflow-stage-split"
    ) < _BASELINE_FIRST_SKILL_IDS.index("hls-prefer-full-workspace-staging-when-fits")
