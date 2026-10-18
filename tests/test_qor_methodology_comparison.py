from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import compare_qor_methodology_benchmarks as comparison  # noqa: E402


def _summary(benchmark: str, ratio: float) -> dict:
    cases = []
    for step_id in comparison.STEP_ORDER:
        case_id = comparison.CASE_MAP[benchmark][step_id]
        cases.append({
            "case_id": case_id,
            "parent_value": "parent",
            "best_parameter_value": "best",
            "parent_cycles": 100,
            "best_cycles": 100 * ratio,
            "candidate_count": 3,
            "feasible_candidate_count": 3,
            "monotonicity_violations": 1,
            "winner_candidate_id": "winner",
        })
    return {"cases": cases}


def test_comparison_normalizes_each_benchmark_to_its_own_parent() -> None:
    rows = comparison._comparison_rows(
        _summary("gemm", 0.5),
        _summary("2mm", 0.8),
    )
    assert len(rows) == 6
    assert rows[0]["gemm_best_cycle_ratio"] == 0.5
    assert rows[0]["gemm_speedup_vs_parent"] == 2.0
    assert rows[0]["2mm_best_cycle_ratio"] == 0.8
    assert rows[0]["2mm_speedup_vs_parent"] == 1.25


def test_gemm_lane_case_uses_the_complete_consolidated_domain() -> None:
    rows = []
    for value, cycles in ((1, 120), (2, 100), (4, 110), (8, 50)):
        rows.append({
            "step_id": "parallel_lanes",
            "included_in_step_comparison": True,
            "is_parent": value == 2,
            "feasible": True,
            "parameter_numeric": value,
            "parameter_label": str(value),
            "latency_cycles_worst": cycles,
            "candidate_id": "frozen_parent" if value == 2 else f"lane-{value}",
            "dsp": value,
            "bram": value,
            "lut": value,
            "ff": value,
            "uram": value,
        })
    case = comparison._gemm_lane_case(rows)
    assert case["candidate_count"] == 3
    assert case["feasible_candidate_count"] == 3
    assert case["best_parameter_value"] == "8"
    assert case["best_cycles"] == 50
    assert case["monotonicity_violations"] == 1
