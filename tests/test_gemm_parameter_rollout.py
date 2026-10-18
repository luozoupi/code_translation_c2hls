from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_gemm_parameter_rollout as rollout  # noqa: E402


def _step_record(
    value: int, cycles: int | None, *, parent: bool = False
) -> dict:
    feasible = cycles is not None
    return {
        "case_id": "gemm_pipeline_compute_ii",
        "benchmark": "hlsfactory_gemm",
        "parameter_value": value,
        "parameter_label": str(value),
        "is_parent": parent,
        "candidate_id": "frozen_parent" if parent else f"ii{value}",
        "status": "feasible" if feasible else "csynth_failed",
        "feasible": feasible,
        "csim_status": "passed",
        "latency_cycles_worst": cycles,
        "interval": cycles + 1 if cycles is not None else None,
        "estimated_clock_period_ns": 2.5 if feasible else None,
        "achieved_pipeline_ii_max": value if feasible else None,
        "dsp": 4 if feasible else None,
        "bram": 3 if feasible else None,
        "lut": 20 if feasible else None,
        "ff": 30 if feasible else None,
        "uram": 0 if feasible else None,
    }


def test_normalize_step_rows_uses_its_own_frozen_parent() -> None:
    rows = rollout._normalize_step_rows([
        _step_record(1, 80),
        _step_record(2, 100, parent=True),
        _step_record(4, None),
    ])
    winner = next(row for row in rows if row["parameter_value"] == 1)
    invalid = next(row for row in rows if row["parameter_value"] == 4)
    assert winner["latency_cycles_worst_ratio"] == 0.8
    assert winner["speedup_vs_parent"] == 1.25
    assert invalid["latency_cycles_worst_ratio"] is None
    assert min(rows, key=rollout._selection_key) is winner


def test_lane_rollout_only_marks_striped_family_primary() -> None:
    lane_records = []
    for architecture in ("direct_reduction", "striped_accumulators"):
        for lanes, cycles in ((1, 200), (2, 100)):
            lane_records.append({
                "architecture": architecture,
                "architecture_label": architecture,
                "requested_lanes": lanes,
                "effective_lanes": lanes,
                "is_family_parent": lanes == 2,
                "candidate_id": f"{architecture}-{lanes}",
                "csim_status": "passed",
                "feasible": True,
                "latency_cycles_worst": cycles,
                "interval": cycles + 1,
                "estimated_clock_period_ns": 2.5,
                "achieved_pipeline_ii_max": 1,
                "dsp": lanes,
                "bram": 1,
                "lut": 2,
                "ff": 3,
                "uram": 0,
            })
    rows = rollout._normalize_lane_rows(lane_records)
    primary = [row for row in rows if row["included_in_step_comparison"]]
    assert len(primary) == 2
    assert {row["architecture"] for row in primary} == {"striped_accumulators"}
    assert all(row["speedup_vs_parent"] is not None for row in rows)


def test_selection_tie_break_matches_campaign_resource_sum() -> None:
    fabric = {
        "candidate_id": "fabric",
        "feasible": True,
        "latency_cycles_worst": 100,
        "dsp": 10,
        "bram": 10,
        "lut": 100,
        "ff": 100,
        "uram": 0,
    }
    dsp = {
        **fabric,
        "candidate_id": "dsp",
        "dsp": 20,
        "lut": 50,
        "ff": 50,
    }
    assert min((fabric, dsp), key=rollout._selection_key) is dsp


def test_step_summary_sorts_incrementally_appended_numeric_values() -> None:
    rows = rollout._normalize_step_rows([
        _step_record(1, 120),
        _step_record(2, 100, parent=True),
        _step_record(8, 80),
        _step_record(4, 90),
    ])
    grouped = {"pipeline_ii": rows}
    original = rollout._primary_groups
    try:
        rollout._primary_groups = lambda _: grouped
        summary = rollout._step_summaries(rows)[0]
    finally:
        rollout._primary_groups = original
    assert summary["tested_values"] == ["1", "2", "4", "8"]
