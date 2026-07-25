#!/usr/bin/env python3
"""Unit tests for multistep cosim ranking / latency table helpers."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from multistep_batch_parallel_bench import (  # noqa: E402
    COSIM_FALLBACK_MAX_ATTEMPTS,
    build_latency_table,
    collect_cosim_candidates,
    lat_opt_improved_vs_seed,
    rank_cosim_candidates,
)


def _write(cell: Path, name: str, payload) -> None:
    path = cell / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_rank_cosim_candidates_prefers_lower_latency_then_later_post():
    cands = [
        {
            "id": "tiling:pre_lat_opt",
            "phase": "tiling",
            "variant": "pre_lat_opt",
            "phase_idx": 1,
            "latency_cycles": 100.0,
        },
        {
            "id": "tiling:post_lat_opt",
            "phase": "tiling",
            "variant": "post_lat_opt",
            "phase_idx": 1,
            "latency_cycles": 100.0,
        },
        {
            "id": "doublebuffer:post_lat_opt",
            "phase": "doublebuffer",
            "variant": "post_lat_opt",
            "phase_idx": 5,
            "latency_cycles": 80.0,
        },
        {
            "id": "phase_b:pre_lat_opt",
            "phase": "phase_b",
            "variant": "pre_lat_opt",
            "phase_idx": 0,
            "latency_cycles": 200.0,
        },
    ]
    ranked = rank_cosim_candidates(cands)
    assert [c["id"] for c in ranked] == [
        "doublebuffer:post_lat_opt",
        "tiling:post_lat_opt",
        "tiling:pre_lat_opt",
        "phase_b:pre_lat_opt",
    ]
    assert COSIM_FALLBACK_MAX_ATTEMPTS == 3
    assert [c["id"] for c in ranked[:3]] == [
        "doublebuffer:post_lat_opt",
        "tiling:post_lat_opt",
        "tiling:pre_lat_opt",
    ]


def test_collect_and_latency_table_from_cell_files():
    with tempfile.TemporaryDirectory() as tmp:
        cell = Path(tmp)
        bench = "gesummv"
        phases = ["phase_b", "tiling", "doublebuffer"]
        _write(cell, f"{bench}_multistep_phase_b.cpp", "phase_b code\n")
        _write(cell, f"{bench}_multistep_phase_b_report.json", {"latency_cycles": 1499})
        _write(cell, f"{bench}_multistep_phase_b_latency_opt.cpp", "phase_b opt\n")
        _write(
            cell,
            f"{bench}_multistep_phase_b_latency_opt_result.json",
            {"success": True, "latency_cycles": 615},
        )
        _write(
            cell,
            f"{bench}_multistep_phase_b_latency_opt_report.json",
            {"latency_cycles": 615},
        )
        _write(cell, f"{bench}_multistep_tiling.cpp", "tiling code\n")
        _write(cell, f"{bench}_multistep_tiling_report.json", {"latency_cycles": 698})
        # lat-opt skipped for tiling → only pre candidate
        _write(cell, f"{bench}_multistep_doublebuffer.cpp", "db code\n")
        _write(cell, f"{bench}_multistep_doublebuffer_report.json", {"latency_cycles": 172})
        _write(cell, f"{bench}_multistep_doublebuffer_latency_opt.cpp", "db opt\n")
        _write(
            cell,
            f"{bench}_multistep_doublebuffer_latency_opt_result.json",
            {"success": True, "latency_cycles": 80},
        )
        _write(
            cell,
            f"{bench}_multistep_doublebuffer_latency_opt_report.json",
            {"latency_cycles": 80},
        )

        cands = collect_cosim_candidates(cell_dir=cell, bench=bench, phases=phases)
        ids = {c["id"] for c in cands}
        assert "phase_b:pre_lat_opt" in ids
        assert "phase_b:post_lat_opt" in ids
        assert "tiling:pre_lat_opt" in ids
        assert "tiling:post_lat_opt" not in ids
        assert "doublebuffer:post_lat_opt" in ids

        ranked = rank_cosim_candidates(cands)
        assert ranked[0]["id"] == "doublebuffer:post_lat_opt"
        assert ranked[0]["latency_cycles"] == 80.0

        table = build_latency_table(cell_dir=cell, bench=bench, phases=phases)
        by_phase = {r["phase"]: r for r in table}
        assert by_phase["phase_b"]["pre_lat_opt_csynth_latency"] == 1499.0
        assert by_phase["phase_b"]["post_lat_opt_csynth_latency"] == 615.0
        assert by_phase["phase_b"]["lat_opt_improved"] is True
        assert by_phase["tiling"]["post_lat_opt_csynth_latency"] is None
        assert by_phase["tiling"]["lat_opt_ran"] is False
        assert by_phase["doublebuffer"]["post_lat_opt_csynth_latency"] == 80.0


def test_lat_opt_improved_vs_seed_strict():
    assert lat_opt_improved_vs_seed(13111, 25585) is False
    assert lat_opt_improved_vs_seed(13111, 13111) is False
    assert lat_opt_improved_vs_seed(13111, 9000) is True
    assert lat_opt_improved_vs_seed(None, 9000) is False


def test_collect_excludes_regressed_post_lat_opt():
    """Worse-than-seed lat-opt must not enter the ranked candidate pool."""
    with tempfile.TemporaryDirectory() as tmp:
        cell = Path(tmp)
        bench = "gemm"
        phases = ["unroll", "pipeline"]
        _write(cell, f"{bench}_multistep_unroll.cpp", "unroll seed\n")
        _write(cell, f"{bench}_multistep_unroll_report.json", {"latency_cycles": 13111})
        _write(cell, f"{bench}_multistep_unroll_latency_opt.cpp", "unroll worse\n")
        _write(
            cell,
            f"{bench}_multistep_unroll_latency_opt_result.json",
            {"success": True, "latency_cycles": 25585},
        )
        _write(
            cell,
            f"{bench}_multistep_unroll_latency_opt_report.json",
            {"latency_cycles": 25585},
        )
        _write(cell, f"{bench}_multistep_pipeline.cpp", "pipe seed\n")
        _write(cell, f"{bench}_multistep_pipeline_report.json", {"latency_cycles": 17175})

        cands = collect_cosim_candidates(cell_dir=cell, bench=bench, phases=phases)
        ids = {c["id"] for c in cands}
        assert "unroll:pre_lat_opt" in ids
        assert "unroll:post_lat_opt" not in ids
        assert "pipeline:pre_lat_opt" in ids
        ranked = rank_cosim_candidates(cands)
        assert ranked[0]["id"] == "unroll:pre_lat_opt"
        assert ranked[0]["latency_cycles"] == 13111.0

        table = build_latency_table(cell_dir=cell, bench=bench, phases=phases)
        by_phase = {r["phase"]: r for r in table}
        assert by_phase["unroll"]["lat_opt_success"] is True
        assert by_phase["unroll"]["lat_opt_improved"] is False


if __name__ == "__main__":
    test_rank_cosim_candidates_prefers_lower_latency_then_later_post()
    test_collect_and_latency_table_from_cell_files()
    test_lat_opt_improved_vs_seed_strict()
    test_collect_excludes_regressed_post_lat_opt()
    print("ok")
