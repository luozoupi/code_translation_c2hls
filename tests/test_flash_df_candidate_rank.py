#!/usr/bin/env python3
"""Unit tests for flash/dataflow candidate ranking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from flash_df_candidate_rank import (  # noqa: E402
    collect_dataflow_side_candidates,
    collect_flash_side_candidates,
    lat_opt_improved_vs_seed,
    promote_rank1_kernel,
    rank_and_promote,
    rank_candidates,
)


def _write(cell: Path, name: str, payload) -> None:
    path = cell / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_lat_opt_improved_strict() -> None:
    assert lat_opt_improved_vs_seed(100, 90) is True
    assert lat_opt_improved_vs_seed(100, 100) is False
    assert lat_opt_improved_vs_seed(100, 110) is False
    assert lat_opt_improved_vs_seed(None, 90) is False


def test_flash_seed_wins_when_lat_opt_worse(tmp_path: Path) -> None:
    bench = "hlsfactory_gemm"
    _write(tmp_path, f"{bench}_selected.cpp", "int selected;")
    _write(tmp_path, f"{bench}_selected_report.json", {"latency_cycles": 100})
    _write(tmp_path, f"{bench}_latency_opt.cpp", "int lat;")
    _write(tmp_path, f"{bench}_latency_opt_report.json", {"latency_cycles": 200})
    _write(tmp_path, f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 200})
    ranked = rank_candidates(collect_flash_side_candidates(tmp_path, bench))
    assert len(ranked) == 2
    assert ranked[0]["id"] == "flash:seed"
    assert ranked[0]["latency_cycles"] == 100.0
    assert ranked[1]["id"] == "flash:lat_opt"


def test_flash_lat_opt_wins_when_better(tmp_path: Path) -> None:
    bench = "hlsfactory_gemm"
    _write(tmp_path, f"{bench}_selected.cpp", "int selected;")
    _write(tmp_path, f"{bench}_selected_report.json", {"latency_cycles": 500})
    _write(tmp_path, f"{bench}_latency_opt.cpp", "int lat;")
    _write(tmp_path, f"{bench}_latency_opt_report.json", {"latency_cycles": 120})
    _write(tmp_path, f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 120})
    ranked = rank_candidates(collect_flash_side_candidates(tmp_path, bench))
    assert ranked[0]["id"] == "flash:lat_opt"
    assert ranked[0]["latency_cycles"] == 120.0


def test_flash_collects_accepted_lat_opt_rounds(tmp_path: Path) -> None:
    """Pool = seed ∪ each accepted round (final latency_opt.cpp is alias of best round)."""
    bench = "hlsfactory_gemm"
    _write(tmp_path, f"{bench}_flash_seed.cpp", "int seed;")
    _write(
        tmp_path,
        f"{bench}_flash_seed_report.json",
        {"latency_cycles": 1000, "latency_cycles_worst": 2000},
    )
    _write(tmp_path, f"{bench}_latency_opt_r1.cpp", "int r1;")
    _write(
        tmp_path,
        f"{bench}_latency_opt_r1_report.json",
        {"latency_cycles": 800, "latency_cycles_worst": 1500},
    )
    _write(tmp_path, f"{bench}_latency_opt_r3.cpp", "int r3;")
    _write(
        tmp_path,
        f"{bench}_latency_opt_r3_report.json",
        {"latency_cycles": 400, "latency_cycles_worst": 700},
    )
    # Final best mirrors r3 — must not double-count.
    _write(tmp_path, f"{bench}_latency_opt.cpp", "int r3;")
    _write(
        tmp_path,
        f"{bench}_latency_opt_report.json",
        {"latency_cycles": 400, "latency_cycles_worst": 700},
    )
    _write(tmp_path, f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 400})
    ranked = rank_candidates(collect_flash_side_candidates(tmp_path, bench))
    ids = [c["id"] for c in ranked]
    assert ids == ["flash:lat_opt_r3", "flash:lat_opt_r1", "flash:seed"]
    assert ranked[0]["latency_cycles"] == 700.0
    assert Path(ranked[0]["code_path"]).name == f"{bench}_latency_opt_r3.cpp"


def test_rank_uses_worst_case_not_best_case(tmp_path: Path) -> None:
    """min(max): prefer latency_cycles_worst over best-case latency_cycles."""
    bench = "hlsfactory_gemm"
    # Best-case would wrongly pick lat_opt (50 < 100); worst-case picks seed (200 < 900).
    _write(tmp_path, f"{bench}_flash_seed.cpp", "int seed;")
    _write(
        tmp_path,
        f"{bench}_flash_seed_report.json",
        {"latency_cycles": 100, "latency_cycles_worst": 200},
    )
    _write(tmp_path, f"{bench}_latency_opt.cpp", "int lat;")
    _write(
        tmp_path,
        f"{bench}_latency_opt_report.json",
        {"latency_cycles": 50, "latency_cycles_worst": 900},
    )
    _write(tmp_path, f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 50})
    ranked = rank_candidates(collect_flash_side_candidates(tmp_path, bench))
    assert ranked[0]["id"] == "flash:seed"
    assert ranked[0]["latency_cycles"] == 200.0
    assert ranked[1]["id"] == "flash:lat_opt"
    assert ranked[1]["latency_cycles"] == 900.0


def test_dataflow_pool_and_promote(tmp_path: Path) -> None:
    bench = "hlsfactory_cholesky"
    _write(tmp_path, f"{bench}_dataflow.cpp", "int df;")
    _write(tmp_path, f"{bench}_dataflow_report.json", {"latency_cycles": 300})
    _write(tmp_path, f"{bench}_dataflow_latency_opt.cpp", "int dflat;")
    _write(tmp_path, f"{bench}_dataflow_latency_opt_report.json", {"latency_cycles": 150})
    _write(
        tmp_path,
        f"{bench}_dataflow_latency_opt_result.json",
        {"success": True, "latency_cycles": 150},
    )
    ranked = rank_and_promote(tmp_path, bench, side="dataflow")
    assert ranked[0]["id"] == "dataflow:lat_opt"
    dest = tmp_path / f"{bench}_dataflow_selected.cpp"
    assert dest.is_file()
    assert "dflat" in dest.read_text(encoding="utf-8")
    ranking = json.loads((tmp_path / f"{bench}_dataflow_candidate_ranking.json").read_text())
    assert ranking["rank1_id"] == "dataflow:lat_opt"


def test_flash_seed_not_stolen_when_selected_is_lat_opt(tmp_path: Path) -> None:
    """If lat_opt overwrote selected, seed must come from final (or flash_seed)."""
    bench = "hlsfactory_symm"
    _write(tmp_path, f"{bench}_final.cpp", "int true_seed;")
    _write(tmp_path, f"{bench}_flash_opt_report.json", {"latency_cycles": 800})
    _write(tmp_path, f"{bench}_selected.cpp", "int broken_lat_opt;")
    _write(tmp_path, f"{bench}_selected_report.json", {"latency_cycles": 600})
    _write(tmp_path, f"{bench}_latency_opt.cpp", "int broken_lat_opt;")
    _write(tmp_path, f"{bench}_latency_opt_report.json", {"latency_cycles": 600})
    _write(
        tmp_path,
        f"{bench}_latency_opt_result.json",
        {"success": True, "latency_cycles": 600, "seed_latency_cycles": 800},
    )
    cands = collect_flash_side_candidates(tmp_path, bench)
    ranked = rank_candidates(cands)
    assert any(c["id"] == "flash:seed" for c in ranked)
    seed = next(c for c in ranked if c["id"] == "flash:seed")
    assert Path(seed["code_path"]).name == f"{bench}_final.cpp"
    assert "true_seed" in Path(seed["code_path"]).read_text(encoding="utf-8")
    assert seed["latency_cycles"] == 800.0
    lat = next(c for c in ranked if c["id"] == "flash:lat_opt")
    assert "broken_lat_opt" in Path(lat["code_path"]).read_text(encoding="utf-8")
    # Distinct bodies → both in pool; lat_opt wins on latency.
    assert ranked[0]["id"] == "flash:lat_opt"


def test_flash_seed_preserved_file_preferred(tmp_path: Path) -> None:
    bench = "hlsfactory_symm"
    _write(tmp_path, f"{bench}_flash_seed.cpp", "int preserved;")
    _write(tmp_path, f"{bench}_flash_seed_report.json", {"latency_cycles": 111})
    _write(tmp_path, f"{bench}_selected.cpp", "int lat;")
    _write(tmp_path, f"{bench}_latency_opt.cpp", "int lat;")
    _write(tmp_path, f"{bench}_latency_opt_report.json", {"latency_cycles": 50})
    _write(tmp_path, f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 50})
    ranked = rank_candidates(collect_flash_side_candidates(tmp_path, bench))
    seed = next(c for c in ranked if c["id"] == "flash:seed")
    assert Path(seed["code_path"]).name == f"{bench}_flash_seed.cpp"
    assert seed["latency_cycles"] == 111.0


def test_flash_seed_included_when_latency_unknown(tmp_path: Path) -> None:
    """Code-only seed must still enter the pool so cosim is not empty."""
    bench = "hlsfactory_lu"
    _write(tmp_path, f"{bench}_final.cpp", "int seed_only;")
    _write(tmp_path, f"{bench}_selected.cpp", "int seed_only;")
    # No reports, no successful lat_opt.
    ranked = rank_candidates(collect_flash_side_candidates(tmp_path, bench))
    assert len(ranked) == 1
    assert ranked[0]["id"] == "flash:seed"
    assert ranked[0]["latency_cycles"] == float("inf")


if __name__ == "__main__":
    import tempfile

    test_lat_opt_improved_strict()
    with tempfile.TemporaryDirectory() as td:
        test_flash_seed_wins_when_lat_opt_worse(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_flash_lat_opt_wins_when_better(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_dataflow_pool_and_promote(Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_promote_flash_rank1_copies_code(Path(td))
    print("ok")
