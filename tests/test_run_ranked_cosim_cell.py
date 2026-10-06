"""Unit tests for ranked cosim walker (dry-run / ranking load)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from flash_df_candidate_rank import rank_and_promote  # noqa: E402
from run_ranked_cosim_cell import run_ranked_cosim  # noqa: E402


def _write(path: Path, text: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, dict):
        path.write_text(json.dumps(text), encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")


def test_ranked_cosim_dry_run_uses_rank1(tmp_path: Path, monkeypatch):
    bench = "hlsfactory_gemm"
    cell = tmp_path / "cell"
    cell.mkdir()
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write(cell / f"{bench}_selected.cpp", "int seed;")
    _write(cell / f"{bench}_selected_report.json", {"latency_cycles": 100})
    _write(cell / f"{bench}_latency_opt.cpp", "int lat;")
    _write(cell / f"{bench}_latency_opt_report.json", {"latency_cycles": 50})
    _write(cell / f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 50})
    rank_and_promote(cell, bench, side="flash")

    monkeypatch.setattr("run_ranked_cosim_cell._supports_cosim", lambda _b: True)

    def _fake_cosim(cell_obj, run_root, *, force=False, dry_run=False):
        return {
            "status": "dry_run" if dry_run else "pass",
            "passed": not dry_run,
            "final_cpp": cell_obj.final_cpp,
        }

    monkeypatch.setattr("run_ranked_cosim_cell.run_cell_cosim", _fake_cosim)

    result = run_ranked_cosim(
        cell_dir=cell,
        bench=bench,
        side="flash",
        campaign_root=campaign,
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["winner_id"] == "flash:lat_opt"
    assert (cell / f"{bench}_flash_cosim_opt_result.json").is_file()


def test_ranked_cosim_walks_to_second_on_fail(tmp_path: Path, monkeypatch):
    bench = "hlsfactory_gemm"
    cell = tmp_path / "cell"
    cell.mkdir()
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write(cell / f"{bench}_selected.cpp", "int seed;")
    _write(cell / f"{bench}_selected_report.json", {"latency_cycles": 100})
    _write(cell / f"{bench}_latency_opt.cpp", "int lat;")
    _write(cell / f"{bench}_latency_opt_report.json", {"latency_cycles": 50})
    _write(cell / f"{bench}_latency_opt_result.json", {"success": True, "latency_cycles": 50})
    rank_and_promote(cell, bench, side="flash")

    monkeypatch.setattr("run_ranked_cosim_cell._supports_cosim", lambda _b: True)

    calls = {"n": 0}

    def _fake_cosim(cell_obj, run_root, *, force=False, dry_run=False):
        calls["n"] += 1
        # First (lat_opt) fails; second (seed) passes.
        if "latency_opt" in Path(cell_obj.final_cpp).name or calls["n"] == 1:
            return {"status": "fail", "passed": False, "error": "boom"}
        return {"status": "pass", "passed": True}

    monkeypatch.setattr("run_ranked_cosim_cell.run_cell_cosim", _fake_cosim)
    result = run_ranked_cosim(
        cell_dir=cell,
        bench=bench,
        side="flash",
        campaign_root=campaign,
        force=True,
    )
    assert result["status"] == "pass"
    assert result["winner_id"] == "flash:seed"
    assert calls["n"] == 2
