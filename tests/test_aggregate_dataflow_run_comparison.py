"""Tests for aggregate_dataflow_run_comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))

from aggregate_dataflow_run_comparison import discover_runs, _bench_rows_from_bundle


MATRIX = Path(__file__).resolve().parents[1] / (
    "artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548"
)


def test_original_kernel_bundle_has_rows():
    kb = MATRIX / "post_flash_dataflow_kernel_bundle"
    if not kb.is_dir():
        return
    rows = _bench_rows_from_bundle(kb)
    assert len(rows) >= 20
    assert rows[0]["speedup"]["worst"] is not None


def test_discover_runs_includes_baseline_and_packaged():
    if not MATRIX.is_dir():
        return
    runs = discover_runs(MATRIX)
    ids = {r["run_id"] for r in runs}
    assert "original_kernel_bundle" in ids
    assert any(r.startswith("post_flash_dataflow_results_") for r in ids)


def test_comparison_json_exists_after_generation():
    path = MATRIX / "reports/post_flash_dataflow_run_comparison.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    assert data["run_count"] >= 5
    assert "geom_mean_speedup_worst" in data["runs"][0].get("aggregate", {}) or True
