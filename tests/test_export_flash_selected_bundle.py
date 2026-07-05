"""Tests for flash selected bundle export."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.pc2.export_flash_selected_bundle import export_cell


def test_export_cell_gemm_structure(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    matrix_root = repo / "artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548"
    cell_dir = matrix_root / "hlsfactory_gemm/devstral2__flash__all_new_skills_avoids_global"
    if not cell_dir.is_dir():
        return

    out = tmp_path / "hlsfactory_gemm"
    manifest = export_cell(
        {
            "bench": "hlsfactory_gemm",
            "cell_dir": str(cell_dir),
            "status": "ok",
        },
        out,
    )
    assert manifest["skipped"] is False
    assert (out / "selected/kernel.cpp").is_file()
    assert (out / "benchmark/testbench.cpp").is_file()
    assert (out / "tcl/run_synth.tcl").is_file()
    assert (out / "flash_cell/hlsfactory_gemm_multistep_results.json").is_file()
    meta = json.loads((out / "selected/meta.json").read_text())
    assert meta["kernel_role"] in {"final", "selected"}
