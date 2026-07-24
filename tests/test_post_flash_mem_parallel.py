"""Tests for post-flash memory parallelism helpers."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import post_flash_mem_parallel as pfm


def test_extract_labeled_cpp_blocks():
    text = """
Here is the code:

```kernel
void kernel_gemm() {}
```

```testbench
int main() { return 0; }
```
"""
    blocks = pfm.extract_labeled_cpp_blocks(text)
    assert "kernel" in blocks and "testbench" in blocks
    assert "kernel_gemm" in blocks["kernel"]
    assert "main" in blocks["testbench"]


def test_extract_labeled_kernel_only():
    text = '```kernel\n#include "gemm.h"\nvoid kernel_gemm() {}\n```'
    blocks = pfm.extract_labeled_cpp_blocks(text)
    assert blocks["kernel"].startswith('#include "gemm.h"')


def test_resolve_selected_kernel_prefers_selected():
    with tempfile.TemporaryDirectory() as tmp:
        cell = Path(tmp)
        bench = "hlsfactory_gemm"
        (cell / f"{bench}_final.cpp").write_text("// final\n")
        (cell / f"{bench}_selected.cpp").write_text("// selected\n")
        path, role = pfm.resolve_selected_kernel(cell, bench)
        assert path.name.endswith("_selected.cpp")
        assert role == "selected"


def test_resolve_prefers_latency_opt(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_latency_opt.cpp"
    assert role == "latency_opt"


def test_resolve_include_post_passes_false_returns_selected(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench, include_post_passes=False)
    assert path.name == f"{bench}_selected.cpp"
    assert role == "selected"


def test_resolve_prefers_pragma_opt_over_base(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_pragma_opt.cpp").write_text("pragma", encoding="utf-8")
    (tmp_path / f"{bench}_pragma_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_pragma_opt.cpp"
    assert role == "pragma_opt"


def test_resolve_ignores_failed_latency_opt_result(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": false}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_selected.cpp"
    assert role == "selected"


def test_resolve_prefers_dataflow_latency_opt_when_better(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("flash_opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true, "latency_cycles": 5000}', encoding="utf-8"
    )
    (tmp_path / f"{bench}_dataflow_latency_opt.cpp").write_text("df_opt", encoding="utf-8")
    (tmp_path / f"{bench}_dataflow_latency_opt_result.json").write_text(
        '{"success": true, "latency_cycles": 1828}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_dataflow_latency_opt.cpp"
    assert role == "dataflow_latency_opt"


def test_resolve_prefers_lower_flash_latency_opt(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("flash_opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true, "latency_cycles": 1000}', encoding="utf-8"
    )
    (tmp_path / f"{bench}_dataflow_latency_opt.cpp").write_text("df_opt", encoding="utf-8")
    (tmp_path / f"{bench}_dataflow_latency_opt_result.json").write_text(
        '{"success": true, "latency_cycles": 2000}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_latency_opt.cpp"
    assert role == "latency_opt"


def test_discover_matrix_cells_from_json():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cell = root / "hlsfactory_gemm" / "devstral2__flash__x"
        cell.mkdir(parents=True)
        (cell / "hlsfactory_gemm_final.cpp").write_text("// k\n")
        (root / "matrix.json").write_text(json.dumps([{
            "bench": "hlsfactory_gemm",
            "cell_dir": str(cell),
            "status": "ok",
        }]))
        cells = pfm.discover_matrix_cells(root)
        assert len(cells) == 1
        assert cells[0]["bench"] == "hlsfactory_gemm"


if __name__ == "__main__":
    test_extract_labeled_cpp_blocks()
    test_extract_labeled_kernel_only()
    test_resolve_selected_kernel_prefers_selected()
    test_resolve_prefers_latency_opt(Path(tempfile.mkdtemp()))
    test_resolve_include_post_passes_false_returns_selected(Path(tempfile.mkdtemp()))
    test_resolve_prefers_pragma_opt_over_base(Path(tempfile.mkdtemp()))
    test_resolve_ignores_failed_latency_opt_result(Path(tempfile.mkdtemp()))
    test_resolve_prefers_dataflow_latency_opt_when_better(Path(tempfile.mkdtemp()))
    test_resolve_prefers_lower_flash_latency_opt(Path(tempfile.mkdtemp()))
    test_discover_matrix_cells_from_json()
    print("test_post_flash_mem_parallel: ok")
