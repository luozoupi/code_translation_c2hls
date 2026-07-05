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
    test_discover_matrix_cells_from_json()
    print("test_post_flash_mem_parallel: ok")
