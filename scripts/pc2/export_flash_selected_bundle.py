#!/usr/bin/env python3
"""Export flash-selected kernels into a self-contained hierarchical bundle.

Each benchmark gets its own directory with:
  - selected/     canonical kernel + synth report metadata
  - flash_cell/   copies of flash-run artifacts (cpp, json, steps/)
  - benchmark/    testbench, header, baseline, metadata, support files
  - tcl/          run_synth.tcl + run_csim.tcl (relative paths)

Example::

    python3 scripts/pc2/export_flash_selected_bundle.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from c2hls_paths import BENCHMARKS_DIR, configure_site
from flash_flow_artifacts import sha256_text
from post_flash_mem_parallel import discover_matrix_cells, resolve_selected_kernel

DEFAULT_PART = os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e")
DEFAULT_CLOCK_NS = float(os.getenv("C2HLS_CLOCK_NS", "3.33"))
DEFAULT_FLOW_TARGET = os.getenv("C2HLS_FLOW_TARGET", "vitis")

FLASH_CELL_GLOBS = (
    "{bench}_final.cpp",
    "{bench}_selected.cpp",
    "{bench}_selected_report.json",
    "{bench}_multistep_results.json",
    "{bench}_history.json",
    "{bench}_flow_manifest.json",
    "{bench}_flash_skills.json",
    "skills_source.json",
    "plain.cpp",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _resolve_bench_dir(bench: str) -> Path | None:
    candidates = [
        BENCHMARKS_DIR / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/chathls_ready" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready" / bench,
        REPO / "benchmarks_autosa_dse" / bench,
    ]
    for path in candidates:
        if path.is_dir() and (path / "metadata.json").is_file():
            return path
    return None


def _bench_metadata(bench: str) -> dict[str, Any]:
    bench_dir = _resolve_bench_dir(bench)
    if bench_dir is None:
        return {}
    return _load_json(bench_dir / "metadata.json")


def _benchmark_files(bench: str) -> list[Path]:
    bench_dir = _resolve_bench_dir(bench)
    if bench_dir is None:
        return []
    meta = _bench_metadata(bench)
    names: set[str] = set()
    for key in (
        "testbench_file",
        "cosim_testbench_file",
        "header_file",
        "plain_c_file",
        "gold_hls_source_file",
        "gold_hls_baseline_file",
        "kernel_file",
    ):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            names.add(val.strip())
    for item in meta.get("cosim_support_files") or []:
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
    for item in meta.get("support_files") or []:
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
    names.add("metadata.json")
    out: list[Path] = []
    for name in sorted(names):
        p = bench_dir / name
        if p.is_file():
            out.append(p)
    return out


def _selected_synth_report(cell_dir: Path, bench: str) -> dict[str, Any]:
    selected_report = cell_dir / f"{bench}_selected_report.json"
    if selected_report.is_file():
        data = _load_json(selected_report)
        return data if isinstance(data, dict) else {}
    multistep = cell_dir / f"{bench}_multistep_results.json"
    if multistep.is_file():
        data = _load_json(multistep)
        if isinstance(data, dict):
            final_report = data.get("final_report")
            if isinstance(final_report, dict):
                return final_report
    return {}


def _write_tcl(
    tcl_dir: Path,
    *,
    top_function: str,
    header_name: str,
    has_testbench: bool,
    extra_kernel_files: list[str],
    extra_tb_files: list[str],
) -> None:
    tcl_dir.mkdir(parents=True, exist_ok=True)
    kernel = "../selected/kernel.cpp"
    header = f"../benchmark/{header_name}" if header_name else ""
    testbench = "../benchmark/testbench.cpp"

    synth_lines = [
        "open_project hls_proj",
        f"set_top {top_function}",
        f"add_files {kernel}",
    ]
    if header:
        synth_lines.append(f"add_files {header}")
    for rel in extra_kernel_files:
        synth_lines.append(f"add_files ../benchmark/{rel}")
    synth_lines.extend([
        f'open_solution "sol1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {DEFAULT_CLOCK_NS} -name default",
        "csynth_design",
        "exit",
    ])
    (tcl_dir / "run_synth.tcl").write_text("\n".join(synth_lines) + "\n", encoding="utf-8")

    if not has_testbench:
        return

    csim_lines = [
        "open_project hls_proj",
        f"set_top {top_function}",
        f"add_files {kernel}",
    ]
    if header:
        csim_lines.append(f"add_files {header}")
    for rel in extra_kernel_files:
        csim_lines.append(f"add_files ../benchmark/{rel}")
    csim_lines.append(f"add_files -tb {testbench}")
    if header:
        csim_lines.append(f"add_files -tb {header}")
    for rel in extra_tb_files:
        csim_lines.append(f"add_files -tb ../benchmark/{rel}")
    csim_lines.extend([
        f'open_solution "sol1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {DEFAULT_CLOCK_NS} -name default",
        "csynth_design",
        "csim_design",
        "exit",
    ])
    (tcl_dir / "run_csim.tcl").write_text("\n".join(csim_lines) + "\n", encoding="utf-8")


def export_cell(
    cell: dict[str, Any],
    out_bench_dir: Path,
    *,
    include_post_flash: bool = False,
) -> dict[str, Any]:
    bench = cell["bench"]
    cell_dir = Path(cell["cell_dir"])
    kernel_path, kernel_role = resolve_selected_kernel(cell_dir, bench)
    bench_meta = _bench_metadata(bench)

    manifest: dict[str, Any] = {
        "benchmark": bench,
        "flash_status": cell.get("status"),
        "source_cell_dir": str(cell_dir),
        "kernel_role": kernel_role,
        "kernel_path": None,
        "skipped": False,
        "skip_reason": None,
        "files": {},
    }

    if kernel_path is None or not kernel_path.is_file():
        manifest["skipped"] = True
        manifest["skip_reason"] = "no selected/final kernel"
        _write_json(out_bench_dir / "bench_manifest.json", manifest)
        return manifest

    selected_dir = out_bench_dir / "selected"
    flash_dir = out_bench_dir / "flash_cell"
    benchmark_dir = out_bench_dir / "benchmark"
    tcl_dir = out_bench_dir / "tcl"

    kernel_code = kernel_path.read_text(encoding="utf-8")
    _copy_file(kernel_path, selected_dir / "kernel.cpp")
    synth_report = _selected_synth_report(cell_dir, bench)
    if synth_report:
        _write_json(selected_dir / "synth_report.json", synth_report)

    selected_meta = {
        "schema": "flash_selected_kernel_v1",
        "benchmark": bench,
        "source_file": kernel_path.name,
        "kernel_role": kernel_role,
        "sha256": sha256_text(kernel_code),
        "latency_cycles": synth_report.get("latency_cycles"),
    }
    _write_json(selected_dir / "meta.json", selected_meta)
    manifest["kernel_path"] = str(selected_dir / "kernel.cpp")

    for pattern in FLASH_CELL_GLOBS:
        rel = pattern.format(bench=bench)
        src = cell_dir / rel
        if src.is_file():
            _copy_file(src, flash_dir / rel)

    steps_src = cell_dir / "steps"
    if steps_src.is_dir():
        _copy_tree(steps_src, flash_dir / "steps")

    if include_post_flash:
        for path in sorted(cell_dir.glob(f"{bench}_mem_parallel_*")):
            if path.is_file():
                _copy_file(path, flash_dir / path.name)

    bench_files = _benchmark_files(bench)
    extra_kernel: list[str] = []
    extra_tb: list[str] = []
    header_name = str(bench_meta.get("header_file") or "")
    tb_name = str(bench_meta.get("testbench_file") or "testbench.cpp")
    for src in bench_files:
        rel = src.name
        _copy_file(src, benchmark_dir / rel)
        if rel in set(bench_meta.get("cosim_support_files") or []):
            extra_tb.append(rel)
        elif rel not in {header_name, tb_name, "metadata.json", "testbench.cpp"}:
            if rel.endswith((".cpp", ".c", ".h", ".hpp")):
                extra_kernel.append(rel)

    top = (
        bench_meta.get("hls_top")
        or bench_meta.get("kernel_top")
        or bench_meta.get("translated_hls_top")
        or "kernel"
    )
    _write_tcl(
        tcl_dir,
        top_function=str(top),
        header_name=header_name,
        has_testbench=bool(tb_name and (benchmark_dir / tb_name).is_file()),
        extra_kernel_files=extra_kernel,
        extra_tb_files=extra_tb,
    )

    manifest["files"] = {
        "selected_kernel": "selected/kernel.cpp",
        "selected_meta": "selected/meta.json",
        "selected_report": "selected/synth_report.json" if synth_report else None,
        "flash_cell": "flash_cell/",
        "benchmark": "benchmark/",
        "tcl": ["tcl/run_synth.tcl", "tcl/run_csim.tcl"],
    }
    manifest["top_function"] = top
    manifest["latency_cycles"] = synth_report.get("latency_cycles")
    _write_json(out_bench_dir / "bench_manifest.json", manifest)
    return manifest


def export_matrix(
    matrix_root: Path,
    out_root: Path,
    *,
    bench_filter: Optional[set[str]] = None,
    include_post_flash: bool = False,
) -> dict[str, Any]:
    cells = discover_matrix_cells(matrix_root)
    if bench_filter:
        cells = [c for c in cells if c["bench"] in bench_filter]

    matrix_out = out_root / matrix_root.name
    matrix_out.mkdir(parents=True, exist_ok=True)

    matrix_file = matrix_root / "matrix.json"
    if matrix_file.is_file():
        _copy_file(matrix_file, matrix_out / "matrix.json")

    bench_manifests: list[dict[str, Any]] = []
    for cell in cells:
        bench = cell["bench"]
        bench_out = matrix_out / bench
        bench_manifests.append(
            export_cell(
                cell,
                bench_out,
                include_post_flash=include_post_flash,
            )
        )

    bundle = {
        "schema": "flash_selected_bundle_v1",
        "matrix_root": str(matrix_root),
        "bundle_dir": str(matrix_out),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_count": len(cells),
        "exported_count": sum(1 for m in bench_manifests if not m.get("skipped")),
        "skipped": [m["benchmark"] for m in bench_manifests if m.get("skipped")],
        "include_post_flash": include_post_flash,
        "benchmarks": bench_manifests,
    }
    _write_json(matrix_out / "bundle_manifest.json", bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export flash-selected kernels into a hierarchical bundle")
    parser.add_argument("--pc2", action="store_true")
    parser.add_argument("--matrix-root", type=str, required=True)
    parser.add_argument(
        "--out-root",
        type=str,
        default="artifacts/pc2/flash_selected_bundle",
        help="Parent directory; each matrix exports to <out-root>/<matrix-name>/",
    )
    parser.add_argument("--benches", type=str, default="", help="Comma-separated bench filter")
    parser.add_argument(
        "--include-post-flash",
        action="store_true",
        help="Also copy mem_parallel_* artifacts from each flash cell",
    )
    args = parser.parse_args()

    if args.pc2:
        configure_site("pc2")

    matrix_root = Path(args.matrix_root).expanduser()
    if not matrix_root.is_absolute():
        matrix_root = REPO / matrix_root
    if not matrix_root.is_dir():
        print(f"matrix root missing: {matrix_root}", file=sys.stderr)
        return 1

    out_root = Path(args.out_root).expanduser()
    if not out_root.is_absolute():
        out_root = REPO / out_root

    bench_filter = None
    if args.benches.strip():
        bench_filter = {b.strip() for b in args.benches.split(",") if b.strip()}

    bundle = export_matrix(
        matrix_root,
        out_root,
        bench_filter=bench_filter,
        include_post_flash=args.include_post_flash,
    )
    print(
        f"exported {bundle['exported_count']}/{bundle['benchmark_count']} benches "
        f"-> {bundle['bundle_dir']}"
    )
    if bundle["skipped"]:
        print("skipped:", ", ".join(bundle["skipped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
