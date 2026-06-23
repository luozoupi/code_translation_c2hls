"""Standalone Vitis cosim for existing flash matrix *_final.cpp cells (no LLM / no c2hls flow)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from c2hls_temp import join_temp_tag, temp_tag_scope

REPO = Path(__file__).resolve().parents[2]
PC2_ARTIFACTS = REPO / "artifacts" / "pc2"
DEFAULT_COSIM_ROOT = PC2_ARTIFACTS / "flash_cosim"
DEFAULT_PART = "xcu280-fsvh2892-2L-e"
DEFAULT_CLOCK_NS = 3.33


@dataclass(frozen=True)
class CosimCell:
    index: int
    cell_id: str
    artifact_dir: str
    artifact_basename: str
    artifact_stamp: str
    matrix_family: str
    bench: str
    setup_tag: str
    variant: str
    mode: str
    model: str
    curation_focus: str
    skills_json: str
    cell_dir: str
    final_cpp: str
    source_matrix_status: str
    supports_cosim: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _artifact_stamp_from_name(name: str) -> str:
    match = re.search(r"(\d{8}_\d{6})$", name)
    return match.group(1) if match else ""


def make_cell_id(artifact_basename: str, bench: str, setup_tag: str) -> str:
    return join_temp_tag(artifact_basename, bench, setup_tag)


def apply_cosim_size_overrides(header_code: str, overrides: dict[str, Any]) -> str:
    if not overrides:
        return header_code
    block_lines: list[str] = []
    for key, value in overrides.items():
        block_lines.append(f"#undef {key}")
        block_lines.append(f"#define {key} {value}")
    block = "\n".join(block_lines)
    return f"// cosim size overrides\n{block}\n{header_code}"


def load_cosim_inputs(bench_dir: Path) -> dict[str, Any]:
    """Load header, cosim testbench, gold kernel, and metadata for cosim."""
    from c2hls import _load_benchmark_inputs

    base = _load_benchmark_inputs(str(bench_dir))
    meta = base["meta"]
    if not meta.get("supports_cosim"):
        raise ValueError(f"{bench_dir.name}: supports_cosim is false")

    tb_name = meta.get("cosim_testbench_file") or meta.get("testbench_file") or ""
    tb_path = bench_dir / tb_name if tb_name else None
    if not tb_path or not tb_path.exists():
        raise FileNotFoundError(f"{bench_dir.name}: cosim testbench missing ({tb_name})")

    testbench_code = tb_path.read_text()
    header_code = apply_cosim_size_overrides(
        base.get("header_code", ""),
        meta.get("cosim_size_overrides") or {},
    )

    extra_files: list[dict[str, str]] = []
    seen: set[str] = set()
    for rel_path in meta.get("cosim_support_files") or []:
        file_path = bench_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"{bench_dir.name}: cosim support file missing ({rel_path})")
        extra_files.append({"path": rel_path, "content": file_path.read_text(), "tb": True})
        seen.add(rel_path)

    gold_src = meta.get("gold_hls_source_file") or "gold_hls_source.cpp"
    gold_src_path = bench_dir / gold_src
    if gold_src_path.exists() and gold_src not in seen:
        extra_files.append({"path": gold_src, "content": gold_src_path.read_text(), "tb": False})
        seen.add(gold_src)

    for item in base.get("extra_files") or []:
        rel_path = item.get("path", "")
        if not rel_path or rel_path in seen:
            continue
        if rel_path.endswith(".cpp"):
            extra_files.append(item)
            seen.add(rel_path)

    top_function = meta.get("translated_hls_top") or meta.get("hls_top") or "workload"
    return {
        "meta": meta,
        "bench_name": base["bench_name"],
        "header_name": base.get("header_name") or "kernel.h",
        "header_code": header_code,
        "testbench_code": testbench_code,
        "extra_files": extra_files,
        "top_function": top_function,
        "part": meta.get("part", DEFAULT_PART),
        "clock_ns": float(meta.get("clock_ns", DEFAULT_CLOCK_NS)),
        "cosim_depths": meta.get("cosim_depths") or {},
    }


def discover_cells(
    *,
    artifacts_root: Path = PC2_ARTIFACTS,
    artifact_glob: str = "flash_*",
    bench_filter: Optional[set[str]] = None,
    artifact_filter: Optional[set[str]] = None,
) -> list[CosimCell]:
    cells: list[CosimCell] = []
    index = 0
    for matrix_path in sorted(artifacts_root.glob(f"{artifact_glob}/matrix.json")):
        artifact_dir = matrix_path.parent
        artifact_basename = artifact_dir.name
        if artifact_filter and artifact_basename not in artifact_filter:
            continue
        rows = json.loads(matrix_path.read_text())
        if not isinstance(rows, list):
            continue
        for row in rows:
            bench = row.get("bench", "")
            if bench_filter and bench not in bench_filter:
                continue
            cell_dir = Path(row.get("cell_dir", ""))
            if not cell_dir.is_dir():
                continue
            finals = sorted(cell_dir.glob(f"{bench}_final.cpp"))
            if not finals:
                continue
            setup_tag = cell_dir.name
            cell_id = make_cell_id(artifact_basename, bench, setup_tag)
            bench_dir = REPO / "benchmarks" / bench
            supports_cosim = False
            if (bench_dir / "metadata.json").exists():
                try:
                    meta = json.loads((bench_dir / "metadata.json").read_text())
                    supports_cosim = bool(meta.get("supports_cosim"))
                except (OSError, json.JSONDecodeError):
                    supports_cosim = False
            cells.append(
                CosimCell(
                    index=index,
                    cell_id=cell_id,
                    artifact_dir=str(artifact_dir),
                    artifact_basename=artifact_basename,
                    artifact_stamp=_artifact_stamp_from_name(artifact_basename),
                    matrix_family=row.get("matrix_family", ""),
                    bench=bench,
                    setup_tag=setup_tag,
                    variant=row.get("variant", ""),
                    mode=row.get("mode", ""),
                    model=row.get("model", ""),
                    curation_focus=row.get("curation_focus", ""),
                    skills_json=row.get("skills_json", ""),
                    cell_dir=str(cell_dir),
                    final_cpp=str(finals[0]),
                    source_matrix_status=row.get("status", ""),
                    supports_cosim=supports_cosim,
                )
            )
            index += 1
    return cells


def cosim_run_root(stamp: Optional[str] = None) -> Path:
    run_stamp = stamp or os.getenv("C2HLS_FLASH_COSIM_STAMP", "").strip() or _utc_stamp()
    root = Path(os.getenv("C2HLS_FLASH_COSIM_ROOT", str(DEFAULT_COSIM_ROOT))) / run_stamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def manifest_path(run_root: Path) -> Path:
    return run_root / "manifest.json"


def cell_result_dir(run_root: Path, cell_id: str) -> Path:
    return run_root / "cells" / cell_id


def cell_result_path(run_root: Path, cell_id: str) -> Path:
    return cell_result_dir(run_root, cell_id) / "cosim_result.json"


def write_manifest(run_root: Path, cells: list[CosimCell], *, extra: Optional[dict] = None) -> Path:
    payload = {
        "schema": "flash_cosim_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "repo": str(REPO),
        "cell_count": len(cells),
        "cells": [cell.to_dict() for cell in cells],
    }
    if extra:
        payload.update(extra)
    path = manifest_path(run_root)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(run_root: Path) -> dict[str, Any]:
    return json.loads(manifest_path(run_root).read_text())


def find_cell(manifest: dict[str, Any], cell_id: str) -> CosimCell:
    for raw in manifest.get("cells", []):
        if raw.get("cell_id") == cell_id:
            return CosimCell(**raw)
    raise KeyError(f"cell_id not in manifest: {cell_id}")


def find_cell_by_index(manifest: dict[str, Any], index: int) -> CosimCell:
    for raw in manifest.get("cells", []):
        if int(raw.get("index", -1)) == index:
            return CosimCell(**raw)
    raise KeyError(f"manifest index not found: {index}")


def run_cell_cosim(
    cell: CosimCell,
    run_root: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    out_dir = cell_result_dir(run_root, cell.cell_id)
    out_path = cell_result_path(run_root, cell.cell_id)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = cell.to_dict()
    provenance["cosim_run_root"] = str(run_root)

    if not cell.supports_cosim:
        result = {
            "status": "skipped",
            "reason": "benchmark does not support cosim",
            "provenance": provenance,
        }
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    hls_code = Path(cell.final_cpp).read_text()
    bench_dir = REPO / "benchmarks" / cell.bench
    inputs = load_cosim_inputs(bench_dir)

    temp_tag = join_temp_tag(cell.bench, cell.setup_tag, "cosim")
    if dry_run:
        result = {
            "status": "dry_run",
            "provenance": provenance,
            "temp_tag": temp_tag,
            "top_function": inputs["top_function"],
            "part": inputs["part"],
            "clock_ns": inputs["clock_ns"],
            "final_cpp": cell.final_cpp,
            "cosim_testbench": inputs["meta"].get("cosim_testbench_file"),
            "cosim_support_files": inputs["meta"].get("cosim_support_files"),
        }
        return result

    t0 = time.time()
    import hls_eval

    with temp_tag_scope(cell.bench, cell.setup_tag, "cosim"):
        cosim = hls_eval.run_cosim(
            hls_code,
            inputs["testbench_code"],
            inputs["header_code"],
            header_name=inputs["header_name"],
            top_function=inputs["top_function"],
            part=inputs["part"],
            clock_ns=inputs["clock_ns"],
            extra_files=inputs["extra_files"],
            interface_depths=inputs["cosim_depths"],
        )
    elapsed = round(time.time() - t0, 3)

    status = "ok" if cosim.get("success") else "fail"
    result = {
        "status": status,
        "passed": bool(cosim.get("passed")),
        "error": cosim.get("error") or "",
        "runtime_seconds": elapsed,
        "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
        "kernel_runtime_us": cosim.get("kernel_runtime_us"),
        "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
        "work_dir": cosim.get("work_dir"),
        "provenance": provenance,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "cosim.log"
    log_path.write_text(cosim.get("log") or "", encoding="utf-8")
    return result
