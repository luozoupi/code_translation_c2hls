"""Gold cosim RTL cycles for benchmarks/ (not benchmarks_cosim fixed corpus)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

DEFAULT_BENCHMARK_COSIM_BASELINE_JSONL = (
    REPO / "misc" / "hlsfactory_baseline_u280_20260616_benchmarks_naive_cosim.jsonl"
)

_BASELINE_CACHE: dict[str, Any] = {"path": None, "map": None}


def bench_short_from_group_path(group_path: list[str] | tuple[str, ...] | None) -> str:
    if not group_path:
        return ""
    return str(group_path[0]).replace("_", "-")


def is_benchmarks_dir_cosim_record(record: dict[str, Any]) -> bool:
    """True when rtl_sim used benchmarks/hlsfactory_* (naive), not benchmarks_cosim."""
    if record.get("report_type") != "rtl_sim":
        return False
    impl = record.get("implementation") or {}
    meta = impl.get("origin_meta") or {}
    if meta.get("cosim_export_suffix") == "fixed_cosim":
        return False
    if meta.get("corpus") == "benchmarks_cosim":
        return False
    bench_dir = str(meta.get("benchmark_dir") or "")
    if "benchmarks_cosim" in bench_dir:
        return False
    if bench_dir and "/benchmarks/hlsfactory_" not in bench_dir.replace("\\", "/"):
        # Allow relative paths like benchmarks/hlsfactory_2mm
        if not bench_dir.startswith("benchmarks/hlsfactory_"):
            return False
    return True


def load_benchmark_cosim_baseline(
    jsonl_path: Path | None = None,
    *,
    force_reload: bool = False,
) -> dict[str, int]:
    """Map bench short name (e.g. jacobi-1d) -> gold cosim kernel_runtime_cycles."""
    path = Path(
        jsonl_path
        or os.environ.get("C2HLS_BENCHMARK_COSIM_BASELINE_JSONL", "")
        or DEFAULT_BENCHMARK_COSIM_BASELINE_JSONL
    ).resolve()
    if (
        not force_reload
        and _BASELINE_CACHE["map"] is not None
        and _BASELINE_CACHE["path"] == str(path)
    ):
        return dict(_BASELINE_CACHE["map"])

    out: dict[str, int] = {}
    if not path.is_file():
        _BASELINE_CACHE["path"] = str(path)
        _BASELINE_CACHE["map"] = out
        return out

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not is_benchmarks_dir_cosim_record(record):
            continue
        rtl = record.get("rtl_sim") or {}
        if str(rtl.get("status") or "").lower() not in ("pass", "passed", "ok"):
            continue
        cycles = rtl.get("kernel_runtime_cycles")
        if cycles is None:
            continue
        try:
            value = int(cycles)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        short = bench_short_from_group_path(record.get("problem", {}).get("group_path"))
        if short:
            out[short] = value

    _BASELINE_CACHE["path"] = str(path)
    _BASELINE_CACHE["map"] = out
    return dict(out)
