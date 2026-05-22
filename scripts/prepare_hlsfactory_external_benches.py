#!/usr/bin/env python3
"""Materialize selected HLSFactory kernels as c2hls benchmark directories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dataset_pipeline.external_adapter import adapt_external_kernel  # noqa: E402


HLSFACTORY = REPO / "external_datasets" / "HLSFactory"
DEFAULT_SOURCE_ROOT = (
    HLSFACTORY
    / "hlsfactory"
    / "hls_dataset_sources"
    / "polybench__reproducible"
    / "polybench__float__small"
)
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks_external" / "HLSFactory" / "polybench_float_small"


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _find_one(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _plain_has_strip_leak(info: dict) -> bool:
    strip = info.get("strip_report") or {}
    return bool(
        strip.get("plain_contains_hls_pragmas")
        or strip.get("plain_contains_accel_pragmas")
        or strip.get("plain_contains_ap_uint")
    )


def _materialize_case(case_dir: Path, output_root: Path, prefix: str) -> dict:
    case = case_dir.name
    if not case_dir.is_dir():
        return {"case": case, "status": "skip", "reason": "case directory not found"}
    kernel = _find_one(case_dir, ("src/*.cpp", "src/*.c"))
    header = _find_one(case_dir, ("src/*.h", "src/*.hpp"))
    testbench = _find_one(case_dir, ("tb/*_tb.cpp", "tb/testbench.cpp", "tb/*.cpp", "tb/*.c"))
    if kernel is None:
        return {"case": case, "status": "skip", "reason": "no src/*.cpp or src/*.c"}

    bench_name = f"{prefix}_{case.replace('-', '_')}"
    out_dir = output_root / bench_name
    info = adapt_external_kernel(
        kernel_path=kernel,
        header_path=header,
        testbench_path=testbench,
        bench_name=bench_name,
        output_dir=out_dir,
        source_repo="HLSFactory",
        top_function=None,
    )
    if _plain_has_strip_leak(info):
        return {
            "case": case,
            "status": "skip",
            "reason": "plain.cpp still contains stripped-HLS leak tokens",
            "bench_name": bench_name,
            "output_dir": str(out_dir),
            **info,
        }
    return {
        "case": case,
        "status": "ok",
        "bench_name": bench_name,
        "output_dir": str(out_dir),
        "kernel": str(kernel),
        "header": str(header) if header else "",
        "testbench": str(testbench) if testbench else "",
        **info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benches", default="gemm,bicg,atax")
    parser.add_argument("--all", action="store_true",
                        help="Materialize every case directory under --source-root.")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated case names to skip after selection.")
    parser.add_argument("--prefix", default="hlsfactory")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    requested = [] if args.all else _split_csv(args.benches)
    if requested:
        case_dirs = [args.source_root / item for item in requested]
    else:
        case_dirs = sorted(path for path in args.source_root.iterdir() if path.is_dir())
    excluded = set(_split_csv(args.exclude))
    if excluded:
        case_dirs = [path for path in case_dirs if path.name not in excluded]
    if args.limit > 0:
        case_dirs = case_dirs[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [_materialize_case(case_dir, args.output_root, args.prefix) for case_dir in case_dirs]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = REPO / "artifacts" / f"hlsfactory_materialized_{stamp}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "rows": rows,
    }, indent=2) + "\n")

    print(json.dumps({
        "manifest": str(manifest),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "skip": sum(1 for row in rows if row.get("status") != "ok"),
        "output_root": str(args.output_root),
    }, indent=2))
    return 0 if any(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
