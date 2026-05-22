#!/usr/bin/env python3
"""Materialize hls-eval kernels as c2hls benchmark directories.

The hls-eval tree is mostly organized as:

    hls_eval_data/<suite>/<case>/<case>.cpp
    hls_eval_data/<suite>/<case>/<case>.h
    hls_eval_data/<suite>/<case>/<case>_tb.cpp
    hls_eval_data/<suite>/<case>/top.txt

This script keeps only cases that can be represented by the current single-top
c2hls benchmark shape. It excludes WIP cases and ap_int/MARS kernels by
default because those are not clean C inputs after stripping.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dataset_pipeline.external_adapter import (  # noqa: E402
    adapt_external_kernel,
    classify_source_file,
    infer_top_function,
)


HLS_EVAL = REPO / "external_datasets" / "hls-eval"
DEFAULT_SOURCE_ROOT = HLS_EVAL / "hls_eval_data"
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks_external" / "hls_eval"
TEXT_AUX_SUFFIXES = {
    ".cfg",
    ".csv",
    ".data",
    ".dat",
    ".in",
    ".json",
    ".md",
    ".old",
    ".out",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_").lower()


def _read_top(case_dir: Path, source_text: str) -> str:
    top_txt = case_dir / "top.txt"
    if top_txt.is_file():
        for line in top_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return infer_top_function(source_text)


def _is_testbench(path: Path) -> bool:
    name = path.name.lower()
    return (
        "_tb." in name
        or "testbench" in name
        or name.startswith("tb_")
        or name.startswith("test_")
        or name == "gen_tb.cpp"
    )


def _source_files(case_dir: Path) -> list[Path]:
    return sorted(
        path for path in case_dir.iterdir()
        if path.is_file()
        and path.suffix in {".c", ".cc", ".cpp", ".cxx"}
        and not _is_testbench(path)
    )


def _header_files(case_dir: Path) -> list[Path]:
    return sorted(
        path for path in case_dir.iterdir()
        if path.is_file() and path.suffix in {".h", ".hh", ".hpp"}
    )


def _testbench_files(case_dir: Path) -> list[Path]:
    return sorted(
        path for path in case_dir.iterdir()
        if path.is_file()
        and path.suffix in {".c", ".cc", ".cpp", ".cxx"}
        and _is_testbench(path)
    )


def _contains_function_definition(source_text: str, name: str) -> bool:
    pattern = re.compile(
        rf"^\s*(?:extern\s+\"C\"\s+)?[\w:<>,~*&\s]+\b{re.escape(name)}\s*\([^;]*\)\s*\{{",
        re.MULTILINE,
    )
    return bool(pattern.search(source_text))


def _pick_kernel(case_dir: Path) -> tuple[Path | None, str]:
    candidates = _source_files(case_dir)
    if not candidates:
        return None, "no non-testbench C/C++ source file"
    if len(candidates) == 1:
        return candidates[0], ""

    top = ""
    top_txt = case_dir / "top.txt"
    if top_txt.is_file():
        top = next(
            (line.strip() for line in top_txt.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()),
            "",
        )
    if top:
        matches = [
            path for path in candidates
            if _contains_function_definition(path.read_text(encoding="utf-8", errors="ignore"), top)
        ]
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return None, f"multiple source files define top {top!r}"

    case_stem = case_dir.name.split("__")[-1].lower()
    name_matches = [path for path in candidates if path.stem.lower() in {case_dir.name.lower(), case_stem}]
    if len(name_matches) == 1:
        return name_matches[0], ""
    return None, f"ambiguous source files: {[path.name for path in candidates]}"


def _pick_header(case_dir: Path, kernel: Path, top: str) -> Path | None:
    headers = _header_files(case_dir)
    if not headers:
        return None
    for wanted in (kernel.stem, top, case_dir.name, case_dir.name.split("__")[-1]):
        for header in headers:
            if header.stem == wanted:
                return header
    return headers[0]


def _pick_testbench(case_dir: Path, kernel: Path) -> Path | None:
    tbs = _testbench_files(case_dir)
    if not tbs:
        return None
    preferred = [
        f"{kernel.stem}_tb",
        f"{case_dir.name}_tb",
        f"{case_dir.name.split('__')[-1]}_tb",
        "testbench",
        "gen_tb",
    ]
    for wanted in preferred:
        for tb in tbs:
            if tb.stem == wanted:
                return tb
    return tbs[0]


def _auxiliary_files(case_dir: Path, selected: set[Path]) -> list[Path]:
    aux: list[Path] = []
    for path in sorted(case_dir.iterdir()):
        if not path.is_file() or path in selected:
            continue
        if path.suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
            continue
        if path.suffix.lower() in TEXT_AUX_SUFFIXES:
            aux.append(path)
    return aux


def _strip_report_has_leak(info: dict) -> bool:
    strip = info.get("strip_report") or {}
    return bool(
        strip.get("plain_contains_hls_pragmas")
        or strip.get("plain_contains_accel_pragmas")
        or strip.get("plain_contains_ap_uint")
    )


def _case_dirs(source_root: Path, suites: set[str]) -> list[Path]:
    dirs: list[Path] = []
    for suite_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if suites and suite_dir.name not in suites:
            continue
        dirs.extend(sorted(path for path in suite_dir.iterdir() if path.is_dir()))
    return dirs


def _materialize_case(case_dir: Path, output_root: Path, prefix: str, *, allow_ap_int: bool) -> dict:
    suite = case_dir.parent.name
    case = case_dir.name
    if "__WIP" in case:
        return {"suite": suite, "case": case, "status": "skip", "reason": "WIP case"}

    kernel, reason = _pick_kernel(case_dir)
    if kernel is None:
        return {"suite": suite, "case": case, "status": "skip", "reason": reason}

    classification = classify_source_file(kernel)
    if classification.classification in {"hls_apuint", "hls_mars"} and not allow_ap_int:
        return {
            "suite": suite,
            "case": case,
            "status": "skip",
            "reason": f"unsupported source class {classification.classification}",
            "kernel": str(kernel),
        }

    raw = kernel.read_text(encoding="utf-8", errors="ignore")
    top = _read_top(case_dir, raw)
    header = _pick_header(case_dir, kernel, top)
    testbench = _pick_testbench(case_dir, kernel)
    selected = {kernel}
    if header:
        selected.add(header)
    if testbench:
        selected.add(testbench)
    aux = _auxiliary_files(case_dir, selected)

    bench_name = f"{prefix}_{_safe_name(suite)}_{_safe_name(case)}"
    out_dir = output_root / bench_name
    info = adapt_external_kernel(
        kernel_path=kernel,
        header_path=header,
        testbench_path=testbench,
        root_support_paths=aux,
        bench_name=bench_name,
        output_dir=out_dir,
        source_repo="hls-eval",
        top_function=top,
    )
    if _strip_report_has_leak(info):
        return {
            "suite": suite,
            "case": case,
            "status": "skip",
            "reason": "plain.cpp still contains stripped-HLS leak tokens",
            "bench_name": bench_name,
            "output_dir": str(out_dir),
            **info,
        }

    return {
        "suite": suite,
        "case": case,
        "status": "ok",
        "bench_name": bench_name,
        "output_dir": str(out_dir),
        "kernel": str(kernel),
        "header": str(header) if header else "",
        "testbench": str(testbench) if testbench else "",
        "auxiliary_files": [str(path) for path in aux],
        **info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suites", default="machsuite,polybench",
                        help="Comma-separated hls-eval suites to materialize; empty means all suites.")
    parser.add_argument("--benches", default="",
                        help="Comma-separated case names or suite/case selectors.")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated case names or suite/case selectors to skip.")
    parser.add_argument("--prefix", default="hlseval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-ap-int", action="store_true",
                        help="Materialize ap_int/ap_uint cases instead of skipping them.")
    args = parser.parse_args()

    suites = set(_split_csv(args.suites))
    requested = set(_split_csv(args.benches))
    excluded = set(_split_csv(args.exclude))

    if not args.source_root.is_dir():
        print(f"error: source root not found: {args.source_root}", file=sys.stderr)
        return 2

    case_dirs = _case_dirs(args.source_root, suites)
    if requested:
        case_dirs = [
            path for path in case_dirs
            if path.name in requested or f"{path.parent.name}/{path.name}" in requested
        ]
    if excluded:
        case_dirs = [
            path for path in case_dirs
            if path.name not in excluded and f"{path.parent.name}/{path.name}" not in excluded
        ]
    if args.limit > 0:
        case_dirs = case_dirs[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        _materialize_case(case_dir, args.output_root, args.prefix, allow_ap_int=args.allow_ap_int)
        for case_dir in case_dirs
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = REPO / "artifacts" / f"hls_eval_materialized_{stamp}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "suites": sorted(suites),
            "rows": rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "manifest": str(manifest),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "skip": sum(1 for row in rows if row.get("status") != "ok"),
        "output_root": str(args.output_root),
    }, indent=2))
    return 0 if any(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
