#!/usr/bin/env python3
"""Export c2hls prefixed benchmarks to ChatHLS benchmark_optimization layout."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from c2hls_port_loop_labels import build_kernel_info, inject_loop_labels

DEFAULT_PART = "xczu7ev-ffvc1156-2-e"
DEFAULT_CLOCK_PERIOD = 10
BASELINE_CANDIDATES = ("gold_hls_baseline_file", "hls_baseline.cpp")
# Default globs for --all-prefixed (legacy dual-track + Tier-A suites).
PREFixed_GLOBS = (
    "hlsfactory_*",
    "machsuite_*",
    "forgebench_*",
    "hp_fft_*",
    "spector_hls_*",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_top(metadata: dict[str, Any]) -> str:
    for key in ("hls_top", "kernel_top", "translated_hls_top"):
        value = metadata.get(key)
        if value:
            return str(value)
    raise ValueError("metadata.json missing hls_top/kernel_top")


def _resolve_baseline_file(metadata: dict[str, Any], src: Path) -> Path:
    for key in BASELINE_CANDIDATES:
        if key in metadata:
            candidate = src / metadata[key]
            if candidate.is_file():
                return candidate
        elif key == "hls_baseline.cpp":
            candidate = src / key
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no baseline source in {src}")


def _strip_local_include(source: str, header_name: str) -> str:
    pattern = re.compile(rf'^\s*#\s*include\s+"{re.escape(header_name)}"\s*\n?', re.M)
    return pattern.sub("", source, count=1)


def _header_already_inlined(baseline: str, header_text: str) -> bool:
    normalized_header = header_text.strip()
    if not normalized_header:
        return True
    return normalized_header in baseline


def _combine_source(
    baseline: str,
    *,
    header_path: Path | None,
) -> tuple[str, str]:
    """Return (combined_source, header_strategy)."""
    if header_path is None or not header_path.is_file():
        return baseline, "none"

    header_text = header_path.read_text()
    header_name = header_path.name

    if _header_already_inlined(baseline, header_text):
        return baseline, "baseline_already_contains_header"

    combined = header_text.rstrip() + "\n\n" + _strip_local_include(baseline, header_name).lstrip()
    return combined, "prepended_header_stripped_include"


def _write_run_hls_tcl(dest: Path, *, top: str) -> None:
    tcl = f"""open_project -reset test_proj
add_files {top}.cpp
set_top {top}
open_solution -reset solution
set_part {{{DEFAULT_PART}}}
create_clock -period {DEFAULT_CLOCK_PERIOD} -name default
csynth_design
exit
"""
    (dest / "run_hls.tcl").write_text(tcl)


def export_bench(src: Path, out_root: Path) -> dict[str, Any]:
    """Export one c2hls benchmark directory to ChatHLS layout under out_root."""
    src = src.resolve()
    out_root = out_root.resolve()
    warnings: list[str] = []

    metadata_path = src / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata.json in {src}")

    metadata = _read_json(metadata_path)
    top = _resolve_top(metadata)
    baseline_path = _resolve_baseline_file(metadata, src)
    baseline = baseline_path.read_text()

    header_path: Path | None = None
    header_file = metadata.get("header_file")
    if header_file:
        candidate = src / header_file
        if candidate.is_file():
            header_path = candidate
        else:
            warnings.append(f"header_file {header_file!r} not found; exporting baseline only")

    combined, header_strategy = _combine_source(
        baseline,
        header_path=header_path,
    )

    labeled, label_count = inject_loop_labels(combined, top=top)
    if label_count == 0:
        warnings.append(f"inject_loop_labels inserted 0 labels for top={top!r}")

    dest_name = src.name
    dest = out_root / dest_name
    dest.mkdir(parents=True, exist_ok=True)

    cpp_path = dest / f"{top}.cpp"
    cpp_path.write_text(labeled)

    if header_path is not None and header_strategy != "baseline_already_contains_header":
        (dest / header_path.name).write_text(header_path.read_text())

    kernel_info = build_kernel_info(labeled, top=top)
    (dest / "kernel_info.txt").write_text(kernel_info)

    _write_run_hls_tcl(dest, top=top)

    manifest = {
        "source": str(src),
        "dest": str(dest),
        "top": top,
        "baseline_file": baseline_path.name,
        "header_file": header_path.name if header_path else None,
        "header_strategy": header_strategy,
        "label_count": label_count,
        "warnings": warnings,
    }
    (dest / "port_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _iter_prefixed_benches(
    benchmarks_roots: list[Path],
    *,
    globs: tuple[str, ...] = PREFixed_GLOBS,
) -> list[Path]:
    """Collect unique bench dirs matching globs under one or more roots."""
    seen: set[str] = set()
    benches: list[Path] = []
    for root in benchmarks_roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for pattern in globs:
            for path in sorted(root.glob(pattern)):
                if not path.is_dir() or path.name in seen:
                    continue
                seen.add(path.name)
                benches.append(path)
    return benches


def _resolve_bench_list(bench_list: Path, search_roots: list[Path]) -> list[Path]:
    """Map each name in a bench list file to a directory under search_roots."""
    names = [
        line.strip()
        for line in bench_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    resolved: list[Path] = []
    missing: list[str] = []
    for name in names:
        found: Path | None = None
        for root in search_roots:
            candidate = root / name
            if candidate.is_dir():
                found = candidate
                break
        if found is None:
            missing.append(name)
        else:
            resolved.append(found)
    if missing:
        raise FileNotFoundError(
            f"could not resolve {len(missing)} benches under {[str(r) for r in search_roots]}: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )
    return resolved


def _export_many(benches: list[Path], out_root: Path, *, strict: bool) -> int:
    failed = 0
    for bench in benches:
        try:
            manifest = export_bench(bench, out_root)
            warn_suffix = ""
            if manifest["warnings"]:
                warn_suffix = f" warnings={len(manifest['warnings'])}"
            print(f"exported {bench.name} -> {manifest['dest']}{warn_suffix}")
        except Exception as exc:
            failed += 1
            print(f"WARN skip {bench.name}: {exc}", file=sys.stderr)
            if strict:
                return 1
    return 1 if failed else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export c2hls prefixed benchmarks to ChatHLS benchmark_optimization layout.",
    )
    parser.add_argument("--bench-dir", type=Path, help="Single benchmark directory to export")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Destination benchmark_optimization root",
    )
    parser.add_argument(
        "--all-prefixed",
        action="store_true",
        help=(
            "Export all matching globs under --benchmarks-root "
            "(default globs: hlsfactory/machsuite/forgebench/hp_fft/spector_hls)"
        ),
    )
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        action="append",
        dest="benchmarks_roots",
        default=None,
        help=(
            "Root directory containing prefixed benchmarks (repeatable). "
            "Default: ./benchmarks"
        ),
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=None,
        help="Glob under each benchmarks-root (repeatable). Default: PREFixed_GLOBS",
    )
    parser.add_argument(
        "--bench-list",
        type=Path,
        help="Text file of bench directory names to export (resolved under --benchmarks-root)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on the first export failure (default: continue and report)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    roots = [p.resolve() for p in (args.benchmarks_roots or [Path("benchmarks")])]
    globs = tuple(args.globs) if args.globs else PREFixed_GLOBS

    if args.bench_list is not None:
        try:
            benches = _resolve_bench_list(args.bench_list.resolve(), roots)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not benches:
            print(f"empty bench list: {args.bench_list}", file=sys.stderr)
            return 1
        return _export_many(benches, args.out_root, strict=args.strict)

    if args.all_prefixed:
        benches = _iter_prefixed_benches(roots, globs=globs)
        if not benches:
            print(
                f"no prefixed benchmarks matching {globs} under {roots}",
                file=sys.stderr,
            )
            return 1
        return _export_many(benches, args.out_root, strict=args.strict)

    if args.bench_dir is None:
        parser.error("either --bench-dir, --all-prefixed, or --bench-list is required")

    try:
        manifest = export_bench(args.bench_dir, args.out_root)
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
