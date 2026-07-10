"""
Ingest HLSFactory polybench_float_small benchmarks into c2hls's benchmarks/
directory so they can be driven by `python c2hls.py --bench hlsfactory_<name>`.

For each HLSFactory benchmark we materialize:
    benchmarks/hlsfactory_<name>/
        <name>.h              copy of HLSFactory src/<name>.h
        hls_baseline.cpp      copy of HLSFactory src/<name>.cpp (pragmas kept)
        gold_hls_source.cpp   identical to hls_baseline (HLSFactory's source IS the gold)
        plain.cpp             baseline with `#pragma HLS ...` lines stripped (LLM input)
        testbench.cpp         copy of HLSFactory tb/<name>_tb.cpp
        metadata.json         per c2hls schema

Also rewrites benchmarks/index.json to include the new entries (existing entries
are preserved unless --replace is used).

Usage:
    python3 ingest_hlsfactory.py [--root <polybench dir>] [--dry-run] [--only a,b,c]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

DEFAULT_ROOT = Path(
    "/mnt/e/courses/UMN/c2hls/HLSFactory/hlsfactory/hls_dataset_sources/"
    "polybench__reproducible/polybench__float__small"
)
BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"
INDEX_PATH = BENCHMARKS_DIR / "index.json"

HLS_PRAGMA_RE = re.compile(r"^[ \t]*#pragma\s+HLS\b.*$", re.MULTILINE)
ACCEL_PRAGMA_RE = re.compile(r"^[ \t]*#pragma\s+ACCEL\b.*$", re.MULTILINE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_pragmas(code: str) -> tuple[str, int, int]:
    hls_count = len(HLS_PRAGMA_RE.findall(code))
    accel_count = len(ACCEL_PRAGMA_RE.findall(code))
    stripped = HLS_PRAGMA_RE.sub("", code)
    stripped = ACCEL_PRAGMA_RE.sub("", stripped)
    return stripped, hls_count, accel_count


def _ingest_one(src_dir: Path, out_root: Path, dry_run: bool) -> dict | None:
    name = src_dir.name
    top_file = src_dir / "top.txt"
    src_subdir = src_dir / "src"
    tb_subdir = src_dir / "tb"
    if not top_file.is_file() or not src_subdir.is_dir():
        return None
    top = top_file.read_text().strip()
    cpp_files = sorted(src_subdir.glob("*.cpp"))
    h_files = sorted(src_subdir.glob("*.h"))
    if not cpp_files or not h_files:
        print(f"  SKIP {name}: missing .cpp or .h")
        return None
    src_cpp = cpp_files[0]
    src_h = h_files[0]
    tb_cpp = None
    if tb_subdir.is_dir():
        tbs = sorted(tb_subdir.glob("*_tb.cpp"))
        if tbs:
            tb_cpp = tbs[0]

    bench_name = f"hlsfactory_{name}"
    out_dir = out_root / bench_name
    header_basename = src_h.name  # keep original (e.g., "fdtd-2d.h")
    kernel_basename = src_cpp.name

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    gold_text = src_cpp.read_text()
    header_text = src_h.read_text()
    plain_text, stripped_hls, stripped_accel = _strip_pragmas(gold_text)
    tb_text = tb_cpp.read_text() if tb_cpp else ""

    if not dry_run:
        (out_dir / header_basename).write_text(header_text)
        (out_dir / "hls_baseline.cpp").write_text(gold_text)
        (out_dir / "gold_hls_source.cpp").write_text(gold_text)
        (out_dir / "plain.cpp").write_text(plain_text)
        if tb_text:
            (out_dir / "testbench.cpp").write_text(tb_text)

    meta = {
        "benchmark": bench_name,
        "source_repo": "HLSFactory",
        "algorithm_source_path": str(src_cpp),
        "gold_hls_source_path": str(src_cpp),
        "gold_hls_source_file": "gold_hls_source.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": kernel_basename,
        "header_file": header_basename,
        "baseline_variant": f"{bench_name}_0_baseline",
        "variants": [
            {
                "name": f"{bench_name}_0_baseline",
                "file": "hls_baseline.cpp",
                "source_path": str(src_cpp),
            }
        ],
        "variant_source_paths": {
            f"{bench_name}_0_baseline": str(src_cpp),
        },
        "plain_c_file": "plain.cpp",
        "testbench_file": "testbench.cpp" if tb_text else None,
        "kernel_top": top,
        "hls_top": top,
        "translated_hls_top": top,
        "support_files": [],
        "include_dirs": [],
        "supports_csim": bool(tb_text),
        # HLSFactory testbenches just print_array() with no pass/fail check, so a
        # cosim PASS means "the synthesized RTL ran cleanly through the testbench"
        # — useful for catching RTL-level bugs csim missed (false-dep, scheduler
        # races, BFM AXI issues) even though it does not verify numerical output.
        "supports_cosim": bool(tb_text),
        "cosim_depths": {},
        # cosim_testbench_file / cosim_support_files / cosim_size_overrides are
        # populated by gen_cosim_testbenches.py + macroize_benches.py — preserved
        # across re-ingest below.
        "cosim_testbench_file": None,
        "cosim_support_files": [],
        "cosim_size_overrides": {},
        "strip_report": {
            "removed_hls_pragmas": stripped_hls,
            "removed_accel_pragmas": stripped_accel,
            "removed_support_includes": 0,
            "removed_ap_int_includes": 0,
            "removed_extern_c_blocks": 0,
            "plain_contains_hls_pragmas": "#pragma HLS" in plain_text,
            "plain_contains_accel_pragmas": "#pragma ACCEL" in plain_text,
            "plain_contains_ap_uint": False,
        },
        "provenance": {
            "gold_hls_source_sha256": _sha256(gold_text),
            "gold_hls_baseline_sha256": _sha256(gold_text),
            "plain_c_sha256": _sha256(plain_text),
            "plain_derived_from_gold_hls": True,
        },
        "header_source_path": str(src_h),
    }
    if not dry_run:
        # Preserve cosim post-processing fields if the existing metadata has them
        # (added by gen_cosim_testbenches.py / macroize_benches.py).
        existing_path = out_dir / "metadata.json"
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text())
                for key in ("cosim_testbench_file", "cosim_support_files",
                            "cosim_size_overrides"):
                    if existing.get(key):
                        meta[key] = existing[key]
            except json.JSONDecodeError:
                pass
        existing_path.write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--only", type=str, default="", help="comma-separated bench names (without 'hlsfactory_' prefix)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-index-update", action="store_true",
                   help="do not modify benchmarks/index.json")
    args = p.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"root not found: {args.root}")

    only = {n.strip() for n in args.only.split(",") if n.strip()}

    existing_index = []
    existing_names: set[str] = set()
    if INDEX_PATH.is_file():
        existing_index = json.loads(INDEX_PATH.read_text())
        existing_names = {entry["benchmark"] for entry in existing_index}

    new_entries: list[dict] = []
    skipped = 0
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        if only and d.name not in only:
            continue
        meta = _ingest_one(d, BENCHMARKS_DIR, args.dry_run)
        if meta is None:
            skipped += 1
            continue
        new_entries.append(meta)
        print(f"  + {meta['benchmark']}  top={meta['kernel_top']}"
              f"  pragmas_stripped={meta['strip_report']['removed_hls_pragmas']}"
              f"  csim={'Y' if meta['supports_csim'] else 'N'}")

    print(f"\nIngested {len(new_entries)} benchmarks ({skipped} skipped)")

    if not args.dry_run and not args.no_index_update:
        # Merge: keep existing non-hlsfactory entries + replace hlsfactory entries
        merged = [e for e in existing_index if not e.get("benchmark", "").startswith("hlsfactory_")]
        merged.extend(new_entries)
        # Sort: original order for legacy, then alpha for hlsfactory
        legacy = [e for e in merged if not e.get("benchmark", "").startswith("hlsfactory_")]
        hlsf = sorted([e for e in merged if e.get("benchmark", "").startswith("hlsfactory_")],
                      key=lambda e: e["benchmark"])
        final = legacy + hlsf
        INDEX_PATH.write_text(json.dumps(final, indent=2))
        print(f"Updated {INDEX_PATH} ({len(final)} entries total)")
    elif args.dry_run:
        print("(dry-run: no files written)")


if __name__ == "__main__":
    main()
