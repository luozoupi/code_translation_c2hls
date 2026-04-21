#!/usr/bin/env python3
"""Corpus hygiene validator for the C-to-HLS benchmark suite.

Checks each benchmark directory for the invariants the RL pipeline relies on:
  - metadata.json is present and parseable
  - plain.cpp exists and is free of HLS-leak tokens (pragmas, ap_uint, mars_wide_bus_type)
  - header file named by metadata is present
  - every variant file referenced in metadata exists on disk
  - if supports_csim is true, testbench.cpp exists
  - support files listed in metadata all exist
  - benchmark status is labelled ("ready" / "disabled")

On failure a benchmark is reported as failing; if --mark-disabled is passed, its
metadata is updated in place with status="disabled" and disabled_reason=<why>.

Exits 0 only if every non-disabled benchmark passes.

Usage:
    python validate_corpus.py                # all benchmarks, report only
    python validate_corpus.py --bench nw     # single benchmark
    python validate_corpus.py --mark-disabled   # persist status on failure
    python validate_corpus.py --json            # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

# Substrings that must NOT appear in plain.cpp (would indicate incomplete strip).
# Patterns are case-insensitive and matched against source lines.
STRIP_LEAK_PATTERNS = [
    (re.compile(r"#pragma\s+HLS\b", re.IGNORECASE), "HLS pragma"),
    (re.compile(r"#pragma\s+ACCEL\b", re.IGNORECASE), "ACCEL pragma"),
    (re.compile(r"\bap_uint\s*<"), "ap_uint<...> (HLS wide-bus type)"),
    (re.compile(r"\bap_int\s*<"), "ap_int<...> (HLS wide-bus type)"),
    (re.compile(r"\bMARS_WIDE_BUS_TYPE\b"), "MARS_WIDE_BUS_TYPE macro"),
    (re.compile(r"\bmemcpy_wide_bus_"), "memcpy_wide_bus_* helper"),
    (re.compile(r'#include\s*["<]support/common/mars_wide_bus'), "mars_wide_bus header include"),
    (re.compile(r'extern\s+"C"\s*\{'), 'extern "C" { block (should be stripped)'),
]


@dataclass
class Issue:
    kind: str       # "error" | "warn"
    code: str       # short stable code for filtering
    message: str


@dataclass
class BenchmarkReport:
    benchmark: str
    path: str
    status: str = "ready"            # "ready" | "disabled" | "failed"
    current_status: str = ""          # whatever metadata.json recorded before validation
    disabled_reason: str = ""
    issues: List[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.kind == "error" for i in self.issues)

    def error(self, code: str, msg: str) -> None:
        self.issues.append(Issue("error", code, msg))

    def warn(self, code: str, msg: str) -> None:
        self.issues.append(Issue("warn", code, msg))


def _load_metadata(bench_dir: Path, report: BenchmarkReport) -> Optional[dict]:
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        report.error("missing_metadata", f"metadata.json not found at {meta_path}")
        return None
    try:
        with meta_path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        report.error("bad_metadata", f"metadata.json is not valid JSON: {exc}")
        return None


def _check_plain_cpp(bench_dir: Path, report: BenchmarkReport) -> None:
    plain = bench_dir / "plain.cpp"
    if not plain.exists():
        report.error("missing_plain_cpp", "plain.cpp not present")
        return
    try:
        text = plain.read_text()
    except OSError as exc:
        report.error("unreadable_plain_cpp", f"cannot read plain.cpp: {exc}")
        return
    if not text.strip():
        report.error("empty_plain_cpp", "plain.cpp is empty")
        return
    lines = text.splitlines()
    for pattern, label in STRIP_LEAK_PATTERNS:
        for i, line in enumerate(lines, start=1):
            # Skip single-line comments — a commented-out pragma is not a leak.
            stripped = line.lstrip()
            if stripped.startswith("//"):
                continue
            if pattern.search(line):
                snippet = line.strip()[:120]
                report.error(
                    "strip_leak",
                    f"plain.cpp line {i} still contains {label}: {snippet!r}",
                )
                break  # one hit per pattern is enough


def _check_header(bench_dir: Path, meta: dict, report: BenchmarkReport) -> None:
    header_name = meta.get("header_file")
    if not header_name:
        return
    header_path = bench_dir / header_name
    if not header_path.exists():
        report.error("missing_header", f"header file {header_name!r} referenced in metadata not found")


def _check_variants(bench_dir: Path, meta: dict, report: BenchmarkReport) -> None:
    variants = meta.get("variants") or []
    if not variants:
        # Single-variant benchmarks fall back to gold_hls_baseline_file.
        baseline = meta.get("gold_hls_baseline_file", "hls_baseline.cpp")
        if not (bench_dir / baseline).exists():
            report.error("missing_baseline", f"gold_hls_baseline_file {baseline!r} not present")
        return

    seen_files = set()
    for i, v in enumerate(variants):
        vfile = v.get("file")
        if not vfile:
            report.error("variant_no_file", f"variants[{i}] has no 'file' key")
            continue
        if vfile in seen_files:
            report.warn("variant_duplicate", f"variant file {vfile!r} listed more than once")
        seen_files.add(vfile)
        if not (bench_dir / vfile).exists():
            report.error("missing_variant_file", f"variants[{i}] file {vfile!r} not on disk")

    # Last variant is the RL ground truth; make sure it's present.
    last = variants[-1]
    last_file = last.get("file")
    if last_file and not (bench_dir / last_file).exists():
        report.error(
            "missing_last_variant",
            f"variants[-1] {last_file!r} (RL ground truth) not on disk",
        )


def _check_testbench(bench_dir: Path, meta: dict, report: BenchmarkReport) -> None:
    if not meta.get("supports_csim"):
        return
    tb_file = meta.get("testbench_file") or "testbench.cpp"
    if not (bench_dir / tb_file).exists():
        report.error(
            "missing_testbench",
            f"supports_csim=true but testbench file {tb_file!r} not found",
        )


def _check_support_files(bench_dir: Path, meta: dict, report: BenchmarkReport) -> None:
    for rel in meta.get("support_files") or []:
        sp = bench_dir / rel
        if not sp.exists():
            report.error("missing_support_file", f"support_files entry {rel!r} not found")


def validate_benchmark(bench_dir: Path) -> BenchmarkReport:
    report = BenchmarkReport(benchmark=bench_dir.name, path=str(bench_dir))
    if not bench_dir.is_dir():
        report.error("not_a_dir", f"{bench_dir} is not a directory")
        report.status = "failed"
        return report

    meta = _load_metadata(bench_dir, report)
    if meta is None:
        report.status = "failed"
        return report

    report.current_status = meta.get("status", "")
    # If previously-marked disabled, skip detailed checks but keep the marker.
    if report.current_status == "disabled":
        report.status = "disabled"
        report.disabled_reason = meta.get("disabled_reason", "previously disabled")
        return report

    _check_plain_cpp(bench_dir, report)
    _check_header(bench_dir, meta, report)
    _check_variants(bench_dir, meta, report)
    _check_testbench(bench_dir, meta, report)
    _check_support_files(bench_dir, meta, report)

    if report.has_errors:
        report.status = "failed"
        # Reason for the --mark-disabled path.
        first_err = next(i for i in report.issues if i.kind == "error")
        report.disabled_reason = f"{first_err.code}: {first_err.message}"
    else:
        report.status = "ready"
    return report


def _mark_disabled(bench_dir: Path, report: BenchmarkReport) -> None:
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return
    with meta_path.open() as f:
        meta = json.load(f)
    meta["status"] = "disabled"
    meta["disabled_reason"] = report.disabled_reason
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def _print_report(reports: List[BenchmarkReport]) -> None:
    width = max((len(r.benchmark) for r in reports), default=10)
    ready = [r for r in reports if r.status == "ready"]
    disabled = [r for r in reports if r.status == "disabled"]
    failed = [r for r in reports if r.status == "failed"]

    for r in reports:
        status_sym = {"ready": "OK ", "disabled": "SKIP", "failed": "FAIL"}.get(r.status, "??? ")
        print(f"  {status_sym}  {r.benchmark:<{width}}  ({len(r.issues)} issues)")
        for issue in r.issues:
            prefix = "   error" if issue.kind == "error" else "   warn "
            print(f"    {prefix}  {issue.code}: {issue.message}")

    print()
    print(f"  ready:    {len(ready)}")
    print(f"  disabled: {len(disabled)}")
    print(f"  failed:   {len(failed)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", help="Validate a single benchmark by name")
    p.add_argument("--benchmarks-dir", default=str(BENCHMARKS_DIR),
                   help="Root directory containing per-benchmark folders")
    p.add_argument("--mark-disabled", action="store_true",
                   help="Persist status='disabled' + disabled_reason on failed benchmarks")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = p.parse_args()

    root = Path(args.benchmarks_dir)
    if not root.is_dir():
        print(f"error: benchmarks dir not found: {root}", file=sys.stderr)
        return 2

    if args.bench:
        targets = [root / args.bench]
        if not targets[0].is_dir():
            print(f"error: benchmark not found: {args.bench}", file=sys.stderr)
            return 2
    else:
        targets = sorted(p for p in root.iterdir() if p.is_dir())

    reports = [validate_benchmark(t) for t in targets]
    if args.mark_disabled:
        for r in reports:
            if r.status == "failed":
                _mark_disabled(Path(r.path), r)
                r.status = "disabled"

    if args.json:
        out = [
            {
                **asdict(r),
                "issues": [asdict(i) for i in r.issues],
            }
            for r in reports
        ]
        print(json.dumps(out, indent=2))
    else:
        _print_report(reports)

    # Exit non-zero when any benchmark is still in the failed bucket.
    return 1 if any(r.status == "failed" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
