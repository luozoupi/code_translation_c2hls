#!/usr/bin/env python3
"""
Scrub stale error fields from saved result artifacts.

This keeps current-state fields aligned with the latest successful run while
preserving superseded attempt failures in a less misleading form.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from c2hls import REPO_ROOT, sanitize_saved_result_record


def _iter_result_files(results_dir: Path) -> list[Path]:
    files = []
    for bench_dir in sorted(results_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        files.extend(sorted(bench_dir.glob("*_results.json")))
        files.extend(sorted(bench_dir.glob("*_multistep_results.json")))
    return files


def _sanitize_file(path: Path) -> bool:
    original = json.loads(path.read_text())
    cleaned = sanitize_saved_result_record(original)
    changed = cleaned != original
    if changed:
        path.write_text(json.dumps(cleaned, indent=2, default=str) + "\n")
    return changed


def _has_multistep_results(results_dir: Path) -> bool:
    return any(results_dir.glob("*/*_multistep_results.json"))


def _regenerate_report(results_dir: Path, multistep: bool) -> None:
    cmd = [sys.executable, "report.py", "--results", str(results_dir)]
    if multistep:
        if not _has_multistep_results(results_dir):
            return
        cmd.extend(["--multistep", "--output", "report_multistep.html"])
    else:
        cmd.extend(["--output", "report.html"])
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize saved benchmark result JSONs")
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "results"),
        help="Results directory to sanitize",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Do not regenerate HTML reports after sanitizing",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    changed = 0
    for path in _iter_result_files(results_dir):
        if _sanitize_file(path):
            changed += 1

    if not args.skip_report:
        _regenerate_report(results_dir, multistep=False)
        _regenerate_report(results_dir, multistep=True)

    print(json.dumps({"changed_files": changed}, indent=2))


if __name__ == "__main__":
    main()
