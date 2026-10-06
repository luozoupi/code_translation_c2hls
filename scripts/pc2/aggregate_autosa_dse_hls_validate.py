#!/usr/bin/env python3
"""Aggregate per-kernel HLS validate summaries into one report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    kernels_dir = run_root / "kernels"
    summaries = []
    for path in sorted(kernels_dir.glob("*/*_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    packages = []
    for summary in summaries:
        packages.extend(summary.get("packages", []))

    report = {
        "schema": "autosa_dse_hls_validate_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_root": str(run_root),
        "clock_ns": 3.33,
        "part": "xcu280-fsvh2892-2L-e",
        "kernel_count": len(summaries),
        "package_count": len(packages),
        "ok": sum(1 for p in packages if p.get("status") == "ok"),
        "failed": sum(1 for p in packages if p.get("status") != "ok"),
        "kernels": summaries,
        "packages": packages,
    }
    out = args.out or (run_root / "hls_validate_report.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "ok": report["ok"], "failed": report["failed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
