#!/usr/bin/env python3
"""Build manifest of naive hlsfactory hls_baseline.cpp cells for standalone cosim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.pc2.flash_cosim_lib import (  # noqa: E402
    cosim_benchmarks_root,
    cosim_full_size_enabled,
    cosim_run_root,
    discover_baseline_cells,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="", help="Cosim run stamp (default: new UTC stamp)")
    parser.add_argument("--bench", action="append", default=[], help="Limit to benchmark name")
    parser.add_argument("--run-root", default="", help="Override cosim output root parent")
    parser.add_argument("--dry-run", action="store_true", help="Print summary only")
    parser.add_argument(
        "--full-size",
        action="store_true",
        help="Cosim at header problem size (no cosim_size_overrides; matches csynth N)",
    )
    args = parser.parse_args()

    if args.full_size:
        import os

        os.environ["C2HLS_FLASH_COSIM_FULL_SIZE"] = "1"

    if args.run_root:
        import os

        os.environ["C2HLS_FLASH_COSIM_ROOT"] = args.run_root
    if args.stamp:
        import os

        os.environ["C2HLS_FLASH_COSIM_STAMP"] = args.stamp

    cells = discover_baseline_cells(
        bench_filter=set(args.bench) if args.bench else None,
    )
    run_root = cosim_run_root(args.stamp or None)

    summary = {
        "run_root": str(run_root),
        "cell_count": len(cells),
        "benches": sorted({c.bench for c in cells}),
        "supports_cosim": sum(1 for c in cells if c.supports_cosim),
        "cosim_size_mode": "full" if cosim_full_size_enabled() else "override",
        "manifest_kind": "baseline",
        "benchmarks_root": str(cosim_benchmarks_root()),
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return 0

    path = write_manifest(
        run_root,
        cells,
        extra={
            "manifest_kind": "baseline",
            "cosim_size_mode": "full" if cosim_full_size_enabled() else "override",
            "benchmarks_root": str(cosim_benchmarks_root()),
            "cosim_kernel_file": "hls_baseline_cosim.cpp",
            "corpus": "benchmarks_cosim",
        },
    )
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
