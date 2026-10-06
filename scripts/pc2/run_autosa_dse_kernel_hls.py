#!/usr/bin/env python3
"""Run c2hls-clock csynth+cosim for all exported packages of one AutoSA kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.autosa_dse_hls_common import (  # noqa: E402
    DEFAULT_CLOCK_NS,
    _bench_name,
    iter_export_packages,
)

DEFAULT_OUT = REPO / "AutoSA_sources"
from scripts.pc2.autosa_dse_hls_validate_lib import (  # noqa: E402
    kernel_cosim_timeout,
    validate_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--sources-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--resynth-only", action="store_true")
    args = parser.parse_args()

    bench = _bench_name(args.kernel_id)
    packages = iter_export_packages(args.sources_root, kernel_ids=[args.kernel_id])
    if not packages:
        print(f"no packages for kernel_id={args.kernel_id} bench={bench}", file=sys.stderr)
        return 1

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    cosim_timeout = kernel_cosim_timeout(args.kernel_id)

    results = []
    for pkg_dir in packages:
        print(f"==> {pkg_dir}", flush=True)
        results.append(
            validate_package(
                pkg_dir,
                clock_ns=args.clock_ns,
                cosim_timeout_s=cosim_timeout,
                resynth_only=args.resynth_only,
            )
        )

    summary = {
        "kernel_id": args.kernel_id,
        "bench": bench,
        "clock_ns": args.clock_ns,
        "cosim_timeout_s": cosim_timeout,
        "resynth_only": args.resynth_only,
        "packages": results,
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") != "ok"),
    }
    out = run_root / f"{args.kernel_id}_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
