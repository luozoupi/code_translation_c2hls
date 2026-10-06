#!/usr/bin/env python3
"""Build manifest of exported AutoSA DSE kernels for parallel HLS validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.autosa_dse_hls_common import DEFAULT_CLOCK_NS, DEFAULT_PART  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "AutoSA_sources"
from scripts.pc2.autosa_dse_hls_validate_lib import kernel_cosim_timeout, kernel_walltime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--export-report", type=Path, default=DEFAULT_OUT / "export_report.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        default="",
        help="Comma-separated kernel_id filter (e.g. large_mm,mm_catapult)",
    )
    args = parser.parse_args()

    kernel_filter: set[str] | None = None
    if args.kernels.strip():
        kernel_filter = {k.strip() for k in args.kernels.split(",") if k.strip()}

    report = json.loads(args.export_report.read_text(encoding="utf-8"))
    kernels = []
    for kernel_id, info in sorted(report.get("kernels", {}).items()):
        if kernel_filter is not None and kernel_id not in kernel_filter:
            continue
        if not info.get("exported"):
            continue
        kernels.append(
            {
                "kernel_id": kernel_id,
                "bench": info["bench"],
                "package_count": len(info["exported"]),
                "packages": [e["path"] for e in info["exported"]],
                "slurm_walltime": kernel_walltime(kernel_id),
                "cosim_timeout_s": kernel_cosim_timeout(kernel_id),
            }
        )

    manifest = {
        "schema": "autosa_dse_hls_validate_manifest_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources_root": str(args.sources_root.resolve()),
        "clock_ns": 3.33,
        "part": "xcu280-fsvh2892-2L-e",
        "device_platform": "xilinx_u280_gen3x16_xdma_1_202211_1",
        "kernel_count": len(kernels),
        "package_count": sum(k["package_count"] for k in kernels),
        "kernels": kernels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **{k: manifest[k] for k in ("kernel_count", "package_count")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
