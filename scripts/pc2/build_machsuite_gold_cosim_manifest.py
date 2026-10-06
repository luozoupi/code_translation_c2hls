#!/usr/bin/env python3
"""Build standalone cosim manifest for MachSuite gold (hls_baseline.cpp) fill-ins.

Default benches = those missing gold cosim cycles in the Jul 10 MachSuite CSV:
  machsuite_aes_table, machsuite_aes_tableless, machsuite_backprop
"""

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
    CosimCell,
    cosim_full_size_enabled,
    cosim_run_root,
    make_cell_id,
    write_manifest,
)

TIER_B_READY = REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready"
DEFAULT_MISSING = [
    "machsuite_aes_table",
    "machsuite_aes_tableless",
    "machsuite_backprop",
]


def discover_machsuite_gold_cells(
    *,
    benches: list[str],
    root: Path = TIER_B_READY,
) -> list[CosimCell]:
    cells: list[CosimCell] = []
    artifact_basename = "machsuite_tier_b_gold_baseline"
    setup_tag = "hls_baseline"
    for index, bench in enumerate(benches):
        bench_dir = root / bench
        meta_path = bench_dir / "metadata.json"
        baseline_cpp = bench_dir / "hls_baseline.cpp"
        if not meta_path.is_file() or not baseline_cpp.is_file():
            raise FileNotFoundError(f"missing gold inputs for {bench} under {bench_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("supports_cosim", True):
            raise ValueError(f"{bench}: supports_cosim is false")
        cell_id = make_cell_id(artifact_basename, bench, setup_tag)
        cells.append(
            CosimCell(
                index=index,
                cell_id=cell_id,
                artifact_dir=str(root),
                artifact_basename=artifact_basename,
                artifact_stamp="gold_fill",
                matrix_family="machsuite_gold_baseline",
                bench=bench,
                setup_tag=setup_tag,
                variant=setup_tag,
                mode="baseline",
                model="",
                curation_focus="",
                skills_json="",
                cell_dir=str(bench_dir),
                final_cpp=str(baseline_cpp.resolve()),
                kernel_source="hls_baseline",
                source_matrix_status="ok",
                supports_cosim=True,
            )
        )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="", help="Cosim run stamp")
    parser.add_argument(
        "--bench",
        action="append",
        default=[],
        help="Bench to include (repeatable; default=missing gold trio)",
    )
    parser.add_argument("--run-root", default="", help="Override C2HLS_FLASH_COSIM_ROOT parent")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-size", action="store_true", default=True)
    args = parser.parse_args()

    import os

    if args.full_size:
        os.environ["C2HLS_FLASH_COSIM_FULL_SIZE"] = "1"
    if args.run_root:
        os.environ["C2HLS_FLASH_COSIM_ROOT"] = args.run_root
    if args.stamp:
        os.environ["C2HLS_FLASH_COSIM_STAMP"] = args.stamp

    benches = args.bench or list(DEFAULT_MISSING)
    cells = discover_machsuite_gold_cells(benches=benches)
    run_root = cosim_run_root(args.stamp or None)
    summary = {
        "run_root": str(run_root),
        "cell_count": len(cells),
        "benches": [c.bench for c in cells],
        "cosim_size_mode": "full" if cosim_full_size_enabled() else "override",
        "manifest_kind": "machsuite_gold_baseline",
        "benchmarks_root": str(TIER_B_READY),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    path = write_manifest(
        run_root,
        cells,
        extra={
            "manifest_kind": "machsuite_gold_baseline",
            "cosim_size_mode": "full" if cosim_full_size_enabled() else "override",
            "benchmarks_root": str(TIER_B_READY),
            "cosim_kernel_file": "hls_baseline.cpp",
            "corpus": "tier_B_ready",
        },
    )
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
