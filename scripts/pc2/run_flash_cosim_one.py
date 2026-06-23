#!/usr/bin/env python3
"""Run standalone cosim for one flash matrix cell (by cell_id or manifest index)."""

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
    cell_result_path,
    cosim_run_root,
    find_cell,
    find_cell_by_index,
    load_manifest,
    manifest_path,
    run_cell_cosim,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Cosim run root (contains manifest.json)")
    parser.add_argument("--cell-id", default="", help="Cell id from manifest")
    parser.add_argument("--index", type=int, default=-1, help="Manifest cell index (for Slurm array)")
    parser.add_argument("--force", action="store_true", help="Re-run even if cosim_result.json exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without Vitis")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not manifest_path(run_root).exists():
        print(f"ERROR: missing manifest: {manifest_path(run_root)}", file=sys.stderr)
        return 2

    manifest = load_manifest(run_root)
    if args.cell_id:
        cell = find_cell(manifest, args.cell_id)
    elif args.index >= 0:
        cell = find_cell_by_index(manifest, args.index)
    else:
        print("ERROR: pass --cell-id or --index", file=sys.stderr)
        return 2

    result = run_cell_cosim(cell, run_root, force=args.force, dry_run=args.dry_run)
    out_path = cell_result_path(run_root, cell.cell_id)
    print(json.dumps({"cell_id": cell.cell_id, "out": str(out_path), **result}, indent=2))
    if result.get("status") == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
