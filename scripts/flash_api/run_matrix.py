#!/usr/bin/env python3
"""Run multiple flash API profiles sequentially (deterministic matrix or top-5).

Examples::

    python3 scripts/flash_api/run_matrix.py --set top5 --dry-run
    python3 scripts/flash_api/run_matrix.py --set deterministic --model claude-sonnet-4-6
    python3 scripts/flash_api/run_matrix.py --profiles nav_o,aav_n,nav_n
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "flash_api"))

from flash_api_lib import DETERMINISTIC_ORDER, PROFILES, TOP5_ORDER, resolve_model_id

SETS = {
    "top5": TOP5_ORDER,
    "deterministic": DETERMINISTIC_ORDER,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run flash API profile sets sequentially")
    parser.add_argument("--set", type=str, choices=sorted(SETS), default="", help="Predefined profile set")
    parser.add_argument("--profiles", type=str, default="", help="Comma-separated profile keys (overrides --set)")
    parser.add_argument("--stamp", type=str, default="", help="Shared stamp for all profiles in this wave")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--benches", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-cosim",
        action="store_true",
        help="Skip cosim for every profile in this matrix (see run_flash_batch.py).",
    )
    args = parser.parse_args()

    if args.profiles:
        keys = [k.strip() for k in args.profiles.split(",") if k.strip()]
    elif args.set:
        keys = list(SETS[args.set])
    else:
        parser.error("pass --set top5|deterministic or --profiles k1,k2,...")

    unknown = [k for k in keys if k not in PROFILES]
    if unknown:
        parser.error(f"unknown profile(s): {unknown}")

    stamp = args.stamp or os.getenv("C2HLS_FLASH_API_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    model = resolve_model_id(args.model)
    runner = REPO / "scripts" / "flash_api" / "run_flash_batch.py"

    print(f"flash_api matrix stamp={stamp} model={model} profiles={keys}")
    rc = 0
    for key in keys:
        cmd = [
            sys.executable,
            str(runner),
            "--profile",
            key,
            "--stamp",
            stamp,
            "--model",
            model,
        ]
        if args.benches:
            cmd.extend(["--benches", args.benches])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.skip_cosim:
            cmd.append("--skip-cosim")
        print(f"\n=== {key} ===", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0:
            rc = proc.returncode
            print(f"FAILED profile={key} rc={rc}", flush=True)
            if not args.dry_run:
                break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
