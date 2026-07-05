#!/usr/bin/env python3
"""Emit tier_A_ready bench list files (remaining / full corpus)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIER_A_READY = REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"
PC2 = REPO / "scripts/pc2"
TESTED_24 = json.loads((PC2 / "batch_parallel_tier_a_24_csim.json").read_text())["pilot"]["benches"]


def list_ready_benches() -> list[str]:
    return sorted(
        p.name
        for p in TIER_A_READY.iterdir()
        if p.is_dir() and (p / "metadata.json").is_file()
    )


def write_list(path: Path, benches: list[str]) -> None:
    path.write_text("\n".join(benches) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        choices=("remaining", "full", "all"),
        default="all",
        help="Which list files to refresh",
    )
    args = parser.parse_args()

    all_benches = list_ready_benches()
    tested = set(TESTED_24)
    remaining = [b for b in all_benches if b not in tested]

    if args.write in ("remaining", "all"):
        write_list(PC2 / "tier_a_30_remaining_benches.txt", remaining)
        print(f"wrote tier_a_30_remaining_benches.txt ({len(remaining)} benches)")
    if args.write in ("full", "all"):
        write_list(PC2 / "tier_a_54_benches.txt", all_benches)
        print(f"wrote tier_a_54_benches.txt ({len(all_benches)} benches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
