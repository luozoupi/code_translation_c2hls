#!/usr/bin/env python3
"""Prepare bench copies + jobs.jsonl for fill_ab_20260730 (no edits to benchmarks_cosim)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fill_ab_timeout_lib import (  # noqa: E402
    FILL_ROOT,
    SCOPE,
    build_jobs,
    prepare_bench_copy,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    FILL_ROOT.mkdir(parents=True, exist_ok=True)
    prepared = []
    for model_tag, benches in SCOPE.items():
        for bench in benches:
            for size_kind in ("full", "small"):
                if args.dry_run:
                    prepared.append({"model_tag": model_tag, "bench": bench, "size_kind": size_kind})
                    continue
                dest = prepare_bench_copy(model_tag=model_tag, bench=bench, size_kind=size_kind)
                prepared.append({"model_tag": model_tag, "bench": bench, "size_kind": size_kind, "dest": str(dest)})
                print(f"prepared {size_kind} {model_tag} {bench} -> {dest}")

    jobs = build_jobs()
    jobs_path = FILL_ROOT / "jobs.jsonl"
    if not args.dry_run:
        with jobs_path.open("w", encoding="utf-8") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")
        (FILL_ROOT / "README.md").write_text(
            "# fill_ab_20260730\n\n"
            "Copies only. Original `benchmarks_cosim/` untouched.\n\n"
            "- `benches_full/`: full problem size → **csynth_full**\n"
            "- `benches_small/`: reduced macros → **cosim_small**\n"
            "- `work/`: per-job results\n"
            "- `csv/`: annotated fill CSVs\n",
            encoding="utf-8",
        )
    print(f"jobs={len(jobs)} path={jobs_path}")
    print(json.dumps({"prepared": len(prepared), "jobs": len(jobs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
