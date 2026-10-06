#!/usr/bin/env python3
"""Run one fill_ab job (csynth_full or cosim_small; gold or flash)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO))

from fill_ab_timeout_lib import FILL_ROOT, run_one_job  # noqa: E402


def load_job(index: int) -> dict:
    jobs_path = FILL_ROOT / "jobs.jsonl"
    with jobs_path.open(encoding="utf-8") as f:
        for line in f:
            job = json.loads(line)
            if int(job["index"]) == int(index):
                return job
    raise SystemExit(f"job index {index} not found in {jobs_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    job = load_job(args.index)
    print(json.dumps({"starting": job}, indent=2))
    payload = run_one_job(job, force=args.force)
    print(json.dumps({k: payload.get(k) for k in ("status", "metric_kind", "side", "latency_cycles", "kernel_runtime_cycles", "error", "runtime_seconds")}, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
