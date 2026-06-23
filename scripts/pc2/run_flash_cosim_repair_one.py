#!/usr/bin/env python3
"""Run one cosim repair job from a session manifest (re-run / debug)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from scripts.pc2.flash_cosim_repair_lib import (  # noqa: E402
    RepairJob,
    job_dir,
    run_repair_job,
)


def _load_session_job(session_root: Path, *, bench: str = "", job_id: str = "") -> RepairJob:
    manifest_path = session_root / "session_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing session manifest: {manifest_path}")
    raw = json.loads(manifest_path.read_text())
    for row in raw.get("jobs", []):
        if bench and row.get("bench") == bench:
            return RepairJob(**row)
        if job_id and row.get("job_id") == job_id:
            return RepairJob(**row)
    needle = bench or job_id
    raise KeyError(f"job not found in session manifest: {needle}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, help="e.g. artifacts/pc2/flash_cosim_repair/<stamp>/all_avoids_new")
    parser.add_argument("--bench", default="", help="bench name (unique within session)")
    parser.add_argument("--job-id", default="", help="source cell id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-loops", type=int, default=1)
    args = parser.parse_args()

    if not args.bench and not args.job_id:
        print("ERROR: --bench or --job-id required", file=sys.stderr)
        return 2

    session_root = Path(args.session_root)
    job = _load_session_job(session_root, bench=args.bench, job_id=args.job_id)
    result = run_repair_job(
        job,
        session_root,
        max_loops=args.max_loops,
        force=args.force,
        dry_run=args.dry_run,
    )
    out = job_dir(session_root, job.job_id) / "repair_result.json"
    print(json.dumps({"job_id": job.job_id, "bench": job.bench, "out": str(out), **result}, indent=2))
    return 0 if result.get("status") in ("ok", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
