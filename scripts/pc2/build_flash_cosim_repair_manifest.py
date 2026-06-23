#!/usr/bin/env python3
"""Build run manifest for 3-session cosim repair (no Slurm array)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from scripts.pc2.flash_cosim_repair_lib import (  # noqa: E402
    DEFAULT_COSIM_RUN,
    REPAIR_SESSION_KEYS,
    build_session_jobs,
    discover_failures_for_artifact,
    get_session_config,
    repair_run_root,
    write_run_manifest,
    write_session_manifest,
    session_run_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="")
    parser.add_argument("--cosim-run", default=str(DEFAULT_COSIM_RUN))
    args = parser.parse_args()

    cosim_root = Path(args.cosim_run)
    run_root = repair_run_root(args.stamp or None)

    sessions = {}
    for key in REPAIR_SESSION_KEYS:
        cfg = get_session_config(key)
        failures = discover_failures_for_artifact(cfg["artifact_basename"], cosim_root)
        jobs = build_session_jobs(
            failures,
            repair_variant=cfg["repair_variant"],
            session_key=key,
        )
        sessions[key] = jobs
        write_session_manifest(
            session_run_root(run_root, key),
            key,
            jobs,
            cosim_run_root=cosim_root,
        )

    path = write_run_manifest(run_root, cosim_run_root=cosim_root, sessions=sessions)
    summary = {
        "manifest": str(path),
        "run_root": str(run_root),
        "sessions": {
            key: {"failures": len(jobs), "repair_variant": get_session_config(key)["repair_variant"]}
            for key, jobs in sessions.items()
        },
        "total_jobs": sum(len(j) for j in sessions.values()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
