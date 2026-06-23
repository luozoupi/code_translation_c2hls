#!/usr/bin/env python3
"""Sequential cosim repair batch for one PC2 session (GPU+compute pair)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from c2hls_paths import apply_runtime_defaults, configure_site  # noqa: E402
from c2hls_temp import configure_temp_env  # noqa: E402
from scripts.pc2.flash_cosim_repair_lib import (  # noqa: E402
    DEFAULT_COSIM_RUN,
    DEFAULT_MAX_REPAIR_LOOPS,
    REPAIR_SESSION_KEYS,
    build_session_jobs,
    discover_failures_for_artifact,
    get_session_config,
    repair_run_root,
    run_repair_batch,
    write_run_manifest,
)


def _configure_pc2_env() -> None:
    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)
    os.environ.setdefault("C2HLS_RUN_COSIM", "1")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def _all_session_jobs(cosim_root: Path) -> dict:
    sessions = {}
    for key in REPAIR_SESSION_KEYS:
        cfg = get_session_config(key)
        failures = discover_failures_for_artifact(cfg["artifact_basename"], cosim_root)
        sessions[key] = build_session_jobs(
            failures,
            repair_variant=cfg["repair_variant"],
            session_key=key,
        )
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc2", action="store_true", help="Configure PC2 runtime defaults")
    parser.add_argument("--session", required=True, choices=REPAIR_SESSION_KEYS)
    parser.add_argument("--stamp", default="")
    parser.add_argument("--cosim-run", default=str(DEFAULT_COSIM_RUN))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-loops",
        type=int,
        default=int(os.getenv("C2HLS_COSIM_REPAIR_MAX_LOOPS", str(DEFAULT_MAX_REPAIR_LOOPS))),
        help="LLM diagnose+repair iterations per failing kernel (default: 1)",
    )
    args = parser.parse_args()

    if args.pc2:
        configure_site("pc2")
        _configure_pc2_env()

    if args.stamp:
        os.environ["C2HLS_FLASH_COSIM_REPAIR_STAMP"] = args.stamp

    cosim_root = Path(args.cosim_run)
    run_root = repair_run_root(args.stamp or None)
    os.environ["C2HLS_FLASH_COSIM_REPAIR_ROOT"] = str(run_root)

    summary = run_repair_batch(
        args.session,
        run_root,
        cosim_root,
        max_loops=args.max_loops,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))

    write_run_manifest(
        run_root,
        cosim_run_root=cosim_root,
        sessions=_all_session_jobs(cosim_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
