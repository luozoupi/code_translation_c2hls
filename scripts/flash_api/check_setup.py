#!/usr/bin/env python3
"""Preflight only — verify team env is ready for flash API runs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "flash_api"))

from flash_api_lib import resolve_model_id
from flash_shared.team_env import (
    active_team_paths_summary,
    bootstrap_team_flash_env,
    flash_cosim_manifest,
    preflight_api_run,
    resolve_skip_cosim,
)


def main() -> int:
    bootstrap_team_flash_env()
    model = resolve_model_id()
    cosim = flash_cosim_manifest()
    print(f"model={model}")
    print(
        f"validation: skip_cosim={cosim['skip_cosim']} "
        f"RUN_COSIM={cosim['C2HLS_RUN_COSIM']} "
        f"COSIM_REQUIRED={cosim['C2HLS_COSIM_REQUIRED']}"
    )
    if resolve_skip_cosim():
        print("  (cosim skipped — use default or unset C2HLS_FLASH_API_SKIP_COSIM for full cosim)")
    print("team_paths:")
    for k, v in active_team_paths_summary().items():
        print(f"  {k}={v}")
    blockers = preflight_api_run(model)
    if blockers:
        print("\nBLOCKERS:")
        for msg in blockers:
            print(f"  - {msg}")
        return 2
    print("\nOK — ready for scripts/flash_api/run_flash_batch.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
