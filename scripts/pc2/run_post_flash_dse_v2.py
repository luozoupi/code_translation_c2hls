#!/usr/bin/env python3
"""Run DSE 2.0 PE×SIMD sweep on existing flash-final cells (no stream).

Example (attach to an existing campaign cell without re-running flash)::

    C2HLS_DSE_V2=1 python3 scripts/pc2/run_post_flash_dse_v2.py --pc2 \\
      --matrix-root artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_repro3 \\
      --benches autosa_mm --dry-run

Never point --matrix-root at frozen 20260830_mmflow or other protected stamps
unless you intentionally want a new sibling write path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_paths import BENCHMARKS_DIR, configure_site
from post_flash_dse import discover_dse_cells
from post_flash_dse_v2 import (
    configure_post_flash_dse_v2_env,
    dse_v2_enabled,
    expand_dse_v2_trials,
    load_dse_v2_grid,
    parse_ijk_from_header,
    resolve_dse_v2_grid_path,
    run_dse_v2_for_cell,
)


_FROZEN_MARKERS = (
    "20260830_mmflow",
    "autosa_mm_variant_sweep_20260918",
    "_repro2",
    "_repro3",
    "_repro4",
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_bench_dir(bench: str) -> Path:
    ready = os.getenv("C2HLS_AUTOSA_READY_ROOT", "").strip()
    candidates = []
    if ready:
        candidates.append(Path(ready) / bench)
    candidates += [
        BENCHMARKS_DIR / bench,
        REPO / "related_work/benchmarks/autosa_ready" / bench,
        REPO / "benchmarks_autosa_dse" / bench,
        REPO / "benchmarks_autosa" / bench,
    ]
    for path in candidates:
        if (path / "metadata.json").is_file():
            return path
    raise ValueError(f"unknown benchmark: {bench}")


def _refuse_frozen(matrix_root: Path, *, force: bool) -> None:
    name = str(matrix_root)
    if force:
        return
    for marker in _FROZEN_MARKERS:
        if marker in name:
            raise SystemExit(
                f"Refusing to write DSE 2.0 into frozen/repro tree {matrix_root}. "
                f"Copy the cell elsewhere or pass --force-frozen (not recommended)."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc2", action="store_true")
    parser.add_argument("--fir", action="store_true")
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--benches", default="autosa_mm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-frozen", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    if args.pc2:
        configure_site("pc2")
    elif args.fir:
        configure_site("fir")

    os.environ["C2HLS_DSE_V2"] = "1"
    os.environ["C2HLS_DSE_V2_CHAIN_FLASH"] = "1"
    os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
    os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
    configure_post_flash_dse_v2_env()

    matrix_root = args.matrix_root.resolve()
    if not args.dry_run:
        _refuse_frozen(matrix_root, force=args.force_frozen)
    skip_existing = not args.no_skip_existing
    benches = set(_split_csv(args.benches))
    cells = [
        c
        for c in discover_dse_cells(matrix_root)
        if not benches or c.get("bench") in benches
    ]
    grid = load_dse_v2_grid()
    print(f"dse_v2_enabled={dse_v2_enabled()} grid={resolve_dse_v2_grid_path()}")
    print(f"cells={len(cells)} skip_existing={skip_existing}")

    if args.dry_run:
        for cell in cells:
            bench = cell["bench"]
            bench_dir = _resolve_bench_dir(bench)
            header = (bench_dir / "kernel.h").read_text(encoding="utf-8")
            i, j, k = parse_ijk_from_header(header)
            trials = expand_dse_v2_trials(i=i, j=j, k=k, data_kind="float", grid=grid)
            print(
                f"  {bench} cell={cell['cell_dir']} I={i} trials={len(trials)} "
                f"ids={[t.trial_id for t in trials]}"
            )
        return 0

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID
    from post_flash_dse import dse_max_tokens, repair_round_limit

    model = os.getenv("C2HLS_MODEL", "").strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(
        gpt_model=model,
        turns_limitation=repair_round_limit(),
        max_completion_tokens=dse_max_tokens(),
    )
    outcomes: list[dict[str, Any]] = []
    for cell in cells:
        bench = cell["bench"]
        cell_dir = Path(cell["cell_dir"])
        bench_dir = _resolve_bench_dir(bench)
        print(f"=== dse_v2 {bench} @ {cell_dir} ===", flush=True)
        outcome = run_dse_v2_for_cell(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orch,
            source_role="flash_final",
            skip_existing=skip_existing,
        )
        outcomes.append(
            {
                "bench": bench,
                "cell_dir": str(cell_dir),
                "success": outcome.success,
                "error": outcome.error,
                "winner": (outcome.result or {}).get("winner_trial_id")
                if isinstance(outcome.result, dict)
                else None,
                "latency_cycles": (outcome.result or {}).get("latency_cycles")
                if isinstance(outcome.result, dict)
                else None,
                "dsp": (outcome.result or {}).get("dsp")
                if isinstance(outcome.result, dict)
                else None,
            }
        )

    summary = {
        "schema": "run_post_flash_dse_v2_summary_v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "matrix_root": str(matrix_root),
        "outcomes": outcomes,
    }
    out = matrix_root / "dse_v2_batch_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(o.get("success") for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
