#!/usr/bin/env python3
"""Post-flash memory parallelism batch (optional 2x / 4x step).

Reads flash matrix cells (e.g. ``flash_all_new_skills_avoids_global_<stamp>/``),
takes each bench's selected/final kernel, and runs the memory-parallel LLM step
with csim + csynth validation (cosim off).

Example::

    python3 scripts/pc2/run_post_flash_mem_parallel.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --dry-run

    python3 scripts/pc2/run_post_flash_mem_parallel.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --benches hlsfactory_gemm,hlsfactory_2mm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_paths import BENCHMARKS_DIR, configure_site
from post_flash_mem_parallel import (
    configure_post_flash_env,
    discover_matrix_cells,
    mem_parallel_factors,
    repair_round_limit,
    resolve_selected_kernel,
    run_memory_parallel_for_cell,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_bench_dir(bench: str) -> Path:
    meta = BENCHMARKS_DIR / bench / "metadata.json"
    if not meta.is_file():
        raise ValueError(f"unknown benchmark: {bench}")
    return BENCHMARKS_DIR / bench


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-flash memory parallelism (2x/4x)")
    parser.add_argument("--pc2", action="store_true", help="PC2 site paths + vLLM")
    parser.add_argument("--matrix-root", type=str, required=True)
    parser.add_argument("--benches", type=str, default="", help="Comma-separated filter")
    parser.add_argument("--factors", type=str, default="", help="e.g. 2,4")
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument("--turns", type=int, default=repair_round_limit())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if result exists")
    args = parser.parse_args()

    if args.pc2:
        configure_site("pc2")
    configure_post_flash_env()

    if args.factors.strip():
        os.environ["C2HLS_MEM_PARALLEL_FACTORS"] = args.factors.strip()
    os.environ["C2HLS_MEM_PARALLEL_REPAIR_ROUNDS"] = str(args.turns)

    matrix_root = Path(args.matrix_root).expanduser()
    if not matrix_root.is_absolute():
        matrix_root = REPO / matrix_root
    if not matrix_root.is_dir():
        print(f"matrix root missing: {matrix_root}", file=sys.stderr)
        return 1

    bench_filter = set(_split_csv(args.benches)) if args.benches else None
    factors = mem_parallel_factors()
    cells = discover_matrix_cells(matrix_root)
    if bench_filter:
        cells = [c for c in cells if c["bench"] in bench_filter]

    plan: list[dict[str, Any]] = []
    for cell in cells:
        bench = cell["bench"]
        cell_dir = Path(cell["cell_dir"])
        kpath, role = resolve_selected_kernel(cell_dir, bench)
        plan.append({
            "bench": bench,
            "cell_dir": str(cell_dir),
            "kernel": str(kpath) if kpath else None,
            "kernel_role": role,
            "factors": list(factors),
            "flash_status": cell.get("status"),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan_path = matrix_root / f"post_flash_mem_parallel_plan_{stamp}.json"
    plan_path.write_text(json.dumps({
        "matrix_root": str(matrix_root),
        "factors": list(factors),
        "repair_rounds": repair_round_limit(),
        "cells": plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"plan: {plan_path} ({len(plan)} cells)")

    if args.dry_run:
        for row in plan:
            print(f"  {row['bench']}: kernel={row['kernel']} factors={row['factors']}")
        return 0

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(gpt_model=model, turns_limitation=args.turns)

    summary: list[dict[str, Any]] = []
    for cell in plan:
        bench = cell["bench"]
        if not cell.get("kernel"):
            print(f"SKIP {bench}: no kernel", flush=True)
            summary.append({"bench": bench, "skipped": True, "reason": "no kernel"})
            continue
        print(f"START {bench}", flush=True)
        t0 = time.time()
        try:
            outcomes = run_memory_parallel_for_cell(
                bench=bench,
                bench_dir=_resolve_bench_dir(bench),
                cell_dir=Path(cell["cell_dir"]),
                orchestrator=orch,
                factors=factors,
                skip_existing=not args.force,
            )
        except Exception as exc:
            print(f"ERROR {bench}: {exc}", flush=True)
            summary.append({"bench": bench, "error": str(exc)})
            continue
        elapsed = round(time.time() - t0, 1)
        row = {
            "bench": bench,
            "elapsed_s": elapsed,
            "factors": [
                {"factor": o.factor, "success": o.success, "error": o.error}
                for o in outcomes
            ],
        }
        summary.append(row)
        print(f"DONE {bench} elapsed={elapsed}s {row['factors']}", flush=True)

    out_summary = matrix_root / f"post_flash_mem_parallel_summary_{stamp}.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_summary}")
    ok = sum(1 for row in summary for f in row.get("factors", []) if f.get("success"))
    total = sum(len(row.get("factors", [])) for row in summary if row.get("factors"))
    print(f"success: {ok}/{total} factor runs")
    return 0 if ok == total or total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
