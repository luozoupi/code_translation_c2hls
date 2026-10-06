#!/usr/bin/env python3
"""PE-array overlap ablation: fuse + ping-pong DATAFLOW on *_dse.cpp.

Does not emit the stream PE-FIFO pack. Does not overwrite *_selected.cpp.
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
from post_flash_overlap import (
    build_overlap_skills_prompt_block,
    configure_post_flash_overlap_env,
    run_overlap_for_cell,
)
from post_flash_stream import (
    discover_stream_cells,
    repair_round_limit,
    resolve_stream_source_kernel,
    stream_max_tokens,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_bench_dir(bench: str) -> Path:
    candidates = [
        BENCHMARKS_DIR / bench,
        REPO / "related_work/benchmarks/autosa_ready" / bench,
        REPO / "benchmarks_autosa_dse" / bench,
        REPO / "benchmarks_autosa" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/chathls_ready" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready" / bench,
        REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready" / bench,
    ]
    for path in candidates:
        if (path / "metadata.json").is_file():
            return path
    raise ValueError(f"unknown benchmark: {bench}")


def _preflight_llm() -> None:
    import urllib.error
    import urllib.request

    base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError("OPENAI_BASE_URL is not set")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    url = f"{base}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"LLM preflight HTTP {resp.status} for {url}")


def _plan_for_matrix(matrix_root: Path, benches: str) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    bench_filter = set(_split_csv(benches)) if benches else None
    cells = discover_stream_cells(matrix_root)
    if bench_filter:
        cells = [c for c in cells if c["bench"] in bench_filter]
    for cell in cells:
        bench = cell["bench"]
        cell_dir = Path(cell["cell_dir"])
        kpath, role, _report = resolve_stream_source_kernel(cell_dir, bench)
        plan.append({
            "bench": bench,
            "cell_dir": str(cell_dir),
            "kernel": str(kpath) if kpath else None,
            "kernel_role": role,
        })
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="PE overlap ablation (fuse + ping-pong DATAFLOW)")
    parser.add_argument("--pc2", action="store_true")
    parser.add_argument("--matrix-root", type=str, default="")
    parser.add_argument("--benches", type=str, default="")
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument("--turns", type=int, default=repair_round_limit())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.pc2:
        configure_site("pc2")
    configure_post_flash_overlap_env()
    os.environ["C2HLS_STREAM_REPAIR_ROUNDS"] = str(args.turns)

    if not args.matrix_root.strip():
        print("--matrix-root is required", file=sys.stderr)
        return 1
    matrix_root = Path(args.matrix_root).expanduser()
    if not matrix_root.is_absolute():
        matrix_root = REPO / matrix_root
    if not matrix_root.is_dir():
        print(f"matrix root missing: {matrix_root}", file=sys.stderr)
        return 1

    plan = _plan_for_matrix(matrix_root, args.benches)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan_path = matrix_root / f"post_flash_overlap_plan_{stamp}.json"
    plan_path.write_text(json.dumps({
        "matrix_root": str(matrix_root),
        "cells": plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"plan: {plan_path} ({len(plan)} cells)")
    if args.dry_run:
        for row in plan:
            print(f"  {row['bench']}: kernel={row['kernel']} role={row.get('kernel_role')}")
        return 0

    _preflight_llm()
    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(
        gpt_model=model,
        turns_limitation=args.turns,
        max_completion_tokens=stream_max_tokens(),
    )
    summary: list[dict[str, Any]] = []
    for row in plan:
        bench = row["bench"]
        if not row.get("kernel"):
            print(f"SKIP {bench}: no dse kernel", flush=True)
            summary.append({"bench": bench, "skipped": True})
            continue
        print(f"START {bench} overlap seed={row.get('kernel_role')}", flush=True)
        t0 = time.time()
        try:
            outcome = run_overlap_for_cell(
                bench=bench,
                bench_dir=_resolve_bench_dir(bench),
                cell_dir=Path(row["cell_dir"]),
                orchestrator=orch,
                skip_existing=not args.force,
            )
        except Exception as exc:
            print(f"ERROR {bench}: {exc}", flush=True)
            summary.append({"bench": bench, "error": str(exc)})
            continue
        payload = outcome.result or {}
        summary.append({
            "bench": bench,
            "elapsed_s": round(time.time() - t0, 1),
            "success": outcome.success,
            "error": outcome.error,
            "latency_cycles": payload.get("latency_cycles"),
            "dsp": payload.get("dsp"),
            "source_kernel_role": payload.get("source_kernel_role"),
        })
        print(
            f"DONE {bench} success={outcome.success} "
            f"lat={payload.get('latency_cycles')} dsp={payload.get('dsp')}",
            flush=True,
        )
    out_summary = matrix_root / f"post_flash_overlap_summary_{stamp}.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_summary}")
    ok = sum(1 for row in summary if row.get("success"))
    attempted = sum(1 for row in summary if not row.get("skipped"))
    print(f"passed: {ok}/{attempted}")
    return 0 if attempted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
