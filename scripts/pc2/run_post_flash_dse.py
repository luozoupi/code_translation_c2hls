#!/usr/bin/env python3
"""Post-flash DSE batch runner (multi-PE GEMM nest rewrite).

Example::

    python3 scripts/pc2/run_post_flash_dse.py --pc2 \\
        --matrix-root artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f \\
        --benches autosa_mm --dry-run
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
from post_flash_dse import (
    configure_post_flash_dse_env,
    discover_dse_cells,
    dse_max_tokens,
    dse_source_role,
    prompt_text_for_docs,
    repair_round_limit,
    resolve_selected_kernel,
    run_dse_for_cell,
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
        raise RuntimeError(
            "OPENAI_BASE_URL is not set. Submit with ./scripts/pc2/start_post_flash_dse.sh "
            "--submit or export OPENAI_BASE_URL."
        )
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    url = f"{base}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM preflight HTTP {resp.status} for {url}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM preflight failed for {url}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-flash DSE (multi-PE GEMM rewrite)")
    parser.add_argument("--pc2", action="store_true")
    parser.add_argument("--matrix-root", type=str, default="")
    parser.add_argument("--cell-dir", type=str, default="", help="Single cell directory")
    parser.add_argument("--benches", type=str, default="")
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument("--turns", type=int, default=repair_round_limit())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if result exists")
    parser.add_argument("--show-prompts", action="store_true")
    args = parser.parse_args()

    if args.show_prompts:
        prompts = prompt_text_for_docs()
        print("=== SYSTEM ===\n")
        print(prompts["system"])
        print("\n=== INITIAL USER (template) ===\n")
        print(prompts["initial_user"])
        print("\n=== REPAIR USER (template) ===\n")
        print(prompts["repair_user"])
        print(f"\nskills: {prompts['skills_path']}")
        print("skill_ids: " + ", ".join(prompts["skill_ids"]))
        return 0

    if args.pc2:
        configure_site("pc2")
    configure_post_flash_dse_env()
    os.environ["C2HLS_POST_FLASH_DSE"] = "1"
    os.environ["C2HLS_DSE_REPAIR_ROUNDS"] = str(args.turns)

    plan: list[dict[str, Any]] = []
    if args.cell_dir.strip():
        cell_dir = Path(args.cell_dir).expanduser()
        if not cell_dir.is_absolute():
            cell_dir = REPO / cell_dir
        selected = list(cell_dir.glob("*_selected.cpp"))
        if not selected:
            print(f"no *_selected.cpp in {cell_dir}", file=sys.stderr)
            return 1
        bench = selected[0].name[: -len("_selected.cpp")]
        if args.benches and bench not in set(_split_csv(args.benches)):
            print(f"cell bench {bench} not in --benches", file=sys.stderr)
            return 1
        kpath, role = resolve_selected_kernel(cell_dir, bench)
        plan.append({
            "bench": bench,
            "cell_dir": str(cell_dir),
            "kernel": str(kpath) if kpath else None,
            "kernel_role": role,
        })
        matrix_root = cell_dir
    else:
        if not args.matrix_root.strip():
            print("--matrix-root or --cell-dir is required", file=sys.stderr)
            return 1
        matrix_root = Path(args.matrix_root).expanduser()
        if not matrix_root.is_absolute():
            matrix_root = REPO / matrix_root
        if not matrix_root.is_dir():
            print(f"matrix root missing: {matrix_root}", file=sys.stderr)
            return 1
        bench_filter = set(_split_csv(args.benches)) if args.benches else None
        cells = discover_dse_cells(matrix_root)
        if bench_filter:
            cells = [c for c in cells if c["bench"] in bench_filter]
        for cell in cells:
            bench = cell["bench"]
            cell_dir = Path(cell["cell_dir"])
            kpath, role = resolve_selected_kernel(cell_dir, bench)
            plan.append({
                "bench": bench,
                "cell_dir": str(cell_dir),
                "kernel": str(kpath) if kpath else None,
                "kernel_role": role,
            })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan_path = Path(matrix_root) / f"post_flash_dse_plan_{stamp}.json"
    if not Path(matrix_root).is_dir():
        Path(matrix_root).mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({
        "matrix_root": str(matrix_root),
        "source_role": dse_source_role(),
        "repair_rounds": repair_round_limit(),
        "cells": plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"plan: {plan_path} ({len(plan)} cells) source_role={dse_source_role()}")

    if args.dry_run:
        for row in plan:
            print(f"  {row['bench']}: kernel={row['kernel']}")
        return 0

    _preflight_llm()

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(
        gpt_model=model,
        turns_limitation=args.turns,
        max_completion_tokens=dse_max_tokens(),
    )

    summary: list[dict[str, Any]] = []
    for row in plan:
        bench = row["bench"]
        if not row.get("kernel"):
            print(f"SKIP {bench}: no selected kernel", flush=True)
            summary.append({"bench": bench, "skipped": True, "reason": "no selected kernel"})
            continue
        print(f"START {bench} dse", flush=True)
        t0 = time.time()
        try:
            outcome = run_dse_for_cell(
                bench=bench,
                bench_dir=_resolve_bench_dir(bench),
                cell_dir=Path(row["cell_dir"]),
                orchestrator=orch,
                source_role=dse_source_role(),
                skip_existing=not args.force,
            )
        except Exception as exc:
            print(f"ERROR {bench}: {exc}", flush=True)
            summary.append({"bench": bench, "error": str(exc)})
            continue
        elapsed = round(time.time() - t0, 1)
        payload = outcome.result or {}
        summary.append({
            "bench": bench,
            "elapsed_s": elapsed,
            "success": outcome.success,
            "error": outcome.error,
            "latency_cycles": payload.get("latency_cycles"),
            "dsp": payload.get("dsp"),
            "promoted": payload.get("promoted"),
        })
        print(
            f"DONE {bench} elapsed={elapsed}s success={outcome.success} "
            f"lat={payload.get('latency_cycles')} dsp={payload.get('dsp')} "
            f"promoted={payload.get('promoted')}",
            flush=True,
        )

    out_summary = Path(matrix_root) / f"post_flash_dse_summary_{stamp}.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_summary}")
    ok = sum(1 for row in summary if row.get("success"))
    attempted = sum(1 for row in summary if not row.get("skipped"))
    print(f"passed: {ok}/{attempted}")
    return 0 if ok == attempted and attempted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
