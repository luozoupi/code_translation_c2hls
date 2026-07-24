#!/usr/bin/env python3
"""Post-pass constrained latency optimization batch runner.

Runs after flash-final or DATAFLOW kernels pass csim+csynth. Uses analysis-guided
plan→modify rounds with trajectory tracking and a hard device budget.

Example::

    python3 scripts/pc2/run_post_flash_latency_opt.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --source flash_final --dry-run

    python3 scripts/pc2/run_post_flash_latency_opt.py --pc2 \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548 \\
        --source dataflow --benches hlsfactory_gemm
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
from post_flash_latency_opt import (
    SOURCE_ROLES,
    SourceRole,
    configure_post_flash_env,
    discover_matrix_cells,
    latency_round_limit,
    prompt_text_for_docs,
    repair_round_limit,
    resolve_latency_source_kernel,
    run_latency_opt_for_cell,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_bench_dir(bench: str) -> Path:
    candidates = [
        BENCHMARKS_DIR / bench,
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
            "OPENAI_BASE_URL is not set. Submit with ./scripts/pc2/start_post_flash_latency_opt.sh "
            "--submit or export OPENAI_BASE_URL."
        )
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    url = f"{base}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM preflight {url} returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM preflight failed: HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM preflight cannot reach {url}: {exc}") from exc
    print(f"LLM preflight ok: {base}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-pass constrained latency optimization")
    parser.add_argument("--pc2", action="store_true", help="PC2 site paths + vLLM")
    parser.add_argument("--matrix-root", type=str, default="")
    parser.add_argument("--benches", type=str, default="", help="Comma-separated filter")
    parser.add_argument(
        "--source",
        type=str,
        choices=list(SOURCE_ROLES),
        default="flash_final",
        help="Input kernel: flash_final (selected/final) or dataflow (passing *_dataflow.cpp)",
    )
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Latency improvement rounds N (default: env C2HLS_LATENCY_OPT_ROUNDS or 3)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=repair_round_limit(),
        help="Repair rounds R per failed validation (default: env or 3)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if result exists")
    parser.add_argument("--show-prompts", action="store_true", help="Print prompts and exit")
    args = parser.parse_args()

    if args.show_prompts:
        prompts = prompt_text_for_docs()
        print("=== PLAN SYSTEM ===\n")
        print(prompts["plan_system"])
        print("\n=== PLAN USER (template) ===\n")
        print(prompts["plan_user"])
        print("\n=== MODIFY SYSTEM ===\n")
        print(prompts["modify_system"])
        print("\n=== MODIFY USER (template) ===\n")
        print(prompts["modify_user"])
        print("\n=== REPAIR USER (template) ===\n")
        print(prompts["repair_user"])
        return 0

    if not args.matrix_root.strip():
        print("--matrix-root is required (unless --show-prompts)", file=sys.stderr)
        return 1

    if args.pc2:
        configure_site("pc2")
    configure_post_flash_env()
    os.environ["C2HLS_POST_FLASH_LATENCY_OPT"] = "1"
    os.environ["C2HLS_LATENCY_OPT_REPAIR_ROUNDS"] = str(args.turns)
    if args.rounds > 0:
        os.environ["C2HLS_LATENCY_OPT_ROUNDS"] = str(args.rounds)

    matrix_root = Path(args.matrix_root).expanduser()
    if not matrix_root.is_absolute():
        matrix_root = REPO / matrix_root
    if not matrix_root.is_dir():
        print(f"matrix root missing: {matrix_root}", file=sys.stderr)
        return 1

    source_role: SourceRole = args.source  # type: ignore[assignment]
    bench_filter = set(_split_csv(args.benches)) if args.benches else None
    cells = discover_matrix_cells(matrix_root)
    if bench_filter:
        cells = [c for c in cells if c["bench"] in bench_filter]

    plan: list[dict[str, Any]] = []
    for cell in cells:
        bench = cell["bench"]
        cell_dir = Path(cell["cell_dir"])
        kpath, role, _ = resolve_latency_source_kernel(cell_dir, bench, source_role)
        plan.append({
            "bench": bench,
            "cell_dir": str(cell_dir),
            "source": source_role,
            "kernel": str(kpath) if kpath else None,
            "kernel_role": role,
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan_path = matrix_root / f"post_flash_latency_opt_plan_{source_role}_{stamp}.json"
    plan_path.write_text(json.dumps({
        "matrix_root": str(matrix_root),
        "source_role": source_role,
        "latency_rounds": latency_round_limit(),
        "repair_rounds": repair_round_limit(),
        "cells": plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"plan: {plan_path} ({len(plan)} cells)")

    if args.dry_run:
        for row in plan:
            print(f"  {row['bench']}: kernel={row['kernel']}")
        return 0

    _preflight_llm()

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(gpt_model=model, turns_limitation=args.turns)

    summary: list[dict[str, Any]] = []
    for row in plan:
        bench = row["bench"]
        if not row.get("kernel"):
            print(f"SKIP {bench}: no source kernel for {source_role}", flush=True)
            summary.append({"bench": bench, "skipped": True, "reason": f"no {source_role} kernel"})
            continue
        print(f"START {bench} ({source_role})", flush=True)
        t0 = time.time()
        try:
            outcome = run_latency_opt_for_cell(
                bench=bench,
                bench_dir=_resolve_bench_dir(bench),
                cell_dir=Path(row["cell_dir"]),
                orchestrator=orch,
                source_role=source_role,
                skip_existing=not args.force,
            )
        except Exception as exc:
            print(f"ERROR {bench}: {exc}", flush=True)
            summary.append({"bench": bench, "error": str(exc)})
            continue
        elapsed = round(time.time() - t0, 1)
        summary.append({
            "bench": bench,
            "source": source_role,
            "elapsed_s": elapsed,
            "success": outcome.success,
            "error": outcome.error,
            "latency_cycles": (outcome.result or {}).get("latency_cycles"),
        })
        print(f"DONE {bench} elapsed={elapsed}s success={outcome.success}", flush=True)

    out_summary = matrix_root / f"post_flash_latency_opt_summary_{source_role}_{stamp}.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_summary}")
    ok = sum(1 for row in summary if row.get("success"))
    attempted = sum(1 for row in summary if not row.get("skipped"))
    print(f"passed: {ok}/{attempted}")
    return 0 if ok == attempted and attempted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
