#!/usr/bin/env python3
"""Post-parent memory self-improve runner (50-iter generic mem, no gold).

Example::

    python3 scripts/pc2/run_post_flash_mem_iter.py --pc2 \\
        --matrix-root artifacts/pc2/autosa_mm_variant_sweep_20260919/cells/mem_flash_ns_f0_r01_camp \\
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

from autosa_mm_variant_sweep import find_variant_cell
from c2hls_paths import BENCHMARKS_DIR, configure_site
from post_flash_mem_iter import (
    mem_iter_max_tokens,
    mem_iter_rounds,
    parent_family_from_cell,
    resolve_mem_seed,
    run_mem_iter_for_cell,
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
        raise RuntimeError(
            "OPENAI_BASE_URL is not set. Export OPENAI_BASE_URL or submit via the sweep follow job."
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


def _load_sidecar(matrix_root: Path) -> dict[str, Any]:
    path = matrix_root / "sweep_cell.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-parent memory self-improve (50 iter)")
    parser.add_argument("--pc2", action="store_true")
    parser.add_argument("--matrix-root", type=str, default="")
    parser.add_argument("--cell-dir", type=str, default="", help="Single HLS cell directory")
    parser.add_argument("--benches", type=str, default="autosa_mm")
    parser.add_argument("--model", type=str, default=os.getenv("C2HLS_MODEL", ""))
    parser.add_argument("--turns", type=int, default=mem_iter_rounds())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if result exists")
    args = parser.parse_args()

    if args.pc2:
        configure_site("pc2")
    os.environ["C2HLS_MEM_ITER"] = "1"
    os.environ["C2HLS_MEM_ITER_ROUNDS"] = str(args.turns)

    sidecar = {}
    if args.cell_dir.strip():
        cell_dir = Path(args.cell_dir).expanduser()
        if not cell_dir.is_absolute():
            cell_dir = REPO / cell_dir
        matrix_root = cell_dir
        sidecar = _load_sidecar(cell_dir)
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
        sidecar = _load_sidecar(matrix_root)
        found = find_variant_cell(matrix_root)
        if found is None:
            print(f"no HLS cell under {matrix_root}", file=sys.stderr)
            return 1
        cell_dir = found

    benches = _split_csv(args.benches) or ["autosa_mm"]
    bench = benches[0]
    parent_family = (
        os.getenv("C2HLS_MEM_PARENT_FAMILY", "").strip()
        or parent_family_from_cell(sidecar)
    )
    try:
        seed_cpp, seed_rpt = resolve_mem_seed(cell_dir, parent_family=parent_family)
    except FileNotFoundError as exc:
        print(f"no mem seed: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    plan_path = matrix_root / f"post_flash_mem_iter_plan_{stamp}.json"
    matrix_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "matrix_root": str(matrix_root),
        "cell_dir": str(cell_dir),
        "bench": bench,
        "parent_family": parent_family,
        "seed": str(seed_cpp),
        "seed_report": str(seed_rpt),
        "rounds": args.turns,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"plan: {plan_path} seed={seed_cpp} parent_family={parent_family}")

    if args.dry_run:
        print(f"  {bench}: seed={seed_cpp} report={seed_rpt}")
        return 0

    _preflight_llm()

    from c2hls import C2HLSOrchestrator, DEFAULT_MODEL_ID

    model = args.model.strip() or DEFAULT_MODEL_ID
    orch = C2HLSOrchestrator(
        gpt_model=model,
        turns_limitation=args.turns,
        max_completion_tokens=mem_iter_max_tokens(),
    )

    print(f"START {bench} mem-iter seed={seed_cpp.name} rounds={args.turns}", flush=True)
    t0 = time.time()
    try:
        outcome = run_mem_iter_for_cell(
            bench=bench,
            bench_dir=_resolve_bench_dir(bench),
            cell_dir=cell_dir,
            orchestrator=orch,
            rounds=args.turns,
            parent_family=parent_family,
            skip_existing=not args.force,
        )
    except Exception as exc:
        print(f"ERROR {bench}: {exc}", flush=True)
        (matrix_root / f"post_flash_mem_iter_summary_{stamp}.json").write_text(
            json.dumps({"bench": bench, "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1
    elapsed = round(time.time() - t0, 1)
    summary = {
        "bench": bench,
        "elapsed_s": elapsed,
        **outcome,
    }
    out_summary = matrix_root / f"post_flash_mem_iter_summary_{stamp}.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE {bench} elapsed={elapsed}s success={outcome.get('success')} "
        f"lat_worst={outcome.get('latency_cycles_worst')} dsp={outcome.get('dsp')} "
        f"iter={outcome.get('iter')}",
        flush=True,
    )
    print(f"summary: {out_summary}")
    return 0 if outcome.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
