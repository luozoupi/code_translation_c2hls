#!/usr/bin/env python3
"""PC2 0-shot flash on ``benchmarks_cosim/`` (all cosim-capable hlsfactory benches).

Variants (separate artifact directories):
  phaseb  Phase B functional baseline, then minimal-prompt flash on HLS
  direct  Skip Phase B; minimal-prompt translate+optimize from plain.cpp

Example::

    python3 scripts/pc2/run_zero_shot_cosim_batch.py --pc2 --variant phaseb --dry-run
    python3 scripts/pc2/run_zero_shot_cosim_batch.py --pc2 --variant direct --stamp 20260706_120000
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

from c2hls_paths import configure_site
from run_flash_fixed_cosim_batch import (
    _cell_dir,
    _compact_summary,
    _load_existing_result,
    _matrix_row,
    model_cell_tag,
)
from zero_shot_cosim_lib import (
    VARIANTS,
    VARIANT_ORDER,
    ZeroShotCosimVariant,
    configure_zero_shot_cosim_env,
    list_cosim_benches,
    resolve_cosim_benches,
    variant_env_snapshot,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _matrix_row_zero_shot(
    *,
    bench: str,
    model_id: str,
    variant: ZeroShotCosimVariant,
    result: dict[str, Any],
    status: str,
    elapsed: float,
    cell: Path,
    error: str,
) -> dict[str, Any]:
    snap = variant_env_snapshot(variant)
    return {
        "bench": bench,
        "model": model_id,
        "mode": "flash",
        "variant": variant.key,
        "matrix_family": snap["matrix_family"],
        "corpus": snap["corpus"],
        "flash_opt_prompt_mode": snap["flash_opt_prompt_mode"],
        "skip_phase_b": snap["skip_phase_b"],
        "record_flow": True,
        "status": status if result.get("success") else "fail",
        "wallclock_s": elapsed,
        "cell_dir": str(cell),
        "error": error or result.get("error"),
        "summary": _compact_summary(result),
    }


def run_variant(
    variant: ZeroShotCosimVariant,
    *,
    stamp: str,
    out_root: Path | None,
    model_id: str,
    benches: list[tuple[str, Path]],
    dry_run: bool,
) -> int:
    out = out_root or Path(
        os.getenv(variant.out_env) or REPO / "artifacts" / "pc2" / f"{variant.artifact_prefix}_{stamp}"
    )
    model_tag = model_cell_tag(model_id)
    snap = variant_env_snapshot(variant)

    plan = {
        "matrix_family": snap["matrix_family"],
        "corpus": snap["corpus"],
        "benchmarks_root": snap["benchmarks_root"],
        "record_flow": True,
        "variant": variant.key,
        "label": variant.label,
        "setup": variant.setup_tag,
        "session_id": variant.session_id,
        "stamp": stamp,
        "out_root": str(out),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "flash_opt_prompt_mode": snap["flash_opt_prompt_mode"],
        "skip_phase_b": snap["skip_phase_b"],
        "skills": snap,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(
        f"variant={variant.key} label={variant.label} benches={len(benches)} "
        f"corpus=benchmarks_cosim skip_phase_b={variant.skip_phase_b}"
    )
    print(f"flash_opt_prompt_mode=zero_shot record_flow=1")
    print(f"out_root={out}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out, bench, model_tag, variant)}")

    if dry_run:
        print("dry-run ok")
        return 0

    configure_zero_shot_cosim_env(variant)
    from c2hls import run_benchmark_multistep

    rows: list[dict[str, Any]] = []
    if (out / "matrix.json").exists():
        rows = json.loads((out / "matrix.json").read_text(encoding="utf-8"))

    for bench, bench_dir in benches:
        cell = _cell_dir(out, bench, model_tag, variant)
        cell.mkdir(parents=True, exist_ok=True)
        result_json = cell / f"{bench}_multistep_results.json"

        existing = _load_existing_result(result_json, bench)
        if existing is not None:
            print(f"SKIP {bench} success={existing.get('success')} (existing)", flush=True)
            rows.append(
                _matrix_row_zero_shot(
                    bench=bench,
                    model_id=model_id,
                    variant=variant,
                    result=existing,
                    status="ok",
                    elapsed=0.0,
                    cell=cell,
                    error="",
                )
            )
            continue

        print(f"START {bench}", flush=True)
        t0 = time.time()
        status = "ok"
        error = ""
        try:
            result = run_benchmark_multistep(
                str(bench_dir),
                output_dir=str(cell),
                gpt_model=model_id,
                turns_limitation=int(os.getenv("C2HLS_TURNS", "4")),
                steps=None,
            )
        except Exception as exc:
            status = "error"
            error = str(exc)
            result = {
                "benchmark": bench,
                "success": False,
                "phase": "exception",
                "error": error,
                "steps": [],
            }
            result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"ERROR {bench}: {exc}", flush=True)

        elapsed = round(time.time() - t0, 1)
        if not result_json.exists():
            result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        rows.append(
            _matrix_row_zero_shot(
                bench=bench,
                model_id=model_id,
                variant=variant,
                result=result,
                status=status,
                elapsed=elapsed,
                cell=cell,
                error=error,
            )
        )
        print(f"DONE {bench} success={result.get('success')} elapsed={elapsed}s", flush=True)

    (out / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PC2 0-shot flash on benchmarks_cosim")
    parser.add_argument("--pc2", action="store_true", required=True)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--benches", type=str, default="")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_variants:
        for key in VARIANT_ORDER:
            v = VARIANTS[key]
            print(
                f"{key:8s} {v.label:40s} skip_phase_b={v.skip_phase_b} "
                f"artifacts=artifacts/pc2/{v.artifact_prefix}_<stamp>/"
            )
        return 0

    if not args.variant or args.variant not in VARIANTS:
        parser.error(f"--variant required; choose from: {', '.join(VARIANT_ORDER)}")

    configure_site("pc2")
    variant = VARIANTS[args.variant]
    stamp = args.stamp or os.getenv(variant.stamp_env) or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-Small-2505")

    if args.benches:
        benches = resolve_cosim_benches(_split_csv(args.benches))
    else:
        benches = resolve_cosim_benches(list_cosim_benches())

    out_root = Path(args.out_root) if args.out_root else None
    return run_variant(
        variant,
        stamp=stamp,
        out_root=out_root,
        model_id=model_id,
        benches=benches,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
