#!/usr/bin/env python3
"""PC2 serial multistep on ``benchmarks_cosim/`` with ``C2HLS_RECORD_FLOW=1`` (aav_n v1).

Example::

    python3 scripts/pc2/run_multistep_fixed_cosim_batch.py --pc2 --dry-run
    python3 scripts/pc2/run_multistep_fixed_cosim_batch.py --pc2 --stamp 20260626_fixed_cosim_multistep
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from c2hls_paths import configure_site
from multistep_fixed_cosim_lib import (
    PILOT_BENCHES,
    STAMP_ENV,
    VARIANTS,
    VARIANT_ORDER,
    MultistepFixedCosimVariant,
    configure_fixed_cosim_multistep_env,
    list_cosim_benches,
    resolve_cosim_benches,
    variant_env_snapshot,
    verify_variant_skills,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def model_cell_tag(model_id: str) -> str:
    low = (model_id or "").lower()
    if "devstral" in low:
        return "devstral2"
    if "sonnet" in low:
        return "sonnet"
    if "haiku" in low:
        return "haiku"
    if "nemotron" in low:
        return "nemotron"
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return slug[:48] or "model"


def _cell_dir(out_root: Path, bench: str, model_tag: str, variant: MultistepFixedCosimVariant) -> Path:
    return out_root / bench / f"{model_tag}__{variant.setup_tag}"


def _load_existing_result(result_json: Path, bench: str) -> dict[str, Any] | None:
    if not result_json.is_file():
        return None
    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict) or "success" not in result:
        return None
    stored = result.get("benchmark") or result.get("name")
    if stored and stored != bench:
        return None
    return result


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("steps") or []
    latencies = {
        step.get("step_name"): (step.get("report") or {}).get("latency_cycles")
        for step in steps
        if isinstance(step, dict)
    }
    return {
        "phase": result.get("phase"),
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "synth_report": result.get("final_report") or result.get("synth_report"),
        "baseline_report": result.get("baseline_report"),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "step_latencies": latencies,
        "llm_usage": result.get("llm_usage") or (result.get("run") or {}).get("llm_usage"),
    }


def _matrix_row(
    *,
    bench: str,
    model_id: str,
    variant: MultistepFixedCosimVariant,
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
        "mode": "multistep",
        "variant": variant.key,
        "matrix_family": snap["matrix_family"],
        "corpus": "benchmarks_cosim",
        "skills_json": snap.get("skills_json"),
        "record_flow": True,
        "status": status if result.get("success") else "fail",
        "wallclock_s": elapsed,
        "cell_dir": str(cell),
        "error": error or result.get("error"),
        "summary": _compact_summary(result),
        "origin_meta": snap.get("origin_meta"),
    }


def run_variant(
    variant: MultistepFixedCosimVariant,
    *,
    stamp: str,
    out_root: Path | None,
    model_id: str,
    benches: list[tuple[str, Path]],
    dry_run: bool,
    verify_only: bool,
) -> int:
    check = verify_variant_skills(variant)
    if not check.get("ok"):
        raise SystemExit(f"variant preflight failed: {check['errors']}")

    out = out_root or Path(
        os.getenv(variant.out_env) or REPO / "artifacts" / "pc2" / f"{variant.artifact_prefix}_{stamp}"
    )
    model_tag = model_cell_tag(model_id)
    snap = variant_env_snapshot(variant)

    plan = {
        "matrix_family": snap["matrix_family"],
        "corpus": snap["corpus"],
        "runner": "serial",
        "record_flow": True,
        "strategy": "static",
        "variant": variant.key,
        "stamp": stamp,
        "out_root": str(out),
        "model": model_id,
        "benches": [name for name, _ in benches],
        "skills": snap,
        "skills_preflight": check,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"variant={variant.key} multistep serial benches={len(benches)} corpus=benchmarks_cosim")
    print(f"out_root={out}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out, bench, model_tag, variant)}")

    if dry_run or verify_only:
        print("dry-run ok" if dry_run else "verify ok")
        return 0

    configure_fixed_cosim_multistep_env(variant)
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
                _matrix_row(
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
            _matrix_row(
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
    parser = argparse.ArgumentParser(description="PC2 serial multistep fixed cosim (aav_n)")
    parser.add_argument("--pc2", action="store_true", required=True)
    parser.add_argument("--variant", type=str, default="aav_n")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--verify-all", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Run 5-bench pilot set only")
    parser.add_argument("--benches", type=str, default="")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_variants:
        for key in VARIANT_ORDER:
            print(f"{key:10s} {VARIANTS[key].label}")
        return 0

    if args.verify_all:
        ok = True
        for key in VARIANT_ORDER:
            check = verify_variant_skills(VARIANTS[key])
            print(json.dumps(check, indent=2))
            ok = ok and check.get("ok", False)
        return 0 if ok else 1

    if args.variant not in VARIANTS:
        parser.error(f"--variant required; choose from: {', '.join(VARIANT_ORDER)}")

    os.environ["C2HLS_SITE"] = "pc2"
    configure_site()

    variant = VARIANTS[args.variant]
    stamp = args.stamp or os.getenv(STAMP_ENV) or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")
    if args.pilot:
        requested = list(PILOT_BENCHES)
    elif args.benches:
        requested = _split_csv(args.benches)
    else:
        requested = list_cosim_benches()
    benches = resolve_cosim_benches(requested)
    out_root = Path(args.out_root) if args.out_root else None

    return run_variant(
        variant,
        stamp=stamp,
        out_root=out_root,
        model_id=model_id,
        benches=benches,
        dry_run=args.dry_run,
        verify_only=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
