#!/usr/bin/env python3
"""PC2 flash smoke on tier_A_ready (90-skills packaged library only).

Artifacts::

    artifacts/pc2/flash_tier_a_smoke_<stamp>/<bench>/<model_tag>__flash__tier_a__90skills/

Example::

    python3 scripts/pc2/run_tier_a_flash_smoke_batch.py --pc2 --dry-run
    python3 scripts/pc2/run_tier_a_flash_smoke_batch.py --pc2 --verify-only
    python3 scripts/pc2/run_tier_a_flash_smoke_batch.py --pc2 --stamp 20260616_smoke
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
from tier_a_flash_lib import (
    DEFAULT_SMOKE_BENCHES,
    OUT_ENV,
    SETUP_TAG,
    STAMP_ENV,
    TIER_A_READY_ROOT,
    configure_tier_a_flash_90skills_env,
    env_snapshot,
    list_tier_a_benches,
    resolve_tier_a_benches,
    verify_skills_90,
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


def _cell_dir(out_root: Path, bench: str, model_tag: str) -> Path:
    return out_root / bench / f"{model_tag}__{SETUP_TAG}"


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("steps") or []
    final_step = steps[-1] if steps else {}
    return {
        "phase": result.get("phase"),
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "synth_report": result.get("final_report") or result.get("synth_report"),
        "baseline_report": result.get("baseline_report"),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "final_step": final_step.get("step_name") if isinstance(final_step, dict) else None,
        "llm_usage": result.get("llm_usage") or (result.get("run") or {}).get("llm_usage"),
    }


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


def run_smoke(
    *,
    stamp: str,
    out_root: Path,
    model_id: str,
    benches: list[tuple[str, Path]],
    dry_run: bool,
    force: bool = False,
) -> int:
    check = verify_skills_90()
    if not check.get("ok"):
        raise SystemExit(f"skills preflight failed: {check.get('errors')}")

    if not TIER_A_READY_ROOT.is_dir():
        raise SystemExit(f"missing tier_A_ready root: {TIER_A_READY_ROOT}")

    model_tag = model_cell_tag(model_id)
    snap = env_snapshot()
    plan = {
        "matrix_family": snap["matrix_family"],
        "corpus": snap["corpus"],
        "benchmarks_root": snap["benchmarks_root"],
        "setup": SETUP_TAG,
        "stamp": stamp,
        "out_root": str(out_root),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "skills": check,
        "env": snap,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"setup={SETUP_TAG} model={model_id} tag={model_tag}")
    print(f"skills_json={check['skills_json']} count={check['skill_count']}")
    print(f"corpus=tier_A_ready benches={len(benches)} record_flow={snap['record_flow']}")
    print(f"out_root={out_root}")
    for bench, bench_dir in benches:
        print(f"  - {bench} ({bench_dir}) -> {_cell_dir(out_root, bench, model_tag)}")

    if dry_run:
        print("dry-run: manifest written, no benchmarks executed")
        return 0

    configure_tier_a_flash_90skills_env()
    from c2hls import run_benchmark_multistep

    rows: list[dict[str, Any]] = []
    matrix_path = out_root / "matrix.json"
    if matrix_path.exists():
        rows = json.loads(matrix_path.read_text(encoding="utf-8"))
    if force:
        bench_names = {name for name, _ in benches}
        rows = [row for row in rows if row.get("bench") not in bench_names]

    for bench, bench_dir in benches:
        cell = _cell_dir(out_root, bench, model_tag)
        cell.mkdir(parents=True, exist_ok=True)
        result_json = cell / f"{bench}_multistep_results.json"

        existing = None if force else _load_existing_result(result_json, bench)
        if existing is not None:
            print(f"SKIP {bench} success={existing.get('success')} (existing)", flush=True)
            rows.append(
                {
                    "bench": bench,
                    "model": model_id,
                    "mode": "flash",
                    "skills": "90_packaged",
                    "status": "ok",
                    "wallclock_s": 0.0,
                    "cell_dir": str(cell),
                    "error": "",
                    "summary": _compact_summary(existing),
                }
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
            result_json.write_text(json.dumps(result, indent=2) + "\n")
            print(f"ERROR {bench}: {exc}", flush=True)

        elapsed = round(time.time() - t0, 1)
        result_json.write_text(json.dumps(result, indent=2) + "\n")

        rows.append(
            {
                "bench": bench,
                "model": model_id,
                "mode": "flash",
                "skills": "90_packaged",
                "status": status if result.get("success") else "fail",
                "wallclock_s": elapsed,
                "cell_dir": str(cell),
                "error": error or result.get("error"),
                "summary": _compact_summary(result),
            }
        )
        print(f"DONE {bench} success={result.get('success')} elapsed={elapsed}s", flush=True)

    matrix_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"matrix: {matrix_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PC2 flash smoke on tier_A_ready (90 skills)")
    parser.add_argument("--pc2", action="store_true", required=True)
    parser.add_argument("--benches", type=str, default=",".join(DEFAULT_SMOKE_BENCHES))
    parser.add_argument("--list-benches", action="store_true", help="List tier_A_ready benches and exit")
    parser.add_argument("--verify-only", action="store_true", help="Preflight skills + bench paths only")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run benches even when multistep results already exist",
    )
    args = parser.parse_args()

    os.environ["C2HLS_SITE"] = "pc2"
    configure_site()

    if args.list_benches:
        for name in list_tier_a_benches():
            print(name)
        return 0

    check = verify_skills_90()
    print("=== skills preflight ===")
    print(json.dumps(check, indent=2))
    if not check.get("ok"):
        return 1

    requested = _split_csv(args.benches)
    try:
        benches = resolve_tier_a_benches(requested)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== bench preflight ===")
    for name, path in benches:
        for req in ("plain.cpp", "hls_baseline.cpp", "metadata.json", "testbench.cpp"):
            ok = (path / req).is_file()
            print(f"  {name}: {req} {'OK' if ok else 'MISSING'}")
            if not ok and req != "testbench.cpp":
                print(f"ERROR: missing required file {path / req}", file=sys.stderr)
                return 1

    if args.verify_only:
        print("verify-only ok")
        return 0

    stamp = args.stamp or os.getenv(STAMP_ENV) or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(
        args.out_root
        or os.getenv(OUT_ENV)
        or REPO / "artifacts" / "pc2" / f"flash_tier_a_smoke_{stamp}"
    )
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")

    return run_smoke(
        stamp=stamp,
        out_root=out_root,
        model_id=model_id,
        benches=benches,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
