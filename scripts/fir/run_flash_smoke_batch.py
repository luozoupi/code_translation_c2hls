#!/usr/bin/env python3
"""Fir flash smoke on in-repo hlsfactory benchmarks (open-weight vLLM, no cosim).

Artifacts::

    artifacts/fir/flash_smoke_<stamp>/<bench>/<model_tag>__flash__all_skills_avoids_global/

Examples::

    python3 scripts/fir/run_flash_smoke_batch.py --fir --dry-run
    python3 scripts/fir/run_flash_smoke_batch.py --fir --benches hlsfactory_gemm
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
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "fir"))

from c2hls_paths import BENCHMARKS_DIR, configure_site, site_artifacts_dir
from flash_lib import (
    ARTIFACT_PREFIX,
    ARTIFACT_PREFIX_COSIM,
    DEFAULT_SMOKE_BENCHES,
    OUT_ENV,
    SETUP_TAG,
    STAMP_ENV,
    configure_fir_flash_env,
    env_snapshot,
)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def model_cell_tag(model_id: str) -> str:
    low = (model_id or "").lower()
    if "devstral" in low:
        return "devstral2"
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return slug[:48] or "model"


def _resolve_benches(requested: list[str]) -> list[tuple[str, Path]]:
    available: dict[str, Path] = {}
    for meta_path in sorted(BENCHMARKS_DIR.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        name = meta.get("benchmark") or meta_path.parent.name
        available[name] = meta_path.parent
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def _cell_dir(out_root: Path, bench: str, model_tag: str) -> Path:
    return out_root / bench / f"{model_tag}__{SETUP_TAG}"


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("steps") or []
    final_step = steps[-1] if steps else {}
    return {
        "phase": result.get("phase"),
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "final_step": final_step.get("step_name") if isinstance(final_step, dict) else None,
        "llm_usage": result.get("llm_usage") or (result.get("run") or {}).get("llm_usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fir flash smoke (open-weight vLLM, no cosim)")
    parser.add_argument("--fir", action="store_true", required=True)
    parser.add_argument("--benches", type=str, default=",".join(DEFAULT_SMOKE_BENCHES))
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--cosim", action="store_true", help="Enable RTL cosim + LLM repair (overrides fir.env cosim=0)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["C2HLS_SITE"] = "fir"
    configure_site("fir")

    stamp = args.stamp or os.getenv(STAMP_ENV) or datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_prefix = ARTIFACT_PREFIX_COSIM if args.cosim else ARTIFACT_PREFIX
    out_root = Path(
        args.out_root or os.getenv(OUT_ENV) or site_artifacts_dir("fir") / f"{artifact_prefix}_{stamp}"
    )
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")
    model_tag = model_cell_tag(model_id)
    benches = _resolve_benches(_split_csv(args.benches))

    configure_fir_flash_env(cosim=args.cosim)

    plan = {
        "site": "fir",
        "setup": SETUP_TAG,
        "cosim": args.cosim,
        "repair": True,
        "stamp": stamp,
        "out_root": str(out_root),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "env": env_snapshot(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"site=fir setup={SETUP_TAG} cosim={args.cosim} benches={len(benches)}")
    print(f"out_root={out_root}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out_root, bench, model_tag)}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    from c2hls import run_benchmark_multistep

    rows: list[dict[str, Any]] = []
    for bench, bench_dir in benches:
        cell = _cell_dir(out_root, bench, model_tag)
        cell.mkdir(parents=True, exist_ok=True)
        result_json = cell / f"{bench}_multistep_results.json"
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
            result = {"benchmark": bench, "success": False, "phase": "exception", "error": error, "steps": []}
            result_json.write_text(json.dumps(result, indent=2) + "\n")
            print(f"ERROR {bench}: {exc}", flush=True)

        elapsed = round(time.time() - t0, 1)
        if not result_json.exists():
            result_json.write_text(json.dumps(result, indent=2) + "\n")

        rows.append({
            "bench": bench,
            "model": model_id,
            "mode": "flash",
            "status": status if result.get("success") else "fail",
            "wallclock_s": elapsed,
            "cell_dir": str(cell),
            "error": error or result.get("error"),
            "summary": _compact_summary(result),
        })
        print(f"DONE {bench} success={result.get('success')} elapsed={elapsed}s", flush=True)

    (out_root / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
