#!/usr/bin/env python3
"""Run flash + skills-off cells for a fixed hlsfactory bench set on PC2.

Artifacts land under::

    artifacts/pc2/flash_noskills_<stamp>/<bench>/<model_tag>__flash__noskills/

This mirrors the phase-8 matrix layout (``sonnet__flash__noskills``) but uses a
separate tree so PC2 / Devstral runs do not touch ``results_matrix_*``.

Environment (set in the shell, local.env, or sbatch)::

    C2HLS_STRATEGY=flash
    C2HLS_FORCE_SKILL_PROMPTS=0
    OPENAI_BASE_URL=http://<gpu-node>:8000/v1
    OPENAI_API_KEY=EMPTY

Example::

    python3 scripts/pc2/run_flash_noskills_batch.py --pc2 --dry-run
    python3 scripts/pc2/run_flash_noskills_batch.py --pc2
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

from c2hls_paths import BENCHMARKS_DIR, apply_runtime_defaults, configure_site
from c2hls_temp import configure_temp_env

DEFAULT_BENCHES = [
    "hlsfactory_trmm",
    "hlsfactory_trisolv",
    "hlsfactory_symm",
    "hlsfactory_3mm",
    "hlsfactory_2mm",
    "hlsfactory_gemm",
    "hlsfactory_jacobi-2d",
]

SETUP_TAG = "flash__noskills"


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


def _configure_flash_noskills_env() -> None:
    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)
    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_SKILL_MODE"] = "skill_off"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "180")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def _all_hlsfactory_names() -> list[str]:
    names: list[str] = []
    for meta_path in sorted(BENCHMARKS_DIR.glob("hlsfactory_*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        names.append(meta.get("benchmark") or meta_path.parent.name)
    return names


def _resolve_benches(requested: list[str]) -> list[tuple[str, Path]]:
    available: dict[str, Path] = {}
    for meta_path in sorted(BENCHMARKS_DIR.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        name = meta.get("benchmark") or meta_path.parent.name
        available[name] = meta_path.parent

    names = requested or DEFAULT_BENCHES
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"unknown benchmark(s): {missing}")
    return [(name, available[name]) for name in names]


def _cell_dir(out_root: Path, bench: str, model_tag: str) -> Path:
    return out_root / bench / f"{model_tag}__{SETUP_TAG}"


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("steps") or []
    final_step = steps[-1] if steps else {}
    cosim = final_step.get("cosim") if isinstance(final_step, dict) else {}
    measured = (cosim or {}).get("measured") if isinstance(cosim, dict) else {}
    return {
        "phase": result.get("phase"),
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "synth_report": result.get("final_report") or result.get("synth_report"),
        "baseline_report": result.get("baseline_report"),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "final_step": final_step.get("step_name") if isinstance(final_step, dict) else None,
        "cosim_cycles": measured.get("latency_cycles_avg") if isinstance(measured, dict) else None,
        "llm_usage": result.get("llm_usage") or (result.get("run") or {}).get("llm_usage"),
    }


def _write_manifest(out_root: Path, payload: dict[str, Any]) -> None:
    path = out_root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_matrix_row(out_root: Path, row: dict[str, Any]) -> None:
    matrix_path = out_root / "matrix.json"
    rows: list[dict[str, Any]]
    if matrix_path.exists():
        rows = json.loads(matrix_path.read_text())
    else:
        rows = []
    rows.append(row)
    matrix_path.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="PC2 flash + no-skills hlsfactory batch")
    parser.add_argument("--pc2", action="store_true", help="Use PC2 site (local.env)")
    parser.add_argument(
        "--benches",
        type=str,
        default=",".join(DEFAULT_BENCHES),
        help="Comma-separated benchmark names",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Override output root (default: artifacts/pc2/flash_noskills_<stamp>)",
    )
    parser.add_argument("--stamp", type=str, default="", help="Stamp for default out-root")
    parser.add_argument("--model", type=str, default="", help="Override C2HLS_MODEL")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument(
        "--all-hlsfactory",
        action="store_true",
        help="Run every benchmarks/hlsfactory_* kernel (default: pilot set of 7)",
    )
    args = parser.parse_args()

    if args.pc2:
        os.environ["C2HLS_SITE"] = "pc2"
    configure_site()

    stamp = args.stamp or os.getenv("C2HLS_FLASH_NOSKILLS_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(
        args.out_root
        or os.getenv("C2HLS_FLASH_NOSKILLS_OUT")
        or REPO / "artifacts" / "pc2" / f"flash_noskills_{stamp}"
    )
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")
    model_tag = model_cell_tag(model_id)
    benches = _resolve_benches(
        _all_hlsfactory_names() if args.all_hlsfactory else _split_csv(args.benches)
    )

    plan = {
        "setup": SETUP_TAG,
        "stamp": stamp,
        "out_root": str(out_root),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "cell_pattern": f"<bench>/{model_tag}__{SETUP_TAG}/",
        "env": {
            "C2HLS_STRATEGY": "flash",
            "C2HLS_FORCE_SKILL_PROMPTS": "0",
            "C2HLS_SKILL_MODE": "skill_off",
            "C2HLS_RUN_COSIM": os.getenv("C2HLS_RUN_COSIM", "0"),
            "C2HLS_COSIM_REQUIRED": os.getenv("C2HLS_COSIM_REQUIRED", "0"),
            "C2HLS_REFERENCE_COSIM": os.getenv("C2HLS_REFERENCE_COSIM", "0"),
            "C2HLS_HW_EMU_FINAL": "0",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(out_root, plan)

    print(f"setup={SETUP_TAG} model={model_id} tag={model_tag}")
    print(f"out_root={out_root}")
    for bench, bench_dir in benches:
        cell = _cell_dir(out_root, bench, model_tag)
        print(f"  - {bench} -> {cell}")

    if args.dry_run:
        print("dry-run: manifest written, no benchmarks executed")
        return 0

    _configure_flash_noskills_env()
    from c2hls import run_benchmark_multistep

    for bench, bench_dir in benches:
        cell = _cell_dir(out_root, bench, model_tag)
        cell.mkdir(parents=True, exist_ok=True)
        result_json = cell / f"{bench}_multistep_results.json"
        print(f"START {bench} -> {cell}", flush=True)
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
        if not result_json.exists():
            result_json.write_text(json.dumps(result, indent=2) + "\n")

        _append_matrix_row(
            out_root,
            {
                "bench": bench,
                "model": model_id,
                "mode": "flash",
                "skills": "off",
                "status": status if result.get("success") else "fail",
                "wallclock_s": elapsed,
                "cell_dir": str(cell),
                "error": error or result.get("error"),
                "summary": _compact_summary(result),
            },
        )
        print(
            f"DONE {bench} success={result.get('success')} elapsed={elapsed}s",
            flush=True,
        )

    print(f"matrix: {out_root / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
