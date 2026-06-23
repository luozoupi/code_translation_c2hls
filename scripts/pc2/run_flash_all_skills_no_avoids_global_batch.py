#!/usr/bin/env python3
"""PC2 flash + global skill library (all recipes, no avoid rules) for hlsfactory_*.

Artifacts: ``artifacts/pc2/flash_all_skills_no_avoids_global_<stamp>/``

Requires ``--pc2`` (``C2HLS_SKILL_PROMPT_MODE=all_skills_no_avoids_global`` is PC2-only).
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

SETUP_TAG = "flash__all_skills_no_avoids_global"
SKILL_PROMPT_MODE = "all_skills_no_avoids_global"
STAMP_ENV = "C2HLS_FLASH_ALL_SKILLS_NO_AVOIDS_GLOBAL_STAMP"
OUT_ENV = "C2HLS_FLASH_ALL_SKILLS_NO_AVOIDS_GLOBAL_OUT"
ARTIFACT_PREFIX = "flash_all_skills_no_avoids_global"


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


def _configure_env() -> None:
    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)
    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = SKILL_PROMPT_MODE
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
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def _cell_dir(out_root: Path, bench: str, model_tag: str) -> Path:
    return out_root / bench / f"{model_tag}__{SETUP_TAG}"


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("steps") or []
    final_step = steps[-1] if steps else {}
    cosim = final_step.get("cosim") if isinstance(final_step, dict) else {}
    measured = (cosim or {}).get("measured") if isinstance(cosim, dict) else {}
    vgt = final_step.get("vs_ground_truth") if isinstance(final_step, dict) else None
    return {
        "phase": result.get("phase"),
        "success": bool(result.get("success")),
        "error": result.get("error"),
        "synth_report": result.get("final_report") or result.get("synth_report"),
        "baseline_report": result.get("baseline_report"),
        "steps_attempted": len(steps),
        "steps_success": sum(1 for step in steps if step.get("success")),
        "final_step": final_step.get("step_name") if isinstance(final_step, dict) else None,
        "vs_ground_truth": vgt,
        "cosim_cycles": measured.get("latency_cycles_avg") if isinstance(measured, dict) else None,
        "llm_usage": result.get("llm_usage") or (result.get("run") or {}).get("llm_usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PC2 flash + global skills (no avoids) batch")
    parser.add_argument("--pc2", action="store_true", required=True, help="Required (PC2-only mode)")
    parser.add_argument("--benches", type=str, default="", help="Comma-separated (default: all hlsfactory_*)")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["C2HLS_SITE"] = "pc2"
    configure_site()

    stamp = args.stamp or os.getenv(STAMP_ENV) or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(
        args.out_root or os.getenv(OUT_ENV) or REPO / "artifacts" / "pc2" / f"{ARTIFACT_PREFIX}_{stamp}"
    )
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")
    model_tag = model_cell_tag(model_id)
    requested = _split_csv(args.benches) if args.benches else _all_hlsfactory_names()
    benches = _resolve_benches(requested)

    plan = {
        "setup": SETUP_TAG,
        "skill_prompt_mode": SKILL_PROMPT_MODE,
        "stamp": stamp,
        "out_root": str(out_root),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"setup={SETUP_TAG} mode={SKILL_PROMPT_MODE} benches={len(benches)}")
    print(f"out_root={out_root}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out_root, bench, model_tag)}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    _configure_env()
    from c2hls import run_benchmark_multistep

    rows: list[dict[str, Any]] = []
    if (out_root / "matrix.json").exists():
        rows = json.loads((out_root / "matrix.json").read_text())

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
            "skills": SKILL_PROMPT_MODE,
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
