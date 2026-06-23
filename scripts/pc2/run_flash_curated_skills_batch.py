#!/usr/bin/env python3
"""PC2 flash batch — LLM-curated skills matrix (packaged 73-skill library).

Example::

    python3 scripts/pc2/run_flash_curated_skills_batch.py --pc2 \\
        --variant all_avoids_json --focus bottleneck --dry-run
    python3 scripts/pc2/run_flash_curated_skills_batch.py --pc2 \\
        --variant no_avoids_llm --focus combined --stamp 20260622_120000
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

from c2hls_paths import BENCHMARKS_DIR, configure_site
from flash_curated_skills_lib import (
    CURATION_FOCUS_VALUES,
    NEW_SKILLS_JSON,
    VARIANTS,
    VARIANT_ORDER,
    FlashCuratedVariant,
    configure_curated_env,
    variant_env_snapshot,
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


def _cell_dir(out_root: Path, bench: str, model_tag: str, variant: FlashCuratedVariant) -> Path:
    return out_root / bench / f"{model_tag}__{variant.setup_tag}"


def _load_existing_result(result_json: Path, bench: str) -> dict[str, Any] | None:
    if not result_json.is_file():
        return None
    try:
        result = json.loads(result_json.read_text())
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
        "skill_curation": result.get("skill_curation"),
        "llm_usage": result.get("llm_usage") or (result.get("run") or {}).get("llm_usage"),
    }


def _matrix_row(
    *,
    bench: str,
    model_id: str,
    variant: FlashCuratedVariant,
    focus: str,
    result: dict[str, Any],
    status: str,
    elapsed: float,
    cell: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "bench": bench,
        "model": model_id,
        "mode": "flash",
        "variant": variant.key,
        "curation_focus": focus,
        "matrix_family": variant.matrix_family,
        "skills_json": str(NEW_SKILLS_JSON),
        "status": status if result.get("success") else "fail",
        "wallclock_s": elapsed,
        "cell_dir": str(cell),
        "error": error or result.get("error"),
        "summary": _compact_summary(result),
    }


def run_variant(
    variant: FlashCuratedVariant,
    *,
    focus: str,
    stamp: str,
    out_root: Path | None,
    model_id: str,
    benches: list[tuple[str, Path]],
    dry_run: bool,
) -> int:
    out = out_root or Path(
        os.getenv(variant.out_env)
        or REPO / "artifacts" / "pc2" / variant.artifact_dir_name(focus, stamp)
    )
    model_tag = model_cell_tag(model_id)

    plan = {
        "matrix_family": variant.matrix_family,
        "variant": variant.key,
        "label": variant.label,
        "curation_focus": focus,
        "setup": variant.setup_tag,
        "session_id": variant.session_id,
        "stamp": stamp,
        "out_root": str(out),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "skills": variant_env_snapshot(variant, focus=focus),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"variant={variant.key} focus={focus} label={variant.label} benches={len(benches)}")
    print(f"skills_json={NEW_SKILLS_JSON}")
    print(f"out_root={out}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out, bench, model_tag, variant)}")

    if dry_run:
        print("dry-run ok")
        return 0

    configure_curated_env(variant, focus=focus)
    from c2hls import run_benchmark_multistep

    rows: list[dict[str, Any]] = []
    if (out / "matrix.json").exists():
        rows = json.loads((out / "matrix.json").read_text())

    for bench, bench_dir in benches:
        cell = _cell_dir(out, bench, model_tag, variant)
        cell.mkdir(parents=True, exist_ok=True)
        result_json = cell / f"{bench}_multistep_results.json"

        existing = _load_existing_result(result_json, bench)
        if existing is not None:
            print(
                f"SKIP {bench} success={existing.get('success')} (existing {result_json.name})",
                flush=True,
            )
            rows.append(
                _matrix_row(
                    bench=bench,
                    model_id=model_id,
                    variant=variant,
                    focus=focus,
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
            result_json.write_text(json.dumps(result, indent=2) + "\n")
            print(f"ERROR {bench}: {exc}", flush=True)

        elapsed = round(time.time() - t0, 1)
        if not result_json.exists():
            result_json.write_text(json.dumps(result, indent=2) + "\n")

        rows.append(
            _matrix_row(
                bench=bench,
                model_id=model_id,
                variant=variant,
                focus=focus,
                result=result,
                status=status,
                elapsed=elapsed,
                cell=cell,
                error=error,
            )
        )
        print(f"DONE {bench} success={result.get('success')} elapsed={elapsed}s", flush=True)

    (out / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PC2 flash batch — LLM-curated skills matrix",
    )
    parser.add_argument("--pc2", action="store_true", required=True, help="Required (PC2-only)")
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help=f"One of: {', '.join(VARIANT_ORDER)}",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default="bottleneck",
        choices=list(CURATION_FOCUS_VALUES),
        help="Curation strategy for this wave",
    )
    parser.add_argument("--list-variants", action="store_true", help="Print variant keys and exit")
    parser.add_argument("--benches", type=str, default="", help="Comma-separated (default: all hlsfactory_*)")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_variants:
        for key in VARIANT_ORDER:
            v = VARIANTS[key]
            print(
                f"{key:20s} session={v.session_id}  "
                f"artifact={v.artifact_prefix}_<focus>_<stamp>"
            )
        return 0

    if not args.variant or args.variant not in VARIANTS:
        parser.error(f"--variant required; choose from: {', '.join(VARIANT_ORDER)}")

    os.environ["C2HLS_SITE"] = "pc2"
    configure_site()

    variant = VARIANTS[args.variant]
    stamp = args.stamp or os.getenv(variant.stamp_env) or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = args.model or os.getenv("C2HLS_MODEL", "mistralai/Devstral-2-123B-Instruct-2512")
    requested = _split_csv(args.benches) if args.benches else _all_hlsfactory_names()
    benches = _resolve_benches(requested)
    out_root = Path(args.out_root) if args.out_root else None

    return run_variant(
        variant,
        focus=args.focus,
        stamp=stamp,
        out_root=out_root,
        model_id=model_id,
        benches=benches,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
