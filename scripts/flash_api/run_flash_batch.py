#!/usr/bin/env python3
"""Run one flash API profile (commercial LLM) — mirrors a PC2 flash variant.

Artifacts::

    artifacts/flash_api/<artifact_prefix>_<stamp>/<bench>/<model_tag>__<setup_tag>/

Examples::

    python3 scripts/flash_api/run_flash_batch.py --profile nav_o --dry-run
    python3 scripts/flash_api/run_flash_batch.py --profile aav_n --model claude-sonnet-4-6
    python3 scripts/flash_api/run_flash_batch.py --list-profiles
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
sys.path.insert(0, str(REPO / "scripts" / "flash_api"))

from c2hls_paths import BENCHMARKS_DIR
from flash_api_lib import (
    DETERMINISTIC_ORDER,
    PROFILES,
    FlashApiProfile,
    apply_profile_env,
    artifact_root,
    model_cell_tag,
    preflight_api_run,
    profile_skills_snapshot,
    resolve_model_id,
)
from flash_shared.team_env import active_team_paths_summary, bootstrap_team_flash_env, flash_cosim_manifest


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _cell_dir(out_root: Path, bench: str, model_tag: str, profile: FlashApiProfile) -> Path:
    return out_root / bench / f"{model_tag}__{profile.setup_tag}"


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


def run_profile(
    profile: FlashApiProfile,
    *,
    stamp: str,
    out_root: Path | None,
    model_id: str,
    benches: list[tuple[str, Path]],
    dry_run: bool,
) -> int:
    out = out_root or artifact_root(profile, stamp)
    model_tag = model_cell_tag(model_id)

    plan = {
        "inference": "commercial_api",
        "profile": profile.key,
        "label": profile.label,
        "short_code": profile.short_code,
        "pc2_mirror": profile.pc2_mirror,
        "family": profile.family,
        "setup": profile.setup_tag,
        "stamp": stamp,
        "out_root": str(out),
        "model": model_id,
        "model_tag": model_tag,
        "benches": [name for name, _ in benches],
        "skills": profile_skills_snapshot(profile),
        "team_paths": active_team_paths_summary(),
        "validation": flash_cosim_manifest(),
        "env_flags": {
            "C2HLS_FLASH_EXPERIMENT": "1",
            "C2HLS_SITE": "team",
            "C2HLS_SKILL_PROMPT_MODE": os.getenv("C2HLS_SKILL_PROMPT_MODE", ""),
            "C2HLS_PACKAGED_SKILLS_JSON": os.getenv("C2HLS_PACKAGED_SKILLS_JSON", ""),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

    print(f"profile={profile.key} ({profile.label}) pc2_mirror={profile.pc2_mirror}")
    print(f"model={model_id} tag={model_tag}")
    print(f"out_root={out}")
    for bench, _ in benches:
        print(f"  - {bench} -> {_cell_dir(out, bench, model_tag, profile)}")

    if dry_run:
        print("dry-run ok")
        return 0

    apply_profile_env(profile)
    from c2hls import run_benchmark_multistep

    rows: list[dict[str, Any]] = []
    if (out / "matrix.json").exists():
        rows = json.loads((out / "matrix.json").read_text())

    for bench, bench_dir in benches:
        cell = _cell_dir(out, bench, model_tag, profile)
        cell.mkdir(parents=True, exist_ok=True)
        result_json = cell / f"{bench}_multistep_results.json"

        existing = _load_existing_result(result_json, bench)
        if existing is not None:
            print(
                f"SKIP {bench} success={existing.get('success')} (existing {result_json.name})",
                flush=True,
            )
            rows.append({
                "bench": bench,
                "model": model_id,
                "mode": "flash",
                "profile": profile.key,
                "pc2_mirror": profile.pc2_mirror,
                "inference": "commercial_api",
                "status": "ok",
                "wallclock_s": 0.0,
                "cell_dir": str(cell),
                "error": "",
                "summary": _compact_summary(existing),
            })
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

        rows.append({
            "bench": bench,
            "model": model_id,
            "mode": "flash",
            "profile": profile.key,
            "pc2_mirror": profile.pc2_mirror,
            "inference": "commercial_api",
            "status": status if result.get("success") else "fail",
            "wallclock_s": elapsed,
            "cell_dir": str(cell),
            "error": error or result.get("error"),
            "summary": _compact_summary(result),
        })
        print(f"DONE {bench} success={result.get('success')} elapsed={elapsed}s", flush=True)

    (out / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"matrix: {out / 'matrix.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flash synthesis batch via commercial LLM API (team site)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help=f"Profile key (e.g. nav_o, aav_n). All: {', '.join(DETERMINISTIC_ORDER)}",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--benches", type=str, default="", help="Comma-separated (default: all hlsfactory_*)")
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--model", type=str, default="", help="LLM model id (default: claude-sonnet-4-6)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-cosim",
        action="store_true",
        help="Skip cosim (csynth + csim only; matches PC2 pilot). Default: run cosim.",
    )
    args = parser.parse_args()

    if args.list_profiles:
        for key in DETERMINISTIC_ORDER:
            p = PROFILES[key]
            print(
                f"{key:16s} {p.short_code:8s}  mirror={p.pc2_mirror:8s}  "
                f"artifact={p.artifact_prefix}_<stamp>"
            )
        print(f"\nTOP5: {', '.join(k for k in PROFILES if k in {'nav_o','aav_n','nav_n','noskills_old','aav_o'})}")
        return 0

    if not args.profile or args.profile not in PROFILES:
        parser.error(f"--profile required; choose from: {', '.join(PROFILES)}")

    bootstrap_team_flash_env(skip_cosim=True if args.skip_cosim else None)
    model_id = resolve_model_id(args.model)
    blockers = preflight_api_run(model_id)
    if blockers:
        print("preflight failed:", file=sys.stderr)
        for msg in blockers:
            print(f"  - {msg}", file=sys.stderr)
        return 2

    profile = PROFILES[args.profile]
    stamp = args.stamp or os.getenv(profile.stamp_env) or datetime.now().strftime("%Y%m%d_%H%M%S")
    requested = _split_csv(args.benches) if args.benches else _all_hlsfactory_names()
    benches = _resolve_benches(requested)
    out_root = Path(args.out_root) if args.out_root else None

    return run_profile(
        profile,
        stamp=stamp,
        out_root=out_root,
        model_id=model_id,
        benches=benches,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
