"""Run one flash bench for Fir batch_parallel."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

from c2hls_paths import BENCHMARKS_DIR, BENCHMARKS_COSIM_DIR, configure_site

from flash_lib import (
    SETUP_TAG,
    SETUP_TAG_90_BASE,
    SETUP_TAG_90_OVERLAY,
    configure_fir_flash_env,
    setup_tag_for_overlay,
)
from zero_shot_lib import FirZeroShotVariant, VARIANTS as ZERO_SHOT_VARIANTS, configure_fir_zero_shot_env


def model_cell_tag(model_id: str) -> str:
    low = (model_id or "").lower()
    if "devstral" in low:
        return "devstral2"
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return slug[:48] or "model"


def _zero_shot_workflow(workflow: str) -> bool:
    return workflow.startswith("zero_shot")


def _resolve_variant(campaign: dict[str, Any]) -> FirZeroShotVariant | None:
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    workflow = str(pilot.get("workflow") or "")
    if not _zero_shot_workflow(workflow):
        return None
    key = str(pilot.get("variant") or "")
    if key in ZERO_SHOT_VARIANTS:
        return ZERO_SHOT_VARIANTS[key]
    if workflow == "zero_shot_direct":
        return ZERO_SHOT_VARIANTS["direct"]
    return ZERO_SHOT_VARIANTS["phaseb"]


def resolve_bench_dir(bench: str) -> Path:
    roots = [BENCHMARKS_COSIM_DIR, BENCHMARKS_DIR]
    for root in roots:
        if not root.is_dir():
            continue
        for meta_path in sorted(root.glob("*/metadata.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            name = meta.get("benchmark") or meta_path.parent.name
            if name == bench:
                return meta_path.parent
    raise ValueError(f"unknown benchmark: {bench}")


def cell_dir(campaign_root: Path, bench: str, model_tag: str, setup_tag: str) -> Path:
    return campaign_root / bench / f"{model_tag}__{setup_tag}"


def load_endpoint_env(endpoint_file: Path) -> None:
    if not endpoint_file.is_file():
        return
    try:
        payload = json.loads(endpoint_file.read_text(encoding="utf-8"))
        url = payload.get("url")
        if url:
            os.environ["OPENAI_BASE_URL"] = str(url)
        model = payload.get("model")
        if model:
            os.environ["C2HLS_MODEL"] = str(model)
    except Exception:
        pass


def _flash_variant_key(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    return str(pilot.get("variant") or pilot.get("flash_variant") or "90_overlay").strip()


def _flash_overlay_enabled(variant_key: str) -> bool:
    key = variant_key.lower()
    if key in {"90_base", "base90", "90_only", "noskills_overlay"}:
        return False
    if key in {"90_overlay", "overlay", "90_plus_overlay", "no_rmw_overlay"}:
        return True
    return key != "90_base"


def _pilot_run_cosim(campaign: dict[str, Any]) -> bool | None:
    pilot = (campaign.get("config") or {}).get("pilot") or {}
    for field in ("run_cosim", "cosim"):
        if field in pilot:
            raw = pilot[field]
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return None


def configure_campaign_flash_env(campaign: dict[str, Any]) -> str:
    variant = _resolve_variant(campaign)
    if variant is not None:
        configure_fir_zero_shot_env(variant)
        return variant.setup_tag
    variant_key = _flash_variant_key(campaign)
    overlay = _flash_overlay_enabled(variant_key)
    run_cosim = _pilot_run_cosim(campaign)
    return configure_fir_flash_env(cosim=run_cosim, overlay=overlay)


def setup_tag_for_campaign(campaign: dict[str, Any]) -> str:
    variant = _resolve_variant(campaign)
    if variant is not None:
        return variant.setup_tag
    variant_key = _flash_variant_key(campaign)
    if variant_key in {SETUP_TAG_90_BASE, SETUP_TAG_90_OVERLAY}:
        return variant_key
    return setup_tag_for_overlay(overlay=_flash_overlay_enabled(variant_key))


def cell_dir_for_campaign(
    campaign_root: Path,
    bench: str,
    model_id: str,
    campaign: dict[str, Any],
) -> Path:
    return cell_dir(
        campaign_root,
        bench,
        model_cell_tag(model_id),
        setup_tag_for_campaign(campaign),
    )


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
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


def run_flash_bench(
    *,
    campaign_root: Path,
    bench: str,
    model_id: str,
    turns: int,
    endpoint_file: Path,
    campaign: dict[str, Any] | None = None,
    job_id: int | None = None,
    worker_id: str = "",
) -> dict[str, Any]:
    if campaign is None:
        from batch_parallel.config import load_campaign

        campaign = load_campaign(campaign_root)
    configure_site("fir")
    setup_tag = configure_campaign_flash_env(campaign)
    load_endpoint_env(endpoint_file)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["C2HLS_FIR_BATCH_CAMPAIGN_ROOT"] = str(campaign_root)
    os.environ["C2HLS_BATCH_LLM_HOOK_MODULE"] = "batch_parallel.gpu_state"
    if job_id is not None:
        os.environ["C2HLS_FIR_BATCH_JOB_ID"] = str(job_id)
    os.environ["C2HLS_FIR_BATCH_BENCH"] = bench
    if worker_id:
        os.environ["C2HLS_FIR_BATCH_WORKER"] = worker_id

    from c2hls import run_benchmark_multistep

    model_tag = model_cell_tag(model_id)
    bench_dir = resolve_bench_dir(bench)
    cell = cell_dir(campaign_root, bench, model_tag, setup_tag)
    cell.mkdir(parents=True, exist_ok=True)
    result_json = cell / f"{bench}_multistep_results.json"

    t0 = time.time()
    status = "ok"
    error = ""
    try:
        result = run_benchmark_multistep(
            str(bench_dir),
            output_dir=str(cell),
            gpt_model=model_id,
            turns_limitation=turns,
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

    elapsed = round(time.time() - t0, 1)
    if not result_json.exists():
        result_json.write_text(json.dumps(result, indent=2) + "\n")

    retry_error = ""
    if result_json.is_file():
        try:
            import sys

            sys.path.insert(0, str(REPO / "scripts" / "explorer"))
            from metrics import RETRYABLE_RUN_ISSUES, bench_run_issues_from_multistep_doc

            doc = json.loads(result_json.read_text(encoding="utf-8"))
            issues = set(bench_run_issues_from_multistep_doc(doc))
            if issues & RETRYABLE_RUN_ISSUES:
                flash = next(
                    (s for s in (doc.get("steps") or []) if s.get("step_name") == "flash"),
                    {},
                )
                retry_error = str(
                    flash.get("attempt_error")
                    or flash.get("error")
                    or next(iter(issues & RETRYABLE_RUN_ISSUES))
                )
        except Exception:
            retry_error = ""

    pilot = (campaign.get("config") or {}).get("pilot") or {}
    variant_key = _flash_variant_key(campaign)
    run_ok = bool(result.get("success")) and not retry_error
    return {
        "bench": bench,
        "model": model_id,
        "mode": "flash",
        "workflow": pilot.get("workflow") or "flash",
        "variant": pilot.get("variant") or variant_key,
        "flash_overlay": _flash_overlay_enabled(variant_key),
        "run_cosim": os.getenv("C2HLS_RUN_COSIM", "0"),
        "status": status if run_ok else "fail",
        "wallclock_s": elapsed,
        "cell_dir": str(cell),
        "error": retry_error or error or result.get("error"),
        "summary": compact_summary(result),
        "result_path": str(result_json),
    }
