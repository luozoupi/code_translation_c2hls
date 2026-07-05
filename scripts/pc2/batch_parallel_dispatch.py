"""Shared job dispatch for Rodinia vs tier_A batch_parallel campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from batch_parallel_bench import execute_job as rodinia_execute_job
from batch_parallel_config import BatchParallelConfig, benches_for_config, load_config
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from batch_parallel_tier_a_lib import (
    TIER_A_VARIANT,
    configure_tier_a_campaign_env,
    is_tier_a_workflow,
    model_cell_tag,
    resolve_tier_a_bench_map,
    tier_a_cell_dir,
)
from tier_a_batch_parallel_bench import execute_job as tier_a_execute_job
from flash_fixed_cosim_lib import VARIANTS, configure_fixed_cosim_flash_env, resolve_cosim_benches
from run_flash_fixed_cosim_batch import _cell_dir as rodinia_cell_dir


def seed_kwargs_for_campaign(campaign: dict[str, Any], cfg: BatchParallelConfig | None = None) -> dict[str, str]:
    from batch_parallel_config import seed_kwargs_for_workflow

    workflow = (
        (campaign.get("config") or {}).get("pilot") or {}
    ).get("workflow") or (cfg.pilot_workflow if cfg else "flash")
    return seed_kwargs_for_workflow(str(workflow))


def resolve_bench_map(
    campaign: dict[str, Any],
    cfg: BatchParallelConfig,
    benches_order: list[str],
) -> dict[str, Path]:
    if is_tier_a_workflow(campaign):
        return resolve_tier_a_bench_map(benches_order)
    return {name: path for name, path in resolve_cosim_benches(benches_order)}


def configure_campaign_env(campaign: dict[str, Any], variant_key: str) -> None:
    if is_tier_a_workflow(campaign):
        configure_tier_a_campaign_env()
        return
    variant = VARIANTS.get(variant_key)
    if variant is None:
        raise ValueError(f"unknown variant {variant_key}")
    configure_fixed_cosim_flash_env(variant)


def validate_variant(campaign: dict[str, Any], variant_key: str) -> bool:
    if is_tier_a_workflow(campaign):
        return variant_key == TIER_A_VARIANT
    return variant_key in VARIANTS


def cell_dir_for_job(
    campaign_root: Path,
    campaign: dict[str, Any],
    job: BatchParallelJob,
    model_id: str,
) -> Path:
    model_tag = model_cell_tag(model_id)
    cell_root = campaign_root / "variants" / job.variant
    if is_tier_a_workflow(campaign):
        return tier_a_cell_dir(cell_root, job.bench, model_tag)
    variant = VARIANTS[job.variant]
    return rodinia_cell_dir(cell_root, job.bench, model_tag, variant)


def run_batch_parallel_job(
    *,
    job: BatchParallelJob,
    queue: BatchParallelQueue,
    campaign: dict[str, Any],
    bench_dir: Path,
    cell_dir: Path,
    model_id: str,
    turns: int,
) -> None:
    if is_tier_a_workflow(campaign):
        tier_a_execute_job(
            job=job,
            queue=queue,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            variant_key=job.variant,
            model_id=model_id,
            turns=turns,
        )
        return
    rodinia_execute_job(
        job=job,
        queue=queue,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        variant_key=job.variant,
        model_id=model_id,
        turns=turns,
    )


def campaign_benches_resolved(campaign: dict[str, Any], cfg: BatchParallelConfig | None = None) -> list[str]:
    if cfg is None:
        cfg = load_config()
    stored = campaign.get("config") or {}
    pilot = stored.get("pilot") or {}
    benches = [str(b) for b in (pilot.get("benches") or [])]
    if benches:
        cfg.pilot_benches = benches
    elif not cfg.pilot_benches:
        cfg = load_config()
    ordered = benches_for_config(cfg)
    return cfg.sort_benches(ordered)
