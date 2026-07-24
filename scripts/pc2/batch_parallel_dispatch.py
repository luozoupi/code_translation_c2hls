"""Shared job dispatch for Rodinia vs tier_A batch_parallel campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from batch_parallel_bench import execute_job as rodinia_execute_job
from batch_parallel_config import BatchParallelConfig, benches_for_config, load_config
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from batch_parallel_autosa_dse_lib import (
    AUTOSA_DSE_VARIANT,
    configure_autosa_dse_campaign_env,
    is_autosa_dse_flash_workflow,
    resolve_autosa_dse_bench_map,
    autosa_dse_cell_dir,
)
from batch_parallel_autosa_lib import (
    AUTOSA_VARIANT,
    configure_autosa_campaign_env,
    is_autosa_workflow,
    resolve_autosa_bench_map,
    autosa_cell_dir,
)
from batch_parallel_tier_a_lib import (
    TIER_A_VARIANT,
    configure_tier_a_campaign_env,
    is_tier_a_workflow,
    model_cell_tag as tier_a_model_cell_tag,
    resolve_tier_a_bench_map,
    tier_a_cell_dir,
)
from batch_parallel_tier_b_lib import (
    TIER_B_FLASH_VARIANT,
    TIER_B_VARIANT,
    configure_tier_b_campaign_env,
    configure_tier_b_flash_campaign_env,
    flash_model_cell_tag,
    is_tier_b_flash_workflow,
    is_tier_b_gold_workflow,
    model_cell_tag as tier_b_gold_model_cell_tag,
    resolve_tier_b_bench_map,
    tier_b_cell_dir,
    tier_b_flash_cell_dir,
)
from batch_parallel_chathls_lib import (
    CHATHLS_FLASH_VARIANT,
    configure_chathls_flash_campaign_env,
    flash_model_cell_tag as chathls_flash_model_cell_tag,
    is_chathls_flash_workflow,
    resolve_chathls_bench_map,
    chathls_flash_cell_dir,
)
from batch_parallel_c2hlsc_lib import (
    C2HLSC_FLASH_VARIANT,
    configure_c2hlsc_flash_campaign_env,
    flash_model_cell_tag as c2hlsc_flash_model_cell_tag,
    is_c2hlsc_flash_workflow,
    resolve_c2hlsc_bench_map,
    c2hlsc_flash_cell_dir,
)
from batch_parallel_multistep_lib import (
    CHATHLS_MULTISTEP_VARIANT,
    TIER_A_MULTISTEP_VARIANT,
    TIER_B_MULTISTEP_VARIANT,
    chathls_multistep_cell_dir,
    configure_chathls_multistep_campaign_env,
    configure_tier_a_multistep_campaign_env,
    configure_tier_b_multistep_campaign_env,
    is_chathls_multistep_workflow,
    is_multistep_workflow,
    is_tier_a_multistep_workflow,
    is_tier_b_multistep_workflow,
    model_cell_tag as multistep_model_cell_tag,
    resolve_chathls_multistep_bench_map,
    resolve_tier_a_multistep_bench_map,
    resolve_tier_b_multistep_bench_map,
    tier_a_multistep_cell_dir,
    tier_b_multistep_cell_dir,
)
from multistep_batch_parallel_bench import execute_job as multistep_execute_job
from tier_a_batch_parallel_bench import execute_job as tier_a_execute_job
from tier_b_flash_batch_parallel_bench import execute_job as tier_b_flash_execute_job
from tier_b_gold_batch_parallel_bench import execute_job as tier_b_gold_execute_job
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
    if is_chathls_multistep_workflow(campaign):
        return resolve_chathls_multistep_bench_map(benches_order)
    if is_tier_a_multistep_workflow(campaign):
        return resolve_tier_a_multistep_bench_map(benches_order)
    if is_tier_b_multistep_workflow(campaign):
        return resolve_tier_b_multistep_bench_map(benches_order)
    if is_autosa_dse_flash_workflow(campaign):
        return resolve_autosa_dse_bench_map(benches_order)
    if is_chathls_flash_workflow(campaign):
        return resolve_chathls_bench_map(benches_order)
    if is_c2hlsc_flash_workflow(campaign):
        return resolve_c2hlsc_bench_map(benches_order)
    if is_autosa_workflow(campaign):
        return resolve_autosa_bench_map(benches_order)
    if is_tier_b_gold_workflow(campaign) or is_tier_b_flash_workflow(campaign):
        return resolve_tier_b_bench_map(benches_order)
    if is_tier_a_workflow(campaign):
        return resolve_tier_a_bench_map(benches_order)
    return {name: path for name, path in resolve_cosim_benches(benches_order)}


def _apply_campaign_tmp_run() -> None:
    """Nest HLS scratch under c2hls_tmp/<campaign_dirname>/… when on a PC2 campaign."""
    import os

    raw = (os.environ.get("BATCH_PARALLEL_CAMPAIGN_ROOT") or "").strip()
    if not raw:
        return
    run_slug = Path(raw).name
    if run_slug:
        os.environ["C2HLS_TMP_RUN"] = run_slug


def configure_campaign_env(campaign: dict[str, Any], variant_key: str) -> None:
    _apply_campaign_tmp_run()
    if is_chathls_multistep_workflow(campaign):
        configure_chathls_multistep_campaign_env()
        return
    if is_tier_a_multistep_workflow(campaign):
        configure_tier_a_multistep_campaign_env()
        return
    if is_tier_b_multistep_workflow(campaign):
        configure_tier_b_multistep_campaign_env()
        return
    if is_autosa_dse_flash_workflow(campaign):
        configure_autosa_dse_campaign_env()
        return
    if is_chathls_flash_workflow(campaign):
        configure_chathls_flash_campaign_env()
        return
    if is_c2hlsc_flash_workflow(campaign):
        configure_c2hlsc_flash_campaign_env()
        return
    if is_autosa_workflow(campaign):
        configure_autosa_campaign_env()
        return
    if is_tier_b_flash_workflow(campaign):
        configure_tier_b_flash_campaign_env()
        return
    if is_tier_b_gold_workflow(campaign):
        configure_tier_b_campaign_env()
        return
    if is_tier_a_workflow(campaign):
        configure_tier_a_campaign_env()
        return
    variant = VARIANTS.get(variant_key)
    if variant is None:
        raise ValueError(f"unknown variant {variant_key}")
    configure_fixed_cosim_flash_env(variant)


def validate_variant(campaign: dict[str, Any], variant_key: str) -> bool:
    if is_chathls_multistep_workflow(campaign):
        return variant_key == CHATHLS_MULTISTEP_VARIANT
    if is_tier_a_multistep_workflow(campaign):
        return variant_key == TIER_A_MULTISTEP_VARIANT
    if is_tier_b_multistep_workflow(campaign):
        return variant_key == TIER_B_MULTISTEP_VARIANT
    if is_autosa_dse_flash_workflow(campaign):
        return variant_key == AUTOSA_DSE_VARIANT
    if is_chathls_flash_workflow(campaign):
        return variant_key == CHATHLS_FLASH_VARIANT
    if is_c2hlsc_flash_workflow(campaign):
        return variant_key == C2HLSC_FLASH_VARIANT
    if is_autosa_workflow(campaign):
        return variant_key == AUTOSA_VARIANT
    if is_tier_b_flash_workflow(campaign):
        return variant_key == TIER_B_FLASH_VARIANT
    if is_tier_b_gold_workflow(campaign):
        return variant_key == TIER_B_VARIANT
    if is_tier_a_workflow(campaign):
        return variant_key == TIER_A_VARIANT
    return variant_key in VARIANTS


def cell_dir_for_job(
    campaign_root: Path,
    campaign: dict[str, Any],
    job: BatchParallelJob,
    model_id: str,
) -> Path:
    cell_root = campaign_root / "variants" / job.variant
    if is_chathls_multistep_workflow(campaign):
        return chathls_multistep_cell_dir(
            cell_root, job.bench, multistep_model_cell_tag(model_id)
        )
    if is_tier_a_multistep_workflow(campaign):
        return tier_a_multistep_cell_dir(
            cell_root, job.bench, multistep_model_cell_tag(model_id)
        )
    if is_tier_b_multistep_workflow(campaign):
        return tier_b_multistep_cell_dir(
            cell_root, job.bench, multistep_model_cell_tag(model_id)
        )
    if is_autosa_dse_flash_workflow(campaign):
        return autosa_dse_cell_dir(cell_root, job.bench, tier_a_model_cell_tag(model_id))
    if is_chathls_flash_workflow(campaign):
        return chathls_flash_cell_dir(
            cell_root, job.bench, chathls_flash_model_cell_tag(model_id)
        )
    if is_c2hlsc_flash_workflow(campaign):
        return c2hlsc_flash_cell_dir(
            cell_root, job.bench, c2hlsc_flash_model_cell_tag(model_id)
        )
    if is_autosa_workflow(campaign):
        return autosa_cell_dir(cell_root, job.bench, tier_a_model_cell_tag(model_id))
    if is_tier_b_flash_workflow(campaign):
        return tier_b_flash_cell_dir(
            cell_root, job.bench, flash_model_cell_tag(model_id)
        )
    if is_tier_b_gold_workflow(campaign):
        return tier_b_cell_dir(cell_root, job.bench, tier_b_gold_model_cell_tag(model_id))
    if is_tier_a_workflow(campaign):
        return tier_a_cell_dir(cell_root, job.bench, tier_a_model_cell_tag(model_id))
    variant = VARIANTS[job.variant]
    return rodinia_cell_dir(cell_root, job.bench, tier_a_model_cell_tag(model_id), variant)


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
    if is_multistep_workflow(campaign):
        multistep_execute_job(
            job=job,
            queue=queue,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            variant_key=job.variant,
            model_id=model_id,
            turns=turns,
        )
        return
    if is_tier_b_gold_workflow(campaign):
        tier_b_gold_execute_job(
            job=job,
            queue=queue,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            variant_key=job.variant,
            model_id=model_id,
            turns=turns,
        )
        return
    if (
        is_tier_b_flash_workflow(campaign)
        or is_autosa_dse_flash_workflow(campaign)
        or is_chathls_flash_workflow(campaign)
        or is_c2hlsc_flash_workflow(campaign)
    ):
        tier_b_flash_execute_job(
            job=job,
            queue=queue,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            variant_key=job.variant,
            model_id=model_id,
            turns=turns,
        )
        return
    if is_autosa_workflow(campaign) or is_tier_a_workflow(campaign):
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
