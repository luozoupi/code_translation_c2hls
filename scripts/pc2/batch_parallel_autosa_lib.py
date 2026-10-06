"""AutoSA helpers for batch_parallel campaigns."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autosa_flash_lib import (
    VARIANT_AAV_N,
    VARIANT_AAV_N_90,
    VARIANT_AAV_N_GF,
    VARIANT_GEMM_FAMILY,
    VARIANT_GENERIC,
    VARIANT_GOLD,
    VARIANT_NAV_N,
    VARIANT_NOSKILLS,
    VARIANT_ONCHIP,
    VARIANT_ONE_SHOT,
    VARIANT_SYSTOLIC_IO,
    VARIANT_ZERO_SHOT,
    apply_bench_timeouts_from_meta,
    configure_autosa_flash_aav_n_env,
    configure_autosa_flash_aav_n_90_env,
    configure_autosa_flash_aav_n_gf_env,
    configure_autosa_flash_gemm_family_env,
    configure_autosa_flash_generic_env,
    configure_autosa_flash_nav_n_env,
    configure_autosa_flash_noskills_env,
    configure_autosa_flash_onchip_env,
    configure_autosa_flash_one_shot_env,
    configure_autosa_flash_systolic_io_env,
    configure_autosa_flash_zero_shot_env,
    configure_autosa_gold_env,
    resolve_autosa_benches,
    setup_tag_for_variant,
)

AUTOSA_VARIANT = VARIANT_NAV_N
AUTOSA_AAV_N_VARIANT = VARIANT_AAV_N
AUTOSA_AAV_N_GF_VARIANT = VARIANT_AAV_N_GF
AUTOSA_VARIANTS = {
    VARIANT_NAV_N,
    VARIANT_AAV_N,
    VARIANT_AAV_N_90,
    VARIANT_AAV_N_GF,
    VARIANT_NOSKILLS,
    VARIANT_ZERO_SHOT,
    VARIANT_ONE_SHOT,
    VARIANT_ONCHIP,
    VARIANT_GENERIC,
    VARIANT_GEMM_FAMILY,
    VARIANT_SYSTOLIC_IO,
    VARIANT_GOLD,
}
WORKFLOW_AUTOSA_FLASH = "autosa_flash"
WORKFLOW_AUTOSA_GOLD = "autosa_gold"


def resolve_autosa_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_autosa_benches(benches)}


def configure_autosa_campaign_env(variant_key: str = AUTOSA_VARIANT) -> None:
    if variant_key == VARIANT_AAV_N_GF:
        configure_autosa_flash_aav_n_gf_env()
        return
    if variant_key == VARIANT_AAV_N_90:
        configure_autosa_flash_aav_n_90_env()
        return
    if variant_key == VARIANT_AAV_N:
        configure_autosa_flash_aav_n_env()
        return
    if variant_key == VARIANT_NOSKILLS:
        configure_autosa_flash_noskills_env()
        return
    if variant_key == VARIANT_ZERO_SHOT:
        configure_autosa_flash_zero_shot_env()
        return
    if variant_key == VARIANT_ONE_SHOT:
        configure_autosa_flash_one_shot_env()
        return
    if variant_key == VARIANT_GOLD:
        configure_autosa_gold_env()
        return
    if variant_key == VARIANT_ONCHIP:
        configure_autosa_flash_onchip_env()
        return
    if variant_key == VARIANT_GENERIC:
        configure_autosa_flash_generic_env()
        return
    if variant_key == VARIANT_GEMM_FAMILY:
        configure_autosa_flash_gemm_family_env()
        return
    if variant_key == VARIANT_SYSTOLIC_IO:
        configure_autosa_flash_systolic_io_env()
        return
    configure_autosa_flash_nav_n_env()


def autosa_cell_dir(
    cell_root: Path,
    bench: str,
    model_tag: str,
    variant_key: str = AUTOSA_VARIANT,
) -> Path:
    tag = setup_tag_for_variant(variant_key)
    return cell_root / bench / f"{model_tag}__{tag}"


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_autosa_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_AUTOSA_FLASH


def is_autosa_gold_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_AUTOSA_GOLD


def model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)


def apply_autosa_bench_timeouts(meta: dict) -> None:
    apply_bench_timeouts_from_meta(meta)
