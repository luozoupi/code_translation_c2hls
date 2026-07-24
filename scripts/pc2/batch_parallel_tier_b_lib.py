"""Tier B helpers for batch_parallel campaigns (gold gate + MachSuite flash)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tier_b_flash_lib import SETUP_TAG as FLASH_SETUP_TAG
from tier_b_flash_lib import configure_tier_b_flash_aav_n_env
from tier_b_gold_lib import configure_tier_b_gold_env, resolve_tier_b_benches

TIER_B_VARIANT = "tier_b_machsuite"
TIER_B_FLASH_VARIANT = "tier_b_aav_n"
WORKFLOW_TIER_B_GOLD = "tier_b_gold"
WORKFLOW_TIER_B_FLASH = "tier_b_flash"
SETUP_TAG = "gold_gate"


def resolve_tier_b_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_b_benches(benches)}


def configure_tier_b_campaign_env() -> None:
    configure_tier_b_gold_env()


def configure_tier_b_flash_campaign_env() -> None:
    configure_tier_b_flash_aav_n_env()


def tier_b_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    _ = model_tag
    return cell_root / bench / SETUP_TAG


def tier_b_flash_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{FLASH_SETUP_TAG}"


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_tier_b_gold_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_TIER_B_GOLD


def is_tier_b_flash_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_TIER_B_FLASH


def model_cell_tag(model_id: str) -> str:
    _ = model_id
    return SETUP_TAG


def flash_model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)
