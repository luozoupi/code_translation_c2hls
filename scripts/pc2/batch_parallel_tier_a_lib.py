"""Tier A helpers for batch_parallel campaigns."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tier_a_flash_lib import (
    SETUP_TAG,
    configure_tier_a_flash_90skills_env,
    resolve_tier_a_benches,
)

TIER_A_VARIANT = "tier_a_90"
WORKFLOW_TIER_A_FLASH = "tier_a_flash"


def resolve_tier_a_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_a_benches(benches)}


def configure_tier_a_campaign_env() -> None:
    configure_tier_a_flash_90skills_env()


def tier_a_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG}"


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_tier_a_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_TIER_A_FLASH


def model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)
