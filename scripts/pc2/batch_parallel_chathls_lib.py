"""ChatHLS flash batch_parallel helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chathls_flash_lib import (
    configure_chathls_flash_aav_n_env,
    get_setup_tag,
    resolve_chathls_benches,
)

CHATHLS_FLASH_VARIANT = "chathls_aav_n"
WORKFLOW_CHATHLS_FLASH = "chathls_flash"


def resolve_chathls_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_chathls_benches(benches)}


def configure_chathls_flash_campaign_env() -> None:
    configure_chathls_flash_aav_n_env()


def chathls_flash_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{get_setup_tag()}"


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_chathls_flash_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_CHATHLS_FLASH


def flash_model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)
