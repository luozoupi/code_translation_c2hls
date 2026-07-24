"""c2hlsc flash batch_parallel helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from c2hlsc_flash_lib import SETUP_TAG, configure_c2hlsc_flash_aav_n_env, resolve_c2hlsc_benches

C2HLSC_FLASH_VARIANT = "c2hlsc_aav_n"
WORKFLOW_C2HLSC_FLASH = "c2hlsc_flash"


def resolve_c2hlsc_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_c2hlsc_benches(benches)}


def configure_c2hlsc_flash_campaign_env() -> None:
    configure_c2hlsc_flash_aav_n_env()


def c2hlsc_flash_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG}"


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_c2hlsc_flash_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_C2HLSC_FLASH


def flash_model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)
