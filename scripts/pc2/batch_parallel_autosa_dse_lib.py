"""AutoSA DSE batch_parallel helpers (90-skill aav_n flash + cosim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autosa_dse_flash_lib import (
    SETUP_TAG,
    apply_bench_timeouts_from_meta,
    configure_autosa_dse_flash_aav_n_env,
    resolve_autosa_dse_benches,
)

AUTOSA_DSE_VARIANT = "autosa_dse_aav_n"
WORKFLOW_AUTOSA_DSE_FLASH = "autosa_dse_flash"


def resolve_autosa_dse_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_autosa_dse_benches(benches)}


def configure_autosa_dse_campaign_env() -> None:
    configure_autosa_dse_flash_aav_n_env()


def autosa_dse_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG}"


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_autosa_dse_flash_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_AUTOSA_DSE_FLASH


def model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)
