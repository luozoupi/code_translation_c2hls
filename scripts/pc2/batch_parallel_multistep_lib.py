"""Multistep batch_parallel helpers for ChatHLS / tier_A / tier_B corpora."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chathls_flash_lib import (
    CHATHLS_READY_ROOT,
    configure_chathls_flash_aav_n_env,
    resolve_chathls_benches,
)
from tier_a_flash_lib import (
    TIER_A_READY_ROOT,
    configure_tier_a_flash_90skills_env,
    resolve_tier_a_benches,
)
from tier_b_flash_lib import configure_tier_b_flash_aav_n_env
from tier_b_gold_lib import TIER_B_READY_ROOT, resolve_tier_b_benches

CHATHLS_MULTISTEP_VARIANT = "chathls_ms_aav_n"
TIER_A_MULTISTEP_VARIANT = "tier_a_ms_aav_n"
TIER_B_MULTISTEP_VARIANT = "tier_b_ms_aav_n"

WORKFLOW_CHATHLS_MULTISTEP = "chathls_multistep"
WORKFLOW_TIER_A_MULTISTEP = "tier_a_multistep"
WORKFLOW_TIER_B_MULTISTEP = "tier_b_multistep"

SETUP_TAG_CHATHLS = "multistep__chathls__aav_n"
SETUP_TAG_TIER_A = "multistep__tier_a__aav_n"
SETUP_TAG_TIER_B = "multistep__tier_b__aav_n"

DEFAULT_OPT_STEPS = [
    "tiling",
    "pipeline",
    "unroll",
    "coalescing",
    "doublebuffer",
]


def opt_steps_from_env() -> list[str]:
    raw = (os.getenv("C2HLS_MULTISTEP_OPT_STEPS") or "").strip()
    if not raw:
        return list(DEFAULT_OPT_STEPS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def workflow_from_campaign(campaign: dict[str, Any]) -> str:
    pilot = (campaign.get("config") or {}).get("pilot") or campaign.get("pilot") or {}
    return str(pilot.get("workflow") or "flash")


def is_chathls_multistep_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_CHATHLS_MULTISTEP


def is_tier_a_multistep_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_TIER_A_MULTISTEP


def is_tier_b_multistep_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_TIER_B_MULTISTEP


def is_multistep_workflow(campaign: dict[str, Any]) -> bool:
    return (
        is_chathls_multistep_workflow(campaign)
        or is_tier_a_multistep_workflow(campaign)
        or is_tier_b_multistep_workflow(campaign)
    )


def resolve_chathls_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_chathls_benches(benches)}


def resolve_tier_a_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_a_benches(benches)}


def resolve_tier_b_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_b_benches(benches)}


def _apply_multistep_common_env() -> None:
    os.environ["C2HLS_STRATEGY"] = "static"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    # Intermediate synth jobs force RUN_COSIM=0 via configure_synth_env.
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_MULTISTEP_OPT_STEPS", ",".join(DEFAULT_OPT_STEPS))


def configure_chathls_multistep_campaign_env() -> None:
    configure_chathls_flash_aav_n_env()
    _apply_multistep_common_env()
    _ = CHATHLS_READY_ROOT


def configure_tier_a_multistep_campaign_env() -> None:
    configure_tier_a_flash_90skills_env()
    _apply_multistep_common_env()
    _ = TIER_A_READY_ROOT


def configure_tier_b_multistep_campaign_env() -> None:
    configure_tier_b_flash_aav_n_env()
    _apply_multistep_common_env()
    _ = TIER_B_READY_ROOT


def model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)


def chathls_multistep_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG_CHATHLS}"


def tier_a_multistep_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG_TIER_A}"


def tier_b_multistep_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG_TIER_B}"
