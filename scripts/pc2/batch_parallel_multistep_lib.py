"""Multistep batch_parallel helpers for ChatHLS / tier_A / tier_B / AutoSA corpora."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from autosa_flash_lib import resolve_autosa_benches
from chathls_flash_lib import (
    CHATHLS_READY_ROOT,
    configure_chathls_flash_aav_n_env,
    resolve_chathls_benches,
)
from multistep_fixed_cosim_lib import (
    VARIANTS as MULTISTEP_FIXED_VARIANTS,
    configure_fixed_cosim_multistep_env,
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
AUTOSA_MS_AAV_N_VARIANT = "autosa_ms_aav_n"
AUTOSA_MS_MSSS_VARIANT = "autosa_ms_msss"
AUTOSA_MS_VARIANTS = {AUTOSA_MS_AAV_N_VARIANT, AUTOSA_MS_MSSS_VARIANT}

WORKFLOW_CHATHLS_MULTISTEP = "chathls_multistep"
WORKFLOW_TIER_A_MULTISTEP = "tier_a_multistep"
WORKFLOW_TIER_B_MULTISTEP = "tier_b_multistep"
WORKFLOW_AUTOSA_MULTISTEP = "autosa_multistep"

SETUP_TAG_CHATHLS = "multistep__chathls__aav_n"
SETUP_TAG_TIER_A = "multistep__tier_a__aav_n"
SETUP_TAG_TIER_B = "multistep__tier_b__aav_n"
SETUP_TAG_AUTOSA_AAV = "multistep__autosa__aav_n"
SETUP_TAG_AUTOSA_MSSS = "multistep__autosa__msss"

_PKG = Path(__file__).resolve().parents[2] / "hls_full_optimization_skills_schema_1_1_package"
SKILLS_90_PLUS_V2 = (
    _PKG / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json"
)
MSSS_V1_NORMW_DIR = _PKG / "multistep_gf_v1_normw"
AAV_PACK_ENV = "C2HLS_AUTOSA_MS_AAV_PACK"
MSSS_PACK_ENV = "C2HLS_AUTOSA_MS_MSSS_PACK"
AAV_PACK_90_V2 = "90_v2"
MSSS_PACK_V1_NORMW = "v1_normw"

_AUTOSA_REPEAT_RE = re.compile(r"^(autosa_mm)_r\d{2}$")

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


def is_autosa_multistep_workflow(campaign: dict[str, Any]) -> bool:
    return workflow_from_campaign(campaign) == WORKFLOW_AUTOSA_MULTISTEP


def is_multistep_workflow(campaign: dict[str, Any]) -> bool:
    return (
        is_chathls_multistep_workflow(campaign)
        or is_tier_a_multistep_workflow(campaign)
        or is_tier_b_multistep_workflow(campaign)
        or is_autosa_multistep_workflow(campaign)
    )


def resolve_chathls_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_chathls_benches(benches)}


def resolve_tier_a_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_a_benches(benches)}


def resolve_tier_b_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    return {name: path for name, path in resolve_tier_b_benches(benches)}


def autosa_repeat_source(name: str) -> str:
    match = _AUTOSA_REPEAT_RE.fullmatch(name)
    return match.group(1) if match else name


def resolve_autosa_multistep_bench_map(benches: list[str]) -> dict[str, Path]:
    sources: list[str] = []
    alias: dict[str, str] = {}
    for name in benches:
        src = autosa_repeat_source(name)
        alias[name] = src
        if src not in sources:
            sources.append(src)
    resolved = {src: path for src, path in resolve_autosa_benches(sources)}
    return {name: resolved[alias[name]] for name in benches}


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
    # Mitigate xelab SIGSEGV on PC2 compute nodes (see hls_eval C2HLS_COSIM_XELAB_MT_OFF).
    os.environ.setdefault("C2HLS_COSIM_XELAB_MT_OFF", "1")
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


def _scrub_autosa_multistep_extras() -> None:
    os.environ["C2HLS_RAG"] = "0"
    os.environ["C2HLS_RAG_ENABLE"] = "0"
    os.environ["C2HLS_RAG_SCRAPE"] = "0"
    os.environ["C2HLS_RAG2"] = "0"
    os.environ["C2HLS_POST_FLASH_LATENCY_OPT"] = "0"
    os.environ["C2HLS_POST_FLASH_PRAGMA_OPT"] = "0"
    os.environ["C2HLS_POST_FLASH_DATAFLOW"] = "0"
    os.environ["C2HLS_POST_FLASH_DSE"] = "0"
    os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
    os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
    os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
    os.environ["C2HLS_ENFORCEMENT"] = "0"
    os.environ["C2HLS_STRATEGY"] = "static"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ.setdefault("C2HLS_MULTISTEP_SKIP_FINAL_COSIM", "1")


def apply_autosa_ms_msss_pack_override() -> None:
    pack = (os.getenv(MSSS_PACK_ENV) or "").strip()
    if not pack or pack in {"90_v2", "default"}:
        return
    if pack not in {MSSS_PACK_V1_NORMW, "v1"}:
        raise ValueError(f"unknown {MSSS_PACK_ENV}={pack!r}")
    if not MSSS_V1_NORMW_DIR.is_dir():
        raise FileNotFoundError(f"missing msss v1+no-RMW slice: {MSSS_V1_NORMW_DIR}")
    os.environ["C2HLS_MSSS_SKILLS_DIR"] = str(MSSS_V1_NORMW_DIR.resolve())


def apply_autosa_ms_aav_pack_override() -> None:
    pack = (os.getenv(AAV_PACK_ENV) or "").strip()
    if not pack or pack in {"v1_normw", "v1", "default"}:
        return
    if pack != AAV_PACK_90_V2:
        raise ValueError(f"unknown {AAV_PACK_ENV}={pack!r}")
    if not SKILLS_90_PLUS_V2.is_file():
        raise FileNotFoundError(f"missing 90+v2 pack: {SKILLS_90_PLUS_V2}")
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(SKILLS_90_PLUS_V2.resolve())
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)


def configure_autosa_multistep_msss_env() -> None:
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = "msss"
    os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)
    os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)
    apply_autosa_ms_msss_pack_override()


def configure_autosa_multistep_campaign_env(
    variant_key: str = AUTOSA_MS_AAV_N_VARIANT,
) -> None:
    if variant_key == AUTOSA_MS_MSSS_VARIANT:
        configure_autosa_multistep_msss_env()
    else:
        configure_fixed_cosim_multistep_env(MULTISTEP_FIXED_VARIANTS["aav_n"])
        apply_autosa_ms_aav_pack_override()
    _apply_multistep_common_env()
    _scrub_autosa_multistep_extras()


def model_cell_tag(model_id: str) -> str:
    from run_tier_a_flash_smoke_batch import model_cell_tag as _tag

    return _tag(model_id)


def chathls_multistep_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG_CHATHLS}"


def tier_a_multistep_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG_TIER_A}"


def tier_b_multistep_cell_dir(cell_root: Path, bench: str, model_tag: str) -> Path:
    return cell_root / bench / f"{model_tag}__{SETUP_TAG_TIER_B}"


def autosa_multistep_setup_tag(variant_key: str) -> str:
    if variant_key == AUTOSA_MS_MSSS_VARIANT:
        return SETUP_TAG_AUTOSA_MSSS
    return SETUP_TAG_AUTOSA_AAV


def autosa_multistep_cell_dir(
    cell_root: Path,
    bench: str,
    model_tag: str,
    variant_key: str = AUTOSA_MS_AAV_N_VARIANT,
) -> Path:
    return cell_root / bench / f"{model_tag}__{autosa_multistep_setup_tag(variant_key)}"
