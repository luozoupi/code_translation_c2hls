"""Frozen ii-target-miss skill libraries for the flash new-skills matrix.

Used by ``scripts/flash_api/`` (commercial LLM) and ``scripts/pc2/`` (vLLM).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

REPO = Path(__file__).resolve().parents[2]
_PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"

# Packaged base libraries (unchanged). Flash overlay is flash_no_RMW_m_axi_skill_entries.json.
NEW_SKILLS_JSON_73 = _PKG / "skills_ii_target_miss_solutions_added(73skills).json"
NEW_SKILLS_JSON_90 = _PKG / "skills_ii_target_miss_solutions_added(90skills).json"
NEW_SKILLS_JSON = NEW_SKILLS_JSON_90

LEGACY_SKILLS_JSON = (
    REPO / "hls_full_optimization_skills_schema_1_1_package" / "skills.json"
)

InferenceKind = Literal["api", "vllm"]

_COMMON_FLASH_ENV: dict[str, str] = {
    "C2HLS_PHASEB_MODE": "functional",
    "C2HLS_PHASE8_BASELINE_ALIGN": "0",
    "C2HLS_PHASE5_GT_PREPOP": "0",
    "C2HLS_HW_EMU_FINAL": "0",
    "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS": "1",
    "C2HLS_GT_BASELINE_FALLBACK": "1",
    "C2HLS_SYNTH_TIMEOUT": "1200",
    "C2HLS_CSIM_TIMEOUT": "180",
    "C2HLS_COSIM_TIMEOUT": "1200",
    "C2HLS_LLM_TIMEOUT": "900",
}

# PC2 vLLM pilot only — team API cosim is set in flash_shared.team_env.
_VLLM_COSIM_OFF: dict[str, str] = {
    "C2HLS_RUN_COSIM": "0",
    "C2HLS_COSIM_REQUIRED": "0",
    "C2HLS_REFERENCE_COSIM": "0",
    "C2HLS_COSIM_TRACE_LEVEL": "none",
}


@dataclass(frozen=True)
class FlashNewVariant:
    key: str
    label: str
    session_id: str
    artifact_prefix: str
    setup_tag: str
    stamp_env: str
    out_env: str
    force_skill_prompts: bool
    skill_prompt_mode: str
    bottleneck_positive: Optional[int] = None
    bottleneck_avoid: Optional[int] = None
    skills_in_prompt: bool = True

    @property
    def matrix_family(self) -> str:
        return "flash_new_skills_ii_target_miss"


VARIANTS: dict[str, FlashNewVariant] = {
    "noskills_new": FlashNewVariant(
        key="noskills_new",
        label="Noskills_new",
        session_id="flash_noskills_new",
        artifact_prefix="flash_noskills_new",
        setup_tag="flash__noskills_new",
        stamp_env="C2HLS_FLASH_NOSKILLS_NEW_STAMP",
        out_env="C2HLS_FLASH_NOSKILLS_NEW_OUT",
        force_skill_prompts=False,
        skill_prompt_mode="bottleneck",
        skills_in_prompt=False,
    ),
    "bn_skills_new_2_2": FlashNewVariant(
        key="bn_skills_new_2_2",
        label="Bn_skills_new_2_2",
        session_id="flash_bn_skills_new_2_2",
        artifact_prefix="flash_bn_skills_new_2_2",
        setup_tag="flash__bn_skills_new_2_2",
        stamp_env="C2HLS_FLASH_BN_SKILLS_NEW_2_2_STAMP",
        out_env="C2HLS_FLASH_BN_SKILLS_NEW_2_2_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="bottleneck",
        bottleneck_positive=2,
        bottleneck_avoid=2,
    ),
    "bn_skills_new_4_2": FlashNewVariant(
        key="bn_skills_new_4_2",
        label="Bn_skills_new_4_2",
        session_id="flash_bn_skills_new_4_2",
        artifact_prefix="flash_bn_skills_new_4_2",
        setup_tag="flash__bn_skills_new_4_2",
        stamp_env="C2HLS_FLASH_BN_SKILLS_NEW_4_2_STAMP",
        out_env="C2HLS_FLASH_BN_SKILLS_NEW_4_2_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="bottleneck",
        bottleneck_positive=4,
        bottleneck_avoid=2,
    ),
    "bn_skills_new_6_2": FlashNewVariant(
        key="bn_skills_new_6_2",
        label="Bn_skills_new_6_2",
        session_id="flash_bn_skills_new_6_2",
        artifact_prefix="flash_bn_skills_new_6_2",
        setup_tag="flash__bn_skills_new_6_2",
        stamp_env="C2HLS_FLASH_BN_SKILLS_NEW_6_2_STAMP",
        out_env="C2HLS_FLASH_BN_SKILLS_NEW_6_2_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="bottleneck",
        bottleneck_positive=6,
        bottleneck_avoid=2,
    ),
    "all_new_skills_avoids_global": FlashNewVariant(
        key="all_new_skills_avoids_global",
        label="flash_all_new_skills_avoids_global",
        session_id="flash_all_new_skills_avoids_global",
        artifact_prefix="flash_all_new_skills_avoids_global",
        setup_tag="flash__all_new_skills_avoids_global",
        stamp_env="C2HLS_FLASH_ALL_NEW_SKILLS_AVOIDS_GLOBAL_STAMP",
        out_env="C2HLS_FLASH_ALL_NEW_SKILLS_AVOIDS_GLOBAL_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="all_skills_avoids_global",
    ),
    "all_new_skills_no_avoids_global": FlashNewVariant(
        key="all_new_skills_no_avoids_global",
        label="flash_all_new_skills_no_avoids_global",
        session_id="flash_all_new_skills_no_avoids_global",
        artifact_prefix="flash_all_new_skills_no_avoids_global",
        setup_tag="flash__all_new_skills_no_avoids_global",
        stamp_env="C2HLS_FLASH_ALL_NEW_SKILLS_NO_AVOIDS_GLOBAL_STAMP",
        out_env="C2HLS_FLASH_ALL_NEW_SKILLS_NO_AVOIDS_GLOBAL_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="all_skills_no_avoids_global",
    ),
}

VARIANT_ORDER = [
    "noskills_new",
    "bn_skills_new_2_2",
    "bn_skills_new_4_2",
    "bn_skills_new_6_2",
    "all_new_skills_avoids_global",
    "all_new_skills_no_avoids_global",
]


def skills_json_for_variant(variant: FlashNewVariant) -> Optional[Path]:
    """Return packaged skills JSON only when skills are injected into the flash prompt."""
    if not variant.force_skill_prompts:
        return None
    if variant.key == "all_new_skills_avoids_global":
        return NEW_SKILLS_JSON_90
    if variant.key == "all_new_skills_no_avoids_global":
        return NEW_SKILLS_JSON_73
    return NEW_SKILLS_JSON_73


def _apply_packaged_skills_env(skills_path: Optional[Path]) -> None:
    if skills_path is not None:
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(skills_path)
        os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    else:
        os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
        os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)


def _apply_flash_skill_entries_env(enabled: bool) -> None:
    from c2hls_paths import FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON

    if enabled:
        os.environ["C2HLS_FLASH_SKILL_ENTRIES_JSON"] = str(
            FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON
        )
    else:
        os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)


def configure_new_skills_env(
    variant: FlashNewVariant,
    *,
    inference: InferenceKind = "api",
) -> None:
    """Apply env for one new-matrix variant."""
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    _apply_packaged_skills_env(skills_json_for_variant(variant))
    _apply_flash_skill_entries_env(variant.force_skill_prompts)

    if variant.force_skill_prompts:
        os.environ["C2HLS_SKILL_MODE"] = "skill_on"
        os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = variant.skill_prompt_mode
        if variant.bottleneck_positive is not None:
            os.environ["C2HLS_BOTTLENECK_POSITIVE_SKILLS"] = str(variant.bottleneck_positive)
        if variant.bottleneck_avoid is not None:
            os.environ["C2HLS_BOTTLENECK_AVOID_SKILLS"] = str(variant.bottleneck_avoid)
    else:
        os.environ["C2HLS_SKILL_MODE"] = "skill_off"
        os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
        os.environ.pop("C2HLS_SKILL_PROMPT_MODE", None)
        os.environ.pop("C2HLS_BOTTLENECK_POSITIVE_SKILLS", None)
        os.environ.pop("C2HLS_BOTTLENECK_AVOID_SKILLS", None)

    flash_env = dict(_COMMON_FLASH_ENV)
    if inference == "vllm":
        flash_env.update(_VLLM_COSIM_OFF)
        flash_env["OPENAI_API_KEY"] = "EMPTY"
    for key, value in flash_env.items():
        os.environ.setdefault(key, value)


def variant_env_snapshot(variant: FlashNewVariant) -> dict[str, str]:
    skills_path = skills_json_for_variant(variant)
    snap = {
        "matrix_family": variant.matrix_family,
        "skills_json": str(skills_path) if skills_path is not None else "",
        "skills_json_mode": (
            "packaged_base_plus_flash_overlay"
            if skills_path is not None
            else "none"
        ),
        "flash_skill_entries_json": (
            os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON", "")
            if variant.force_skill_prompts
            else ""
        ),
        "legacy_skills_json": str(LEGACY_SKILLS_JSON),
        "C2HLS_FORCE_SKILL_PROMPTS": "1" if variant.force_skill_prompts else "0",
        "C2HLS_SKILL_PROMPT_MODE": variant.skill_prompt_mode,
    }
    if variant.bottleneck_positive is not None:
        snap["C2HLS_BOTTLENECK_POSITIVE_SKILLS"] = str(variant.bottleneck_positive)
    if variant.bottleneck_avoid is not None:
        snap["C2HLS_BOTTLENECK_AVOID_SKILLS"] = str(variant.bottleneck_avoid)
    if variant.skill_prompt_mode == "all_skills_avoids_global":
        snap["skill_injection"] = (
            "global: all positive + all avoid skills "
            "(90-skill base + flash_no_RMW_m_axi_skill_entries.json overlay)"
        )
    elif variant.skill_prompt_mode == "all_skills_no_avoids_global":
        snap["skill_injection"] = (
            "global: all positive skills only, no avoid tier "
            "(73-skill base + flash_no_RMW_m_axi_skill_entries.json overlay)"
        )
    elif variant.force_skill_prompts:
        snap["skill_injection"] = (
            f"bottleneck: top-{variant.bottleneck_positive} positive + "
            f"top-{variant.bottleneck_avoid} avoid for top bottleneck kind"
        )
    else:
        snap["skill_injection"] = "none (skills not injected into flash prompt)"
    return snap
