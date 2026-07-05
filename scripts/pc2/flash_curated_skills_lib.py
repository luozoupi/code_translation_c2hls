"""PC2 flash matrix — LLM-curated skills from packaged 73-skill library.

Separate from legacy and deterministic new-matrix runs. Each variant may run
with curation focus bottleneck | warnings | combined (one focus per wave).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]

NEW_SKILLS_JSON_BASE = (
    REPO
    / "hls_full_optimization_skills_schema_1_1_package"
    / "skills_ii_target_miss_solutions_added(73skills).json"
)
NEW_SKILLS_JSON = NEW_SKILLS_JSON_BASE

CURATION_FOCUS_VALUES = ("bottleneck", "warnings", "combined")


@dataclass(frozen=True)
class FlashCuratedVariant:
    key: str
    label: str
    session_id: str
    artifact_prefix: str
    setup_tag: str
    stamp_env: str
    out_env: str
    curation_enabled: bool
    curation_sector: str  # json_only | json_plus_llm | n/a
    include_avoids: bool
    skills_in_prompt: bool

    @property
    def matrix_family(self) -> str:
        return "flash_llm_curated_skills"

    def artifact_dir_name(self, focus: str, stamp: str) -> str:
        return f"{self.artifact_prefix}_{focus}_{stamp}"


VARIANTS: dict[str, FlashCuratedVariant] = {
    "noskills": FlashCuratedVariant(
        key="noskills",
        label="Curated_noskills",
        session_id="flash_curated_noskills",
        artifact_prefix="flash_curated_noskills",
        setup_tag="flash__curated_noskills",
        stamp_env="C2HLS_FLASH_CURATED_NOSKILLS_STAMP",
        out_env="C2HLS_FLASH_CURATED_NOSKILLS_OUT",
        curation_enabled=False,
        curation_sector="n/a",
        include_avoids=False,
        skills_in_prompt=False,
    ),
    "all_avoids_json": FlashCuratedVariant(
        key="all_avoids_json",
        label="Curated_all_avoids_json_only",
        session_id="flash_curated_all_avoids_json",
        artifact_prefix="flash_curated_all_avoids_json",
        setup_tag="flash__curated_all_avoids_json",
        stamp_env="C2HLS_FLASH_CURATED_ALL_AVOIDS_JSON_STAMP",
        out_env="C2HLS_FLASH_CURATED_ALL_AVOIDS_JSON_OUT",
        curation_enabled=True,
        curation_sector="json_only",
        include_avoids=True,
        skills_in_prompt=True,
    ),
    "all_avoids_llm": FlashCuratedVariant(
        key="all_avoids_llm",
        label="Curated_all_avoids_json_plus_llm",
        session_id="flash_curated_all_avoids_llm",
        artifact_prefix="flash_curated_all_avoids_llm",
        setup_tag="flash__curated_all_avoids_llm",
        stamp_env="C2HLS_FLASH_CURATED_ALL_AVOIDS_LLM_STAMP",
        out_env="C2HLS_FLASH_CURATED_ALL_AVOIDS_LLM_OUT",
        curation_enabled=True,
        curation_sector="json_plus_llm",
        include_avoids=True,
        skills_in_prompt=True,
    ),
    "no_avoids_json": FlashCuratedVariant(
        key="no_avoids_json",
        label="Curated_no_avoids_json_only",
        session_id="flash_curated_no_avoids_json",
        artifact_prefix="flash_curated_no_avoids_json",
        setup_tag="flash__curated_no_avoids_json",
        stamp_env="C2HLS_FLASH_CURATED_NO_AVOIDS_JSON_STAMP",
        out_env="C2HLS_FLASH_CURATED_NO_AVOIDS_JSON_OUT",
        curation_enabled=True,
        curation_sector="json_only",
        include_avoids=False,
        skills_in_prompt=True,
    ),
    "no_avoids_llm": FlashCuratedVariant(
        key="no_avoids_llm",
        label="Curated_no_avoids_json_plus_llm",
        session_id="flash_curated_no_avoids_llm",
        artifact_prefix="flash_curated_no_avoids_llm",
        setup_tag="flash__curated_no_avoids_llm",
        stamp_env="C2HLS_FLASH_CURATED_NO_AVOIDS_LLM_STAMP",
        out_env="C2HLS_FLASH_CURATED_NO_AVOIDS_LLM_OUT",
        curation_enabled=True,
        curation_sector="json_plus_llm",
        include_avoids=False,
        skills_in_prompt=True,
    ),
}

VARIANT_ORDER = [
    "noskills",
    "all_avoids_json",
    "all_avoids_llm",
    "no_avoids_json",
    "no_avoids_llm",
]

_COMMON_FLASH_ENV = {
    "C2HLS_PHASEB_MODE": "functional",
    "C2HLS_PHASE8_BASELINE_ALIGN": "0",
    "C2HLS_PHASE5_GT_PREPOP": "0",
    "C2HLS_HW_EMU_FINAL": "0",
    "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS": "1",
    "C2HLS_RUN_COSIM": "0",
    "C2HLS_COSIM_REQUIRED": "0",
    "C2HLS_REFERENCE_COSIM": "0",
    "C2HLS_COSIM_TRACE_LEVEL": "none",
    "C2HLS_SYNTH_TIMEOUT": "1200",
    "C2HLS_CSIM_TIMEOUT": "180",
    "C2HLS_COSIM_TIMEOUT": "1200",
    "C2HLS_LLM_TIMEOUT": "900",
    "OPENAI_API_KEY": "EMPTY",
}


def configure_curated_env(variant: FlashCuratedVariant, *, focus: str) -> None:
    """Apply env for one curated-matrix variant and curation focus."""
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    from flash_shared.new_skills_lib import _apply_flash_skill_entries_env

    if focus not in CURATION_FOCUS_VALUES:
        raise ValueError(f"unknown curation focus: {focus}")

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"

    if variant.curation_enabled:
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(NEW_SKILLS_JSON)
        os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    else:
        os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
        os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)

    _apply_flash_skill_entries_env(variant.curation_enabled)

    os.environ.pop("C2HLS_BOTTLENECK_POSITIVE_SKILLS", None)
    os.environ.pop("C2HLS_BOTTLENECK_AVOID_SKILLS", None)

    if variant.curation_enabled:
        os.environ["C2HLS_SKILL_MODE"] = "skill_on"
        os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = "llm_curated"
        os.environ["C2HLS_SKILL_CURATION_ENABLED"] = "1"
        os.environ["C2HLS_SKILL_CURATION_FOCUS"] = focus
        os.environ["C2HLS_SKILL_CURATION_SECTOR"] = variant.curation_sector
        os.environ["C2HLS_SKILL_CURATION_INCLUDE_AVOIDS"] = (
            "1" if variant.include_avoids else "0"
        )
    else:
        os.environ["C2HLS_SKILL_MODE"] = "skill_off"
        os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
        os.environ.pop("C2HLS_SKILL_PROMPT_MODE", None)
        os.environ["C2HLS_SKILL_CURATION_ENABLED"] = "0"
        os.environ.pop("C2HLS_SKILL_CURATION_FOCUS", None)
        os.environ.pop("C2HLS_SKILL_CURATION_SECTOR", None)
        os.environ.pop("C2HLS_SKILL_CURATION_INCLUDE_AVOIDS", None)

    for key, value in _COMMON_FLASH_ENV.items():
        os.environ.setdefault(key, value)


def variant_env_snapshot(variant: FlashCuratedVariant, *, focus: str) -> dict[str, str]:
    skills_path = NEW_SKILLS_JSON if variant.curation_enabled else None
    snap = {
        "matrix_family": variant.matrix_family,
        "skills_json": str(skills_path) if skills_path is not None else "",
        "skills_json_mode": (
            "packaged_base_plus_flash_overlay"
            if skills_path is not None
            else "none"
        ),
        "curation_focus": focus,
        "C2HLS_SKILL_CURATION_ENABLED": "1" if variant.curation_enabled else "0",
    }
    if variant.curation_enabled:
        snap["C2HLS_SKILL_PROMPT_MODE"] = "llm_curated"
        snap["C2HLS_SKILL_CURATION_SECTOR"] = variant.curation_sector
        snap["C2HLS_SKILL_CURATION_INCLUDE_AVOIDS"] = (
            "1" if variant.include_avoids else "0"
        )
        snap["skill_injection"] = (
            f"llm_curated: sector={variant.curation_sector} "
            f"focus={focus} avoids={variant.include_avoids}"
        )
    else:
        snap["skill_injection"] = "none (noskills control; curation skipped)"
    return snap
