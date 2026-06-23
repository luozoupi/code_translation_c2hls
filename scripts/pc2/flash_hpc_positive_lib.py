"""PC2 flash matrix using ``skills_flash_hpc_positive_v1.json`` only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]

def _resolve_skills_json() -> Path:
    override = os.getenv("C2HLS_HPC_POSITIVE_SKILLS_JSON", "").strip()
    if override:
        return Path(override)
    version = os.getenv("C2HLS_HPC_POSITIVE_SKILLS_VERSION", "v2").strip().lstrip("v")
    return (
        REPO
        / "hls_full_optimization_skills_schema_1_1_package"
        / f"skills_flash_hpc_positive_v{version}.json"
    )


HPC_POSITIVE_SKILLS_JSON = _resolve_skills_json()


@dataclass(frozen=True)
class FlashHpcPositiveVariant:
    key: str
    label: str
    session_id: str
    artifact_prefix: str
    setup_tag: str
    stamp_env: str
    out_env: str
    force_skill_prompts: bool
    skill_prompt_mode: str  # bottleneck | all_skills_no_avoids_global
    bottleneck_positive: Optional[int] = None
    bottleneck_avoid: Optional[int] = None

    @property
    def matrix_family(self) -> str:
        ver = os.getenv("C2HLS_HPC_POSITIVE_SKILLS_VERSION", "v2").strip().lstrip("v")
        return f"flash_hpc_positive_v{ver}"


VARIANTS: dict[str, FlashHpcPositiveVariant] = {
    "noskills": FlashHpcPositiveVariant(
        key="noskills",
        label="Noskills (hpc v1 matrix)",
        session_id="flash_hpc_positive_v2_noskills",
        artifact_prefix="flash_hpc_positive_v2_noskills",
        setup_tag="flash__hpc_positive_v2__noskills",
        stamp_env="C2HLS_FLASH_HPC_POSITIVE_V1_STAMP",
        out_env="C2HLS_FLASH_HPC_POSITIVE_V1_NOSKILLS_OUT",
        force_skill_prompts=False,
        skill_prompt_mode="bottleneck",
    ),
    "all_skills": FlashHpcPositiveVariant(
        key="all_skills",
        label="All skills (hpc positive v1, global)",
        session_id="flash_hpc_positive_v2_all_skills",
        artifact_prefix="flash_hpc_positive_v2_all_skills",
        setup_tag="flash__hpc_positive_v2__all_skills",
        stamp_env="C2HLS_FLASH_HPC_POSITIVE_V1_STAMP",
        out_env="C2HLS_FLASH_HPC_POSITIVE_V1_ALL_SKILLS_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="all_skills_no_avoids_global",
    ),
    "bn_4_2": FlashHpcPositiveVariant(
        key="bn_4_2",
        label="Bn 4+2 (hpc positive v1, bottleneck)",
        session_id="flash_hpc_positive_v2_bn_4_2",
        artifact_prefix="flash_hpc_positive_v2_bn_4_2",
        setup_tag="flash__hpc_positive_v2__bn_4_2",
        stamp_env="C2HLS_FLASH_HPC_POSITIVE_V1_STAMP",
        out_env="C2HLS_FLASH_HPC_POSITIVE_V1_BN_4_2_OUT",
        force_skill_prompts=True,
        skill_prompt_mode="bottleneck",
        bottleneck_positive=4,
        bottleneck_avoid=2,
    ),
}

VARIANT_ORDER = ["noskills", "all_skills", "bn_4_2"]

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


def configure_hpc_positive_env(variant: FlashHpcPositiveVariant) -> None:
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(HPC_POSITIVE_SKILLS_JSON)
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"

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

    for key, value in _COMMON_FLASH_ENV.items():
        os.environ.setdefault(key, value)


def variant_env_snapshot(variant: FlashHpcPositiveVariant) -> dict[str, str]:
    snap = {
        "matrix_family": variant.matrix_family,
        "skills_json": str(HPC_POSITIVE_SKILLS_JSON),
        "skills_json_mode": "packaged_only",
        "C2HLS_FORCE_SKILL_PROMPTS": "1" if variant.force_skill_prompts else "0",
        "C2HLS_SKILL_PROMPT_MODE": variant.skill_prompt_mode,
    }
    if variant.bottleneck_positive is not None:
        snap["C2HLS_BOTTLENECK_POSITIVE_SKILLS"] = str(variant.bottleneck_positive)
    if variant.bottleneck_avoid is not None:
        snap["C2HLS_BOTTLENECK_AVOID_SKILLS"] = str(variant.bottleneck_avoid)
    if variant.skill_prompt_mode == "all_skills_no_avoids_global":
        snap["skill_injection"] = "global: all hpc_positive_v1 skills (30 positive-only)"
    elif variant.force_skill_prompts:
        snap["skill_injection"] = (
            f"bottleneck: top-{variant.bottleneck_positive} positive + "
            f"top-{variant.bottleneck_avoid} avoid (library has no avoid tier — avoid picks empty)"
        )
    else:
        snap["skill_injection"] = "none"
    return snap
