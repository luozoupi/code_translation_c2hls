"""Commercial LLM API flash batch — mirrors PC2 flash tests on the team server.

Artifacts: ``artifacts/flash_api/<artifact_prefix>_<stamp>/``

No PC2 cluster, ``local.env``, vLLM, or ``--pc2`` required.  Uses team path
defaults from ``c2hls_paths.TEAM_DEFAULTS`` (same as ``run_agentic_sweep.py``).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
_SHARED = REPO / "scripts" / "flash_shared"
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from flash_shared.new_skills_lib import (  # noqa: E402
    VARIANTS as NEW_VARIANTS,
    configure_new_skills_env,
    skills_json_for_variant,
    variant_env_snapshot,
)
from flash_shared.team_env import (  # noqa: E402
    FLASH_PILOT_ENV,
    active_team_paths_summary,
    bootstrap_team_flash_env,
    finalize_api_llm_env,
    preflight_api_run,
)

DEFAULT_API_MODEL = "claude-sonnet-4-6"

_NEW_KEY_MAP = {
    "noskills_new": "noskills_new",
    "bn22_new": "bn_skills_new_2_2",
    "bn42_new": "bn_skills_new_4_2",
    "bn62_new": "bn_skills_new_6_2",
    "nav_n": "all_new_skills_no_avoids_global",
    "aav_n": "all_new_skills_avoids_global",
}


@dataclass(frozen=True)
class FlashApiProfile:
    key: str
    label: str
    short_code: str
    artifact_prefix: str
    setup_tag: str
    pc2_mirror: str
    family: str
    stamp_env: str
    out_env: str
    configure: str


def model_cell_tag(model_id: str) -> str:
    low = (model_id or "").lower()
    if "devstral" in low:
        return "devstral2"
    if "sonnet" in low:
        return "sonnet"
    if "haiku" in low:
        return "haiku"
    if "opus" in low:
        return "opus"
    if "gpt" in low or low.startswith("o1") or low.startswith("o3") or low.startswith("o4"):
        return "gpt"
    if "nemotron" in low:
        return "nemotron"
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return slug[:48] or "model"


def resolve_model_id(cli_model: str = "") -> str:
    return (
        (cli_model or "").strip()
        or os.getenv("C2HLS_MODEL", "").strip()
        or os.getenv("C2HLS_API_MODEL", "").strip()
        or DEFAULT_API_MODEL
    )


def artifact_root(profile: FlashApiProfile, stamp: str) -> Path:
    from c2hls_paths import FLASH_API_ARTIFACTS_DIR

    override = os.getenv(profile.out_env, "").strip()
    if override:
        return Path(override)
    return FLASH_API_ARTIFACTS_DIR / f"{profile.artifact_prefix}_{stamp}"


def _apply_flash_strategy_env() -> None:
    bootstrap_team_flash_env()
    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"


def configure_noskills_old() -> None:
    _apply_flash_strategy_env()
    os.environ["C2HLS_SKILL_MODE"] = "skill_off"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
    os.environ.pop("C2HLS_SKILL_PROMPT_MODE", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)


def configure_bn22_old() -> None:
    _apply_flash_strategy_env()
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = "bottleneck"
    os.environ.setdefault("C2HLS_BOTTLENECK_POSITIVE_SKILLS", "2")
    os.environ.setdefault("C2HLS_BOTTLENECK_AVOID_SKILLS", "2")
    os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)


def configure_nav_o() -> None:
    _apply_flash_strategy_env()
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_no_avoids_global"
    os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)


def configure_aav_o() -> None:
    _apply_flash_strategy_env()
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
    os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)


def _configure_new_variant(profile_key: str) -> None:
    variant_key = _NEW_KEY_MAP[profile_key]
    bootstrap_team_flash_env()
    configure_new_skills_env(NEW_VARIANTS[variant_key], inference="api")


_CONFIGURE: dict[str, Callable[[], None]] = {
    "configure_noskills_old": configure_noskills_old,
    "configure_bn22_old": configure_bn22_old,
    "configure_nav_o": configure_nav_o,
    "configure_aav_o": configure_aav_o,
    "configure_noskills_new": lambda: _configure_new_variant("noskills_new"),
    "configure_bn22_new": lambda: _configure_new_variant("bn22_new"),
    "configure_bn42_new": lambda: _configure_new_variant("bn42_new"),
    "configure_bn62_new": lambda: _configure_new_variant("bn62_new"),
    "configure_nav_n": lambda: _configure_new_variant("nav_n"),
    "configure_aav_n": lambda: _configure_new_variant("aav_n"),
}


PROFILES: dict[str, FlashApiProfile] = {
    "noskills_old": FlashApiProfile(
        key="noskills_old", label="Noskills (old)", short_code="nosk_o",
        artifact_prefix="flash_noskills", setup_tag="flash__noskills", pc2_mirror="nosk_o",
        family="legacy", stamp_env="C2HLS_FLASH_API_NOSKILLS_OLD_STAMP",
        out_env="C2HLS_FLASH_API_NOSKILLS_OLD_OUT", configure="configure_noskills_old",
    ),
    "bn22_old": FlashApiProfile(
        key="bn22_old", label="Bn 2+2 (old)", short_code="bn22_o",
        artifact_prefix="flash_skills", setup_tag="flash__skills", pc2_mirror="bn22_o",
        family="legacy", stamp_env="C2HLS_FLASH_API_BN22_OLD_STAMP",
        out_env="C2HLS_FLASH_API_BN22_OLD_OUT", configure="configure_bn22_old",
    ),
    "nav_o": FlashApiProfile(
        key="nav_o", label="No avoids (old)", short_code="nav_o",
        artifact_prefix="flash_all_skills_no_avoids_global",
        setup_tag="flash__all_skills_no_avoids_global", pc2_mirror="nav_o",
        family="legacy", stamp_env="C2HLS_FLASH_API_NAV_O_STAMP",
        out_env="C2HLS_FLASH_API_NAV_O_OUT", configure="configure_nav_o",
    ),
    "aav_o": FlashApiProfile(
        key="aav_o", label="All+avoids (old)", short_code="aav_o",
        artifact_prefix="flash_all_skills_avoids_global",
        setup_tag="flash__all_skills_avoids_global", pc2_mirror="aav_o",
        family="legacy", stamp_env="C2HLS_FLASH_API_AAV_O_STAMP",
        out_env="C2HLS_FLASH_API_AAV_O_OUT", configure="configure_aav_o",
    ),
    "noskills_new": FlashApiProfile(
        key="noskills_new", label="Noskills (new)", short_code="nosk_n",
        artifact_prefix="flash_noskills_new", setup_tag="flash__noskills_new", pc2_mirror="nosk_n",
        family="new", stamp_env="C2HLS_FLASH_API_NOSKILLS_NEW_STAMP",
        out_env="C2HLS_FLASH_API_NOSKILLS_NEW_OUT", configure="configure_noskills_new",
    ),
    "bn22_new": FlashApiProfile(
        key="bn22_new", label="Bn 2+2 (new)", short_code="bn22_n",
        artifact_prefix="flash_bn_skills_new_2_2", setup_tag="flash__bn_skills_new_2_2",
        pc2_mirror="bn22_n", family="new", stamp_env="C2HLS_FLASH_API_BN22_NEW_STAMP",
        out_env="C2HLS_FLASH_API_BN22_NEW_OUT", configure="configure_bn22_new",
    ),
    "bn42_new": FlashApiProfile(
        key="bn42_new", label="Bn 4+2 (new)", short_code="bn42_n",
        artifact_prefix="flash_bn_skills_new_4_2", setup_tag="flash__bn_skills_new_4_2",
        pc2_mirror="bn42_n", family="new", stamp_env="C2HLS_FLASH_API_BN42_NEW_STAMP",
        out_env="C2HLS_FLASH_API_BN42_NEW_OUT", configure="configure_bn42_new",
    ),
    "bn62_new": FlashApiProfile(
        key="bn62_new", label="Bn 6+2 (new)", short_code="bn62_n",
        artifact_prefix="flash_bn_skills_new_6_2", setup_tag="flash__bn_skills_new_6_2",
        pc2_mirror="bn62_n", family="new", stamp_env="C2HLS_FLASH_API_BN62_NEW_STAMP",
        out_env="C2HLS_FLASH_API_BN62_NEW_OUT", configure="configure_bn62_new",
    ),
    "nav_n": FlashApiProfile(
        key="nav_n", label="No avoids (new)", short_code="nav_n",
        artifact_prefix="flash_all_new_skills_no_avoids_global",
        setup_tag="flash__all_new_skills_no_avoids_global", pc2_mirror="nav_n",
        family="new", stamp_env="C2HLS_FLASH_API_NAV_N_STAMP",
        out_env="C2HLS_FLASH_API_NAV_N_OUT", configure="configure_nav_n",
    ),
    "aav_n": FlashApiProfile(
        key="aav_n", label="All+avoids (new)", short_code="aav_n",
        artifact_prefix="flash_all_new_skills_avoids_global",
        setup_tag="flash__all_new_skills_avoids_global", pc2_mirror="aav_n",
        family="new", stamp_env="C2HLS_FLASH_API_AAV_N_STAMP",
        out_env="C2HLS_FLASH_API_AAV_N_OUT", configure="configure_aav_n",
    ),
}

DETERMINISTIC_ORDER = [
    "noskills_old", "bn22_old", "nav_o", "aav_o",
    "noskills_new", "bn22_new", "bn42_new", "bn62_new", "aav_n", "nav_n",
]
TOP5_ORDER = ["nav_o", "aav_n", "nav_n", "noskills_old", "aav_o"]


def apply_profile_env(profile: FlashApiProfile) -> None:
    fn = _CONFIGURE.get(profile.configure)
    if fn is None:
        raise RuntimeError(f"no configure hook for profile {profile.key}")
    fn()
    finalize_api_llm_env()


def profile_skills_snapshot(profile: FlashApiProfile) -> dict[str, str]:
    if profile.family == "new":
        variant = NEW_VARIANTS[_NEW_KEY_MAP[profile.key]]
        return {k: str(v) for k, v in variant_env_snapshot(variant).items()}
    pkg = REPO / "hls_full_optimization_skills_schema_1_1_package" / "skills.json"
    if profile.key == "nav_o":
        return {
            "skills_json": str(pkg),
            "skill_injection": "global positive only (skills.json, 41 injected after U280 filter)",
        }
    if profile.key == "aav_o":
        return {
            "skills_json": str(pkg),
            "skill_injection": "global positive + avoid (skills.json, 55 injected)",
        }
    if profile.key == "bn22_old":
        return {"skills_json": str(pkg), "skill_injection": "bottleneck 2+2 (skills.json)"}
    return {"skill_injection": "none"}
