"""Tier B MachSuite flash helpers (90-skill aav_n library, csim+csynth+cosim)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tier_a_flash_lib import resolve_packaged_skills_json, verify_skills_90
from tier_b_gold_lib import TIER_B_READY_ROOT, resolve_tier_b_benches

REPO = Path(__file__).resolve().parents[2]

SETUP_TAG = "flash__tier_b_machsuite__aav_n"
MATRIX_FAMILY = "flash_tier_b_machsuite_aav_n"
DEFAULT_SYNTH_TIMEOUT_S = 3600
DEFAULT_CSIM_TIMEOUT_S = 600
DEFAULT_COSIM_TIMEOUT_S = 43200


def configure_tier_b_flash_aav_n_env() -> None:
    import sys

    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    scripts_root = Path(__file__).resolve().parents[1]
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    skills = verify_skills_90()
    if not skills.get("ok"):
        raise RuntimeError(f"90-skill library invalid: {skills.get('errors')}")

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(resolve_packaged_skills_json())
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    from flash_shared.new_skills_lib import _apply_flash_skill_entries_env

    _apply_flash_skill_entries_env(True)
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    # Cosim on for generated + gold reference gates.
    os.environ["C2HLS_RUN_COSIM"] = "1"
    os.environ["C2HLS_COSIM_REQUIRED"] = "0"
    os.environ["C2HLS_REFERENCE_COSIM"] = "1"
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", str(DEFAULT_CSIM_TIMEOUT_S))
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", str(DEFAULT_COSIM_TIMEOUT_S))
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def env_snapshot() -> dict[str, Any]:
    return {
        "matrix_family": MATRIX_FAMILY,
        "corpus": "tier_B_ready",
        "benchmarks_root": str(TIER_B_READY_ROOT.resolve()),
        "skill_prompt_mode": "all_skills_avoids_global",
        "skills_json": str(resolve_packaged_skills_json()),
        "setup_tag": SETUP_TAG,
        "run_cosim": True,
        "reference_cosim": True,
        "timeouts": {
            "synth_s": int(os.getenv("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))),
            "csim_s": int(os.getenv("C2HLS_CSIM_TIMEOUT", str(DEFAULT_CSIM_TIMEOUT_S))),
            "cosim_s": int(os.getenv("C2HLS_COSIM_TIMEOUT", str(DEFAULT_COSIM_TIMEOUT_S))),
        },
    }


def resolve_machsuite_benches(requested: list[str]) -> list[tuple[str, Path]]:
    return resolve_tier_b_benches(requested, root=TIER_B_READY_ROOT)
