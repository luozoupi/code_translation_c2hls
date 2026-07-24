"""0-shot flash on ``benchmarks_cosim/`` — minimal LLM prompt, no skill library.

Two arms (separate artifact roots):
  phaseb  — functional Phase B baseline, then zero-shot flash on HLS
  direct  — skip Phase B; zero-shot translate+optimize from plain.cpp
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from flash_fixed_cosim_lib import (
    BENCHMARKS_COSIM_DIR,
    MATRIX_FAMILY as _BASE_MATRIX_FAMILY,
    list_cosim_benches,
    resolve_cosim_benches,
)

REPO = Path(__file__).resolve().parents[2]

InferenceKind = Literal["vllm"]
MATRIX_FAMILY = "flash_zero_shot_cosim"


@dataclass(frozen=True)
class ZeroShotCosimVariant:
    key: str
    label: str
    session_id: str
    artifact_prefix: str
    setup_tag: str
    skip_phase_b: bool

    @property
    def stamp_env(self) -> str:
        return f"C2HLS_ZERO_SHOT_{self.key.upper()}_STAMP"

    @property
    def out_env(self) -> str:
        return f"C2HLS_ZERO_SHOT_{self.key.upper()}_OUT"


VARIANTS: dict[str, ZeroShotCosimVariant] = {
    "phaseb": ZeroShotCosimVariant(
        key="phaseb",
        label="0-shot flash after Phase B",
        session_id="flash_zero_shot_cosim_phaseb",
        artifact_prefix="flash_zero_shot_cosim_phaseb",
        setup_tag="flash__zero_shot_cosim__phaseb",
        skip_phase_b=False,
    ),
    "direct": ZeroShotCosimVariant(
        key="direct",
        label="0-shot flash from plain C (no Phase B)",
        session_id="flash_zero_shot_cosim_direct",
        artifact_prefix="flash_zero_shot_cosim_direct",
        setup_tag="flash__zero_shot_cosim__direct",
        skip_phase_b=True,
    ),
}

VARIANT_ORDER = ["phaseb", "direct"]


def configure_zero_shot_cosim_env(variant: ZeroShotCosimVariant) -> None:
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env
    from flash_shared.new_skills_lib import _apply_flash_skill_entries_env

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_RECORD_FLOW"] = "1"
    os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] = "zero_shot"
    os.environ["C2HLS_SKILL_MODE"] = "skill_off"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
    os.environ.pop("C2HLS_SKILL_PROMPT_MODE", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
    os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)
    _apply_flash_skill_entries_env(False)

    if variant.skip_phase_b:
        os.environ["C2HLS_SKIP_PHASE_B"] = "1"
    else:
        os.environ.pop("C2HLS_SKIP_PHASE_B", None)
        os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")

    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "7200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "57600")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def variant_env_snapshot(variant: ZeroShotCosimVariant) -> dict:
    return {
        "matrix_family": MATRIX_FAMILY,
        "parent_matrix_family": _BASE_MATRIX_FAMILY,
        "benchmarks_root": str(BENCHMARKS_COSIM_DIR.resolve()),
        "corpus": "benchmarks_cosim",
        "record_flow": True,
        "variant": variant.key,
        "label": variant.label,
        "flash_opt_prompt_mode": "zero_shot",
        "skip_phase_b": variant.skip_phase_b,
        "skills_in_prompt": False,
        "force_skill_prompts": False,
    }


__all__ = [
    "BENCHMARKS_COSIM_DIR",
    "MATRIX_FAMILY",
    "VARIANTS",
    "VARIANT_ORDER",
    "ZeroShotCosimVariant",
    "configure_zero_shot_cosim_env",
    "list_cosim_benches",
    "resolve_cosim_benches",
    "variant_env_snapshot",
]
