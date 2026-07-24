"""Fir 0-shot flash — minimal LLM prompt, no skill library."""

from __future__ import annotations

import os
from dataclasses import dataclass

from c2hls_paths import apply_runtime_defaults
from c2hls_temp import configure_temp_env


@dataclass(frozen=True)
class FirZeroShotVariant:
    key: str
    label: str
    setup_tag: str
    skip_phase_b: bool


VARIANTS: dict[str, FirZeroShotVariant] = {
    "phaseb": FirZeroShotVariant(
        key="phaseb",
        label="absolute 0-shot flash after Phase B (no repair)",
        setup_tag="flash__abs_zero_shot_cosim__phaseb",
        skip_phase_b=False,
    ),
    "direct": FirZeroShotVariant(
        key="direct",
        label="absolute 0-shot flash from plain C (no Phase B, no repair)",
        setup_tag="flash__abs_zero_shot_cosim__direct",
        skip_phase_b=True,
    ),
}


def configure_fir_zero_shot_env(variant: FirZeroShotVariant) -> None:
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
    # Override Fir site default (C2HLS_RUN_COSIM=0 from configure_site).
    os.environ["C2HLS_RUN_COSIM"] = "1"
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ["C2HLS_DISABLE_REPAIR"] = "1"
    os.environ["C2HLS_DISABLE_CORRECTNESS_REPAIR"] = "1"
    os.environ["C2HLS_QUALITY_REPAIR_TURNS"] = "0"
    os.environ["C2HLS_SYNTH_TIMEOUT"] = "14400"
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "57600")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    # Keep Vitis csynth/cosim parallelism at 16 even when Slurm allocates more CPUs.
    os.environ.setdefault("C2HLS_VITIS_JOBS", "16")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
