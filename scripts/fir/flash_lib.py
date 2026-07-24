"""Fir flash helpers — open-weight vLLM + scratch Vitis (no PC2 / no team API)."""

from __future__ import annotations

import os
from typing import Any

from c2hls_paths import apply_runtime_defaults, site_artifacts_dir
from c2hls_temp import configure_temp_env

SETUP_TAG = "flash__all_skills_avoids_global"
SETUP_TAG_90_BASE = "flash__90skills_cosim"
SETUP_TAG_90_OVERLAY = "flash__90skills_no_rmw_overlay_cosim"
SKILL_PROMPT_MODE = "all_skills_avoids_global"
STAMP_ENV = "C2HLS_FIR_FLASH_SMOKE_STAMP"
OUT_ENV = "C2HLS_FIR_FLASH_SMOKE_OUT"
ARTIFACT_PREFIX = "flash_smoke"
ARTIFACT_PREFIX_COSIM = "flash_cosim"
DEFAULT_SMOKE_BENCHES = ("hlsfactory_gemm",)


def setup_tag_for_overlay(*, overlay: bool) -> str:
    return SETUP_TAG_90_OVERLAY if overlay else SETUP_TAG_90_BASE


def _fir_flash_cosim_requested() -> bool:
    raw = os.getenv("C2HLS_FIR_FLASH_COSIM", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def configure_fir_flash_env(
    *,
    cosim: bool | None = None,
    overlay: bool = True,
) -> str:
    """Flash env for Fir: 90-skill library + optional overlay + optional RTL cosim."""
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[2]
    SKILLS_90_JSON = (
        REPO
        / "hls_full_optimization_skills_schema_1_1_package"
        / "skills_ii_target_miss_solutions_added(90skills).json"
    )

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = SKILL_PROMPT_MODE
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(SKILLS_90_JSON.resolve())
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    from flash_shared.new_skills_lib import _apply_flash_skill_entries_env

    if overlay:
        _apply_flash_skill_entries_env(True)
    else:
        _apply_flash_skill_entries_env(False)
        os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    run_cosim = cosim if cosim is not None else _fir_flash_cosim_requested()
    if run_cosim:
        # Override fir.env / FIR_DEFAULTS pilot flags (setdefault would keep cosim off).
        os.environ["C2HLS_RUN_COSIM"] = "1"
        os.environ["C2HLS_COSIM_REQUIRED"] = "1"
        os.environ["C2HLS_REFERENCE_COSIM"] = "1"
        os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
        os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "57600")
        os.environ["C2HLS_DISABLE_REPAIR"] = "0"
        os.environ["C2HLS_DISABLE_CORRECTNESS_REPAIR"] = "0"
        os.environ.setdefault("C2HLS_QUALITY_REPAIR_TURNS", "2")
        os.environ.setdefault("C2HLS_TURNS", "4")
    else:
        os.environ.setdefault("C2HLS_RUN_COSIM", "0")
        os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
        os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
        os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "180")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    return setup_tag_for_overlay(overlay=overlay)


def env_snapshot() -> dict[str, Any]:
    return {
        "site": "fir",
        "corpus": "benchmarks",
        "skill_prompt_mode": SKILL_PROMPT_MODE,
        "model": os.getenv("C2HLS_MODEL", ""),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "vitis_settings": os.getenv("C2HLS_VITIS_SETTINGS", ""),
        "xilinx_sif": os.getenv("XILINX_SIF", os.getenv("C2HLS_XILINX_SIF", "")),
        "use_container": os.getenv("C2HLS_USE_CONTAINER", ""),
        "tmp_root": os.getenv("C2HLS_TMP_ROOT", ""),
        "run_cosim": os.getenv("C2HLS_RUN_COSIM", "0"),
        "cosim_required": os.getenv("C2HLS_COSIM_REQUIRED", "0"),
        "disable_repair": os.getenv("C2HLS_DISABLE_REPAIR", "0"),
        "disable_correctness_repair": os.getenv("C2HLS_DISABLE_CORRECTNESS_REPAIR", "0"),
        "quality_repair_turns": os.getenv("C2HLS_QUALITY_REPAIR_TURNS", "2"),
        "turns": os.getenv("C2HLS_TURNS", "4"),
        "packaged_skills_json": os.getenv("C2HLS_PACKAGED_SKILLS_JSON", ""),
        "flash_skill_entries_json": os.getenv("C2HLS_FLASH_SKILL_ENTRIES_JSON", ""),
        "flash_overlay": bool(os.getenv("C2HLS_FLASH_SKILL_ENTRIES_JSON", "").strip()),
    }
