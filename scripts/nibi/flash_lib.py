"""Nibi flash helpers — open-weight vLLM + scratch Vitis (SHARCNET Nibi)."""

from __future__ import annotations

import os
from typing import Any

from c2hls_paths import apply_runtime_defaults, site_artifacts_dir
from c2hls_temp import configure_temp_env

SETUP_TAG = "flash__all_skills_avoids_global"
SKILL_PROMPT_MODE = "all_skills_avoids_global"
STAMP_ENV = "C2HLS_NIBI_FLASH_SMOKE_STAMP"
OUT_ENV = "C2HLS_NIBI_FLASH_SMOKE_OUT"
ARTIFACT_PREFIX = "flash_smoke"
DEFAULT_SMOKE_BENCHES = ("hlsfactory_gemm",)


def configure_nibi_flash_env(*, cosim: bool | None = None) -> str:
    """Flash env for Nibi: 90-skill library + optional overlay + optional RTL cosim."""
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

    _apply_flash_skill_entries_env(True)
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    run_cosim = bool(cosim)
    if run_cosim:
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
    return SETUP_TAG


def env_snapshot() -> dict[str, Any]:
    return {
        "site": "nibi",
        "corpus": "benchmarks",
        "skill_prompt_mode": SKILL_PROMPT_MODE,
        "model": os.getenv("C2HLS_MODEL", ""),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "vitis_settings": os.getenv("C2HLS_VITIS_SETTINGS", ""),
        "xilinx_sif": os.getenv("XILINX_SIF", os.getenv("C2HLS_XILINX_SIF", "")),
        "use_container": os.getenv("C2HLS_USE_CONTAINER", ""),
        "tmp_root": os.getenv("C2HLS_TMP_ROOT", ""),
        "glm_model_path": os.getenv("GLM_MODEL_PATH", ""),
        "vllm_model_path": os.getenv("VLLM_MODEL_PATH", ""),
        "run_cosim": os.getenv("C2HLS_RUN_COSIM", "0"),
        "cosim_required": os.getenv("C2HLS_COSIM_REQUIRED", "0"),
        "packaged_skills_json": os.getenv("C2HLS_PACKAGED_SKILLS_JSON", ""),
        "flash_skill_entries_json": os.getenv("C2HLS_FLASH_SKILL_ENTRIES_JSON", ""),
    }


def preflight_blockers() -> list[str]:
    """Return human-readable blockers before spending GPU time."""
    from pathlib import Path

    errors: list[str] = []
    use_container = os.getenv("C2HLS_USE_CONTAINER", "1").strip().lower() not in {"0", "false", "no", "off"}
    if use_container:
        sif = (
            os.getenv("C2HLS_XILINX_SIF", "").strip()
            or os.getenv("XILINX_SIF", "").strip()
        )
        if not sif:
            errors.append("C2HLS_XILINX_SIF is unset (standalone Apptainer SIF path).")
        elif not Path(sif).is_file():
            errors.append(
                f"Standalone Vitis SIF missing: {sif}\n"
                "  Copy xilinx_vitis_2023.2.standalone.sif under scratch/containers/."
            )
    else:
        vitis = os.getenv("C2HLS_VITIS_SETTINGS", "").strip()
        if not vitis:
            errors.append("C2HLS_VITIS_SETTINGS is unset.")
        elif not Path(vitis).is_file():
            errors.append(
                f"C2HLS_VITIS_SETTINGS missing: {vitis}\n"
                "  Install Vitis 2023.2 under scratch (see fir-vitis-u280-setup.md)."
            )
    weights = os.getenv("VLLM_MODEL_PATH", "").strip()
    if not weights:
        errors.append("VLLM_MODEL_PATH is unset.")
    elif not Path(weights).is_dir():
        errors.append(f"LLM weights directory missing: {weights}")
    py = os.getenv("C2HLS_PYTHON", "python3").strip()
    if py and not Path(py).is_file():
        errors.append(f"C2HLS_PYTHON missing: {py} (run scripts/nibi/setup_compute_env.sh)")
    return errors
