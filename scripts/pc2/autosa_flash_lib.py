"""AutoSA autosa_ready corpus helpers for PC2 flash (90 skills or gemm_flatten + overlay).

nav_n: 90 pack, no avoids. aav_n: 90 pack + avoids + no-RMW overlay.
aav_n_gf: gemm_flatten_v1 + avoids + no-RMW overlay.
aav_n_gf v2 pack (opt-in): gemm_flatten_v2 preventive LANES/II=4 skills; default stays v1.
aav_n_90: 90 pack + avoids, no gemm_flatten pack, no no-RMW overlay.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
AUTOSA_READY_ROOT = REPO / "related_work/benchmarks/autosa_ready"
AUTOSA_READY_ROOT_ENV = "C2HLS_AUTOSA_READY_ROOT"


def resolve_autosa_ready_root() -> Path:
    raw = os.getenv(AUTOSA_READY_ROOT_ENV, "").strip()
    if raw:
        return Path(raw)
    return AUTOSA_READY_ROOT


SKILLS_PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
SKILLS_90_JSON = SKILLS_PKG / "skills_ii_target_miss_solutions_added(90skills).json"
SKILLS_90_GEMM_JSON = (
    SKILLS_PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
)
# Derived preventive pack (LANES=16 load + k+=4 named acc banks). Default mmflow stays v1.
SKILLS_90_GEMM_V2_JSON = (
    SKILLS_PKG / "skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v2.json"
)
SKILLS_ONCHIP_JSON = SKILLS_PKG / "flash_onchip_wide_gemm_skill_entries.json"
SKILLS_GENERIC_JSON = SKILLS_PKG / "flash_generic_hls_skill_entries.json"
SKILLS_GEMM_FAMILY_JSON = SKILLS_PKG / "flash_gemm_family_skill_entries.json"
SKILLS_SYSTOLIC_IO_JSON = SKILLS_PKG / "flash_systolic_io_skill_entries.json"

SETUP_TAG = "flash__autosa__nav_n"
SETUP_TAG_AAV_N = "flash__autosa__aav_n"
SETUP_TAG_AAV_N_90 = "flash__autosa__aav_n_90"
SETUP_TAG_AAV_N_GF = "flash__autosa__aav_n_gf"
SETUP_TAG_NOSKILLS = "flash__autosa__noskills"
SETUP_TAG_ZERO_SHOT = "flash__autosa__zero_shot"
SETUP_TAG_ONE_SHOT = "flash__autosa__one_shot"
SETUP_TAG_ONCHIP = "flash__autosa__onchip_gemm"
SETUP_TAG_GENERIC = "flash__autosa__generic_hls"
SETUP_TAG_GEMM_FAMILY = "flash__autosa__gemm_family"
SETUP_TAG_SYSTOLIC_IO = "flash__autosa__systolic_io"
SETUP_TAG_GOLD = "gold_gate"
VARIANT_NAV_N = "autosa_nav_n"
VARIANT_AAV_N = "autosa_aav_n"
VARIANT_AAV_N_90 = "autosa_aav_n_90"
VARIANT_AAV_N_GF = "autosa_aav_n_gf"
VARIANT_NOSKILLS = "autosa_noskills"
VARIANT_ZERO_SHOT = "autosa_zero_shot"
VARIANT_ONE_SHOT = "autosa_one_shot"
VARIANT_ONCHIP = "autosa_onchip_gemm"
VARIANT_GENERIC = "autosa_generic_hls"
VARIANT_GEMM_FAMILY = "autosa_gemm_family"
VARIANT_SYSTOLIC_IO = "autosa_systolic_io"
VARIANT_GOLD = "autosa_gold"
STAMP_ENV = "C2HLS_AUTOSA_FLASH_STAMP"
OUT_ENV = "C2HLS_AUTOSA_FLASH_OUT"
MATRIX_FAMILY = "flash_autosa_ready_nav_n"

_SKILL_ENV_KEYS = (
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_SKILL_PROMPT_ORDER_JSON",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_DSE_SKILL_ENTRIES_JSON",
    "C2HLS_STREAM_SKILL_ENTRIES_JSON",
    "C2HLS_FLASH_ONCHIP",
    "C2HLS_FLASH_SKILL_BIN",
)

_MM_FLOW_FLAVOR_ALIASES = {
    "": "skills",
    "skill": "skills",
    "with_skills": "skills",
    "aav_n_gf": "skills",
    "aav_n_90": "aav_n_90",
    "90only": "aav_n_90",
    "90_only": "aav_n_90",
    "no_overlay": "aav_n_90",
    "no_skills": "noskills",
    "no_skill": "noskills",
    "0shot": "zero_shot",
    "0_shot": "zero_shot",
    "zeroshot": "zero_shot",
    "oneshot": "one_shot",
    "1shot": "one_shot",
    "1_shot": "one_shot",
}


def setup_tag_for_variant(variant_key: str) -> str:
    if variant_key == VARIANT_AAV_N_GF:
        return SETUP_TAG_AAV_N_GF
    if variant_key == VARIANT_AAV_N_90:
        return SETUP_TAG_AAV_N_90
    if variant_key == VARIANT_AAV_N:
        return SETUP_TAG_AAV_N
    if variant_key == VARIANT_NOSKILLS:
        return SETUP_TAG_NOSKILLS
    if variant_key == VARIANT_ZERO_SHOT:
        return SETUP_TAG_ZERO_SHOT
    if variant_key == VARIANT_ONE_SHOT:
        return SETUP_TAG_ONE_SHOT
    if variant_key == VARIANT_ONCHIP:
        return SETUP_TAG_ONCHIP
    if variant_key == VARIANT_GENERIC:
        return SETUP_TAG_GENERIC
    if variant_key == VARIANT_GEMM_FAMILY:
        return SETUP_TAG_GEMM_FAMILY
    if variant_key == VARIANT_SYSTOLIC_IO:
        return SETUP_TAG_SYSTOLIC_IO
    if variant_key == VARIANT_GOLD:
        return SETUP_TAG_GOLD
    return SETUP_TAG

DEFAULT_CSIM_TIMEOUT_S = 1800
DEFAULT_SYNTH_TIMEOUT_S = 14400


def _configure_autosa_runtime_env() -> None:
    """Part, clock, timeouts, strategy=flash. Does not touch skill JSON."""
    import sys

    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = Path(__file__).resolve().parents[1]
    for extra in (str(repo_root), str(scripts_root)):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ["C2HLS_PHASEB_FROM_GOLD"] = "0"
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", str(DEFAULT_CSIM_TIMEOUT_S))
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def _strip_autosa_skill_env() -> None:
    os.environ["C2HLS_SKILL_MODE"] = "skill_off"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
    for key in _SKILL_ENV_KEYS:
        os.environ.pop(key, None)


def _configure_autosa_flash_env(
    *,
    include_avoids: bool,
    skills_json: Path | None = None,
    flash_overlay: bool = True,
) -> None:
    _configure_autosa_runtime_env()
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = (
        "all_skills_avoids_global" if include_avoids else "all_skills_no_avoids_global"
    )
    pack = Path(skills_json) if skills_json is not None else SKILLS_90_JSON
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(pack.resolve())
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    from flash_shared.new_skills_lib import _apply_flash_skill_entries_env

    _apply_flash_skill_entries_env(flash_overlay)


def configure_autosa_flash_nav_n_env() -> None:
    _configure_autosa_flash_env(include_avoids=False)


def configure_autosa_flash_aav_n_env() -> None:
    _configure_autosa_flash_env(include_avoids=True)


def configure_autosa_flash_aav_n_gf_env() -> None:
    _configure_autosa_flash_env(
        include_avoids=True,
        skills_json=SKILLS_90_GEMM_JSON,
    )


def configure_autosa_flash_aav_n_90_env() -> None:
    """Plain 90-skill pack + avoids. No gemm_flatten pack, no no-RMW overlay."""
    _configure_autosa_flash_env(
        include_avoids=True,
        skills_json=SKILLS_90_JSON,
        flash_overlay=False,
    )
    os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)


def configure_autosa_flash_onchip_env() -> None:
    """Distilled 940-class pack only. No 90-skill dump, no no-RMW overlay."""
    _configure_autosa_flash_env(
        include_avoids=True,
        skills_json=SKILLS_ONCHIP_JSON,
        flash_overlay=False,
    )
    os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)
    os.environ["C2HLS_FLASH_ONCHIP"] = "1"
    os.environ["C2HLS_FLASH_SKILL_BIN"] = "onchip"


def _configure_curated_bin_env(skills_json: Path, bin_name: str) -> None:
    """Short curated pack, no 90-skill dump, no no-RMW overlay, not onchip."""
    _configure_autosa_flash_env(
        include_avoids=True,
        skills_json=skills_json,
        flash_overlay=False,
    )
    os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)
    os.environ.pop("C2HLS_FLASH_ONCHIP", None)
    os.environ["C2HLS_FLASH_SKILL_BIN"] = bin_name


def configure_autosa_flash_generic_env() -> None:
    _configure_curated_bin_env(SKILLS_GENERIC_JSON, "generic")


def configure_autosa_flash_gemm_family_env() -> None:
    _configure_curated_bin_env(SKILLS_GEMM_FAMILY_JSON, "gemm_family")


def configure_autosa_flash_systolic_io_env() -> None:
    _configure_curated_bin_env(SKILLS_SYSTOLIC_IO_JSON, "systolic_io")


def configure_autosa_flash_noskills_env() -> None:
    _configure_autosa_runtime_env()
    _strip_autosa_skill_env()
    os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"


def configure_autosa_flash_zero_shot_env() -> None:
    _configure_autosa_runtime_env()
    _strip_autosa_skill_env()
    os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
    os.environ["C2HLS_SKIP_PHASE_B"] = "1"
    os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] = "zero_shot"
    os.environ.setdefault("C2HLS_TURNS", "1")


def configure_autosa_flash_one_shot_env() -> None:
    """True one-shot codegen: skip phase B, zero-shot prompt, no FLASH MODE extras."""
    configure_autosa_flash_zero_shot_env()
    os.environ["C2HLS_ONE_SHOT"] = "1"
    os.environ["C2HLS_FLASH_ONLY"] = "1"
    os.environ["C2HLS_POST_FLASH_DSE"] = "0"
    os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
    os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
    os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
    os.environ["C2HLS_DSE_V2"] = "0"
    os.environ["C2HLS_DSE_V2_CHAIN_FLASH"] = "0"
    for key in (
        "C2HLS_FLASH_MIN_DSP",
        "C2HLS_FLASH_MAX_DSP",
        "C2HLS_FLASH_DSP_REDO",
        "C2HLS_FLASH_DSP_FILL_PCT",
        "C2HLS_FLASH_ONCHIP",
        "C2HLS_FLASH_ONCHIP_TILE",
        "C2HLS_FLASH_PE_BLK",
        "C2HLS_FLASH_ROW_UF",
        "C2HLS_FLASH_TILE_PP",
        "C2HLS_FLASH_SKILL_BIN",
    ):
        os.environ.pop(key, None)


def configure_autosa_gold_env() -> None:
    """Csynth the autosa_ready seed (hls_baseline.cpp == plain.cpp ABI). No LLM."""
    _configure_autosa_runtime_env()
    _strip_autosa_skill_env()
    os.environ.pop("C2HLS_FLASH_ONCHIP", None)
    os.environ.pop("C2HLS_FLASH_SKILL_BIN", None)
    os.environ.pop("C2HLS_FLASH_MIN_DSP", None)
    os.environ.pop("C2HLS_FLASH_MAX_DSP", None)
    os.environ.pop("C2HLS_FLASH_PE_BLK", None)
    os.environ.pop("C2HLS_FLASH_ROW_UF", None)
    os.environ.pop("C2HLS_FLASH_K_TILE", None)
    os.environ.pop("C2HLS_FLASH_ONCHIP_TILE", None)
    os.environ["C2HLS_REFERENCE_ONLY"] = "1"
    os.environ["C2HLS_FLASH_ONLY"] = "0"
    os.environ["C2HLS_POST_FLASH_DSE"] = "0"
    os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
    os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
    os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
    os.environ["C2HLS_SKIP_PHASE_B"] = "1"
    os.environ.setdefault("C2HLS_TURNS", "0")
    os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_GOLD


def normalize_mm_flow_flavor(flavor: str) -> str:
    raw = (flavor or "skills").strip().lower().replace("-", "_")
    mapped = _MM_FLOW_FLAVOR_ALIASES.get(raw, raw)
    if mapped not in {"skills", "aav_n_90", "noskills", "zero_shot", "one_shot"}:
        raise ValueError(
            f"unknown mm-flow flavor {flavor!r} "
            "(use skills, aav_n_90, noskills, zero_shot, or one_shot)"
        )
    return mapped


def _mm_flow_prefixes(flavor: str) -> tuple[str, str]:
    pe32 = os.getenv("C2HLS_PE_RECIPE", "").strip() in {"autosa_mm_32x8", "32x8"}
    if flavor == "noskills":
        if pe32:
            return "mmns32", "batch_parallel_autosa_mm_32x8_flow_noskills"
        return "mmns", "batch_parallel_autosa_mm_flow_noskills"
    if flavor == "zero_shot":
        return "mmzs", "batch_parallel_autosa_mm_flow_zero_shot"
    if flavor == "one_shot":
        return "mm1s", "batch_parallel_autosa_mm_flow_one_shot"
    if flavor == "aav_n_90":
        if pe32:
            return "mm90_32", "batch_parallel_autosa_mm_32x8_flow_aav_n_90"
        return "mm90", "batch_parallel_autosa_mm_flow_aav_n_90"
    if pe32:
        return "mm32x8", "batch_parallel_autosa_mm_32x8_flow_aav_n_gf"
    return "mmflow", "batch_parallel_autosa_mm_flow_aav_n_gf"


def apply_mm_flow_flavor(flavor: str) -> dict[str, Any]:
    """Launcher-side mm-flow flavor. Mutates os.environ. Does not submit jobs."""
    flavor = normalize_mm_flow_flavor(flavor)
    if flavor in {"zero_shot", "one_shot"} and os.getenv("C2HLS_PE_RECIPE", "").strip():
        raise ValueError(f"--pe-recipe is not valid with --flavor {flavor}")
    os.environ["C2HLS_MM_FLOW_FLAVOR"] = flavor
    job_prefix, artifact_prefix = _mm_flow_prefixes(flavor)
    if flavor == "skills":
        os.environ.setdefault("BATCH_PARALLEL_VARIANT", VARIANT_AAV_N_GF)
        os.environ.setdefault("PC2_BATCH_JOB_PREFIX", job_prefix)
        os.environ.setdefault("BATCH_PARALLEL_ARTIFACT_PREFIX", artifact_prefix)
        os.environ["C2HLS_POST_FLASH_DSE"] = "1"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "1"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "1"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "1"
        os.environ.pop("C2HLS_POST_FLASH_NO_SKILLS", None)
        os.environ.pop("C2HLS_SKIP_PHASE_B", None)
        os.environ.pop("C2HLS_FLASH_OPT_PROMPT_MODE", None)
        os.environ.pop("C2HLS_ONE_SHOT", None)
        os.environ.setdefault("C2HLS_SKILL_PROMPT_MODE", "all_skills_avoids_global")
        os.environ.setdefault(
            "C2HLS_PACKAGED_SKILLS_JSON", str(SKILLS_90_GEMM_JSON.resolve())
        )
    elif flavor == "aav_n_90":
        os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_AAV_N_90
        os.environ["PC2_BATCH_JOB_PREFIX"] = job_prefix
        os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"] = artifact_prefix
        os.environ["C2HLS_POST_FLASH_DSE"] = "1"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "1"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "1"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "1"
        os.environ.pop("C2HLS_POST_FLASH_NO_SKILLS", None)
        os.environ.pop("C2HLS_SKIP_PHASE_B", None)
        os.environ.pop("C2HLS_FLASH_OPT_PROMPT_MODE", None)
        os.environ.pop("C2HLS_ONE_SHOT", None)
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(SKILLS_90_JSON.resolve())
        os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
        os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)
    elif flavor == "noskills":
        _strip_autosa_skill_env()
        os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
        os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_NOSKILLS
        os.environ["PC2_BATCH_JOB_PREFIX"] = job_prefix
        os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"] = artifact_prefix
        os.environ["C2HLS_POST_FLASH_DSE"] = "1"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "1"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "1"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "1"
        os.environ.pop("C2HLS_SKIP_PHASE_B", None)
        os.environ.pop("C2HLS_FLASH_OPT_PROMPT_MODE", None)
        os.environ.pop("C2HLS_ONE_SHOT", None)
    elif flavor == "zero_shot":
        _strip_autosa_skill_env()
        os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
        os.environ["C2HLS_SKIP_PHASE_B"] = "1"
        os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] = "zero_shot"
        os.environ["C2HLS_TURNS"] = "1"
        os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_ZERO_SHOT
        os.environ["PC2_BATCH_JOB_PREFIX"] = job_prefix
        os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"] = artifact_prefix
        os.environ["C2HLS_POST_FLASH_DSE"] = "0"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
        os.environ.pop("C2HLS_ONE_SHOT", None)
    else:
        _strip_autosa_skill_env()
        os.environ["C2HLS_POST_FLASH_NO_SKILLS"] = "1"
        os.environ["C2HLS_SKIP_PHASE_B"] = "1"
        os.environ["C2HLS_FLASH_OPT_PROMPT_MODE"] = "zero_shot"
        os.environ["C2HLS_ONE_SHOT"] = "1"
        os.environ["C2HLS_TURNS"] = "1"
        os.environ["BATCH_PARALLEL_VARIANT"] = VARIANT_ONE_SHOT
        os.environ["PC2_BATCH_JOB_PREFIX"] = job_prefix
        os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"] = artifact_prefix
        os.environ["C2HLS_POST_FLASH_DSE"] = "0"
        os.environ["C2HLS_DSE_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_DSE_V2"] = "0"
        os.environ["C2HLS_DSE_V2_CHAIN_FLASH"] = "0"
        os.environ["C2HLS_FLASH_ONLY"] = "1"
        for key in (
            "C2HLS_FLASH_MIN_DSP",
            "C2HLS_FLASH_MAX_DSP",
            "C2HLS_FLASH_DSP_REDO",
            "C2HLS_FLASH_ONCHIP",
        ):
            os.environ.pop(key, None)
    sweep_prefix = os.getenv("C2HLS_SWEEP_JOB_PREFIX", "").strip()
    if sweep_prefix:
        os.environ["PC2_BATCH_JOB_PREFIX"] = sweep_prefix
    sweep_art = os.getenv("C2HLS_SWEEP_ARTIFACT_PREFIX", "").strip()
    if sweep_art:
        os.environ["BATCH_PARALLEL_ARTIFACT_PREFIX"] = sweep_art
    return {
        "flavor": flavor,
        "variant": os.environ.get("BATCH_PARALLEL_VARIANT", ""),
        "job_prefix": os.environ.get("PC2_BATCH_JOB_PREFIX", ""),
        "artifact_prefix": os.environ.get("BATCH_PARALLEL_ARTIFACT_PREFIX", ""),
        "dse": os.environ.get("C2HLS_POST_FLASH_DSE", "0"),
        "stream": os.environ.get("C2HLS_POST_FLASH_STREAM", "0"),
        "no_skills": os.environ.get("C2HLS_POST_FLASH_NO_SKILLS", "0"),
        "skip_phase_b": os.environ.get("C2HLS_SKIP_PHASE_B", "0"),
        "flash_opt_prompt_mode": os.environ.get("C2HLS_FLASH_OPT_PROMPT_MODE", ""),
        "one_shot": os.environ.get("C2HLS_ONE_SHOT", "0"),
        "flash_only": os.environ.get("C2HLS_FLASH_ONLY", "0"),
        "flash_min_dsp": os.environ.get("C2HLS_FLASH_MIN_DSP", ""),
        "turns": os.environ.get("C2HLS_TURNS", ""),
        "packaged_skills": os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", ""),
    }


def apply_bench_timeouts_from_meta(meta: dict[str, Any]) -> None:
    csim = meta.get("csim_timeout_s", DEFAULT_CSIM_TIMEOUT_S)
    synth = meta.get("synth_timeout_s", DEFAULT_SYNTH_TIMEOUT_S)
    os.environ["C2HLS_CSIM_TIMEOUT"] = str(int(csim))
    os.environ["C2HLS_SYNTH_TIMEOUT"] = str(int(synth))


def resolve_autosa_benches(
    requested: list[str],
    root: Path | None = None,
) -> list[tuple[str, Path]]:
    root = Path(root) if root is not None else resolve_autosa_ready_root()
    if not root.is_dir():
        raise FileNotFoundError(f"autosa_ready root missing: {root}")
    available: dict[str, Path] = {}
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = meta.get("benchmark") or meta_path.parent.name
        available[name] = meta_path.parent
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown autosa_ready benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def list_autosa_benches(root: Path = AUTOSA_READY_ROOT) -> list[str]:
    names: list[str] = []
    if not root.is_dir():
        return names
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        bench_dir = meta_path.parent
        if not (bench_dir / "plain.cpp").is_file():
            continue
        if not (bench_dir / "hls_baseline.cpp").is_file():
            continue
        names.append(meta.get("benchmark") or bench_dir.name)
    return names
