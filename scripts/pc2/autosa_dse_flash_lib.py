"""AutoSA DSE (benchmarks_autosa_dse) flash helpers — 90-skill aav_n, csim+csynth (cosim off)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tier_a_flash_lib import SKILLS_90_JSON, verify_skills_90

REPO = Path(__file__).resolve().parents[2]
AUTOSA_DSE_ROOT = REPO / "benchmarks_autosa_dse"

SETUP_TAG = "flash__autosa_dse__aav_n"
MATRIX_FAMILY = "flash_autosa_dse_aav_n"

DEFAULT_SYNTH_TIMEOUT_S = 14400
DEFAULT_CSIM_TIMEOUT_S = 7200
DEFAULT_COSIM_TIMEOUT_S = 43200


def configure_autosa_dse_flash_aav_n_env() -> None:
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
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(SKILLS_90_JSON.resolve())
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    from flash_shared.new_skills_lib import _apply_flash_skill_entries_env

    _apply_flash_skill_entries_env(True)
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    # Phase B must seed from plain.cpp (LLM translate / skip_phase_a plain seed),
    # NOT from gold AutoSA HLS. Gold is reference-only for comparison.
    os.environ["C2HLS_PHASEB_FROM_GOLD"] = "0"
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
    # Cosim off: AutoSA kernels are too large for full-kernel LLM cosim repair.
    os.environ["C2HLS_RUN_COSIM"] = "0"
    os.environ["C2HLS_COSIM_REQUIRED"] = "0"
    os.environ["C2HLS_REFERENCE_COSIM"] = "0"
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", str(DEFAULT_CSIM_TIMEOUT_S))
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", str(DEFAULT_COSIM_TIMEOUT_S))
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def apply_bench_timeouts_from_meta(meta: dict[str, Any]) -> None:
    csim = meta.get("csim_timeout_s", DEFAULT_CSIM_TIMEOUT_S)
    synth = meta.get("synth_timeout_s", DEFAULT_SYNTH_TIMEOUT_S)
    cosim = meta.get("cosim_timeout_s", DEFAULT_COSIM_TIMEOUT_S)
    os.environ["C2HLS_CSIM_TIMEOUT"] = str(int(csim))
    os.environ["C2HLS_SYNTH_TIMEOUT"] = str(int(synth))
    if meta.get("supports_cosim"):
        os.environ["C2HLS_COSIM_TIMEOUT"] = str(int(cosim))


def resolve_autosa_dse_benches(
    requested: list[str],
    root: Path = AUTOSA_DSE_ROOT,
) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"benchmarks_autosa_dse root missing: {root}")
    available: dict[str, Path] = {}
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = meta.get("c2hls_bench_id") or meta.get("benchmark") or meta_path.parent.name
        available[name] = meta_path.parent
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown benchmarks_autosa_dse benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def list_autosa_dse_benches(root: Path = AUTOSA_DSE_ROOT) -> list[str]:
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
        names.append(meta.get("c2hls_bench_id") or meta.get("benchmark") or bench_dir.name)
    return names
