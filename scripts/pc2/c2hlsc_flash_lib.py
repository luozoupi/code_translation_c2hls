"""c2hlsc-ready flash helpers — pure 90-skill aav_n (no no_RMW overlay)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tier_a_flash_lib import SKILLS_90_JSON, verify_skills_90

REPO = Path(__file__).resolve().parents[2]
C2HLSC_READY_ROOT = (
    REPO / "related_work/benchmarks/HLSFactory_benchmarks/c2hlsc_ready"
)

SETUP_TAG = "flash__c2hlsc__aav_n"
MATRIX_FAMILY = "flash_c2hlsc_aav_n"
DEFAULT_SYNTH_TIMEOUT_S = 3600
DEFAULT_CSIM_TIMEOUT_S = 600
DEFAULT_COSIM_TIMEOUT_S = 43200


def configure_c2hlsc_flash_aav_n_env() -> None:
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

    _apply_flash_skill_entries_env(False)
    os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)

    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
    os.environ.setdefault("C2HLS_PHASEB_MODE", "functional")
    os.environ.setdefault("C2HLS_PHASE8_BASELINE_ALIGN", "0")
    os.environ.setdefault("C2HLS_PHASE5_GT_PREPOP", "0")
    os.environ.setdefault("C2HLS_HW_EMU_FINAL", "0")
    os.environ.setdefault("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "1")
    os.environ.setdefault("C2HLS_GT_BASELINE_FALLBACK", "1")
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


def resolve_c2hlsc_benches(
    requested: list[str], *, root: Path | None = None
) -> list[tuple[str, Path]]:
    root = root or C2HLSC_READY_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"c2hlsc_ready missing: {root}")
    available = {
        p.name: p
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / "metadata.json").is_file()
    }
    if not requested:
        return [(n, available[n]) for n in sorted(available)]
    out: list[tuple[str, Path]] = []
    missing: list[str] = []
    for name in requested:
        if name in available:
            out.append((name, available[name]))
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"c2hlsc benches not found under {root}: {missing}; "
            f"available={sorted(available)}"
        )
    return out


def env_snapshot() -> dict[str, Any]:
    return {
        "matrix_family": MATRIX_FAMILY,
        "corpus": "c2hlsc_ready",
        "benchmarks_root": str(C2HLSC_READY_ROOT.resolve()),
        "skill_prompt_mode": "all_skills_avoids_global",
        "skills_json": str(SKILLS_90_JSON.resolve()),
        "flash_overlay": None,
        "setup_tag": SETUP_TAG,
        "run_cosim": True,
        "reference_cosim": True,
        "cosim_required": False,
        "timeouts": {
            "synth_s": int(os.getenv("C2HLS_SYNTH_TIMEOUT", str(DEFAULT_SYNTH_TIMEOUT_S))),
            "csim_s": int(os.getenv("C2HLS_CSIM_TIMEOUT", str(DEFAULT_CSIM_TIMEOUT_S))),
            "cosim_s": int(os.getenv("C2HLS_COSIM_TIMEOUT", str(DEFAULT_COSIM_TIMEOUT_S))),
        },
    }
