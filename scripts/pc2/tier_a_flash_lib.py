"""Tier A ready corpus helpers for PC2 flash smoke / batch runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TIER_A_READY_ROOT = REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"
SKILLS_PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
SKILLS_90_JSON = SKILLS_PKG / "skills_ii_target_miss_solutions_added(90skills).json"

DEFAULT_SMOKE_BENCHES = [
    "spector_hls_dct",
    "hp_fft_n256__UF1",
]

SETUP_TAG = "flash__tier_a__90skills"
STAMP_ENV = "C2HLS_TIER_A_FLASH_SMOKE_STAMP"
OUT_ENV = "C2HLS_TIER_A_FLASH_SMOKE_OUT"
MATRIX_FAMILY = "flash_tier_a_ready_smoke"


def count_skills(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    skills = data.get("skills") or []
    return len(skills) if isinstance(skills, list) else 0


def verify_skills_90() -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "skills_json": str(SKILLS_90_JSON.resolve()),
        "skill_count": 0,
        "expected_skill_count": 90,
    }
    if not SKILLS_90_JSON.is_file():
        out["ok"] = False
        out["errors"].append(f"missing skills file: {SKILLS_90_JSON}")
        return out
    got = count_skills(SKILLS_90_JSON)
    out["skill_count"] = got
    if got != 90:
        out["ok"] = False
        out["errors"].append(f"skill count {got} != expected 90")
    return out


def list_tier_a_benches(root: Path = TIER_A_READY_ROOT) -> list[str]:
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


def resolve_tier_a_benches(
    requested: list[str],
    root: Path = TIER_A_READY_ROOT,
) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"tier_A_ready root missing: {root}")
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
        raise ValueError(f"unknown tier_A_ready benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def configure_tier_a_flash_90skills_env() -> None:
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "flash"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_SKILL_MODE"] = "skill_on"
    os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
    os.environ["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(SKILLS_90_JSON.resolve())
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    os.environ.setdefault("C2HLS_RECORD_FLOW", "1")
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
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "180")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "900")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def env_snapshot() -> dict[str, Any]:
    snap = {
        "matrix_family": MATRIX_FAMILY,
        "corpus": "tier_A_ready",
        "benchmarks_root": str(TIER_A_READY_ROOT.resolve()),
        "record_flow": os.getenv("C2HLS_RECORD_FLOW", "1") == "1",
        "skill_prompt_mode": "all_skills_avoids_global",
        "force_skill_prompts": True,
        "skills_json": str(SKILLS_90_JSON.resolve()),
        "skills_json_mode": "packaged_only",
        "target_part": os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
        "target_clock_ns": os.getenv("C2HLS_CLOCK_NS", "3.33"),
        "timeouts": {
            "synth_s": int(os.getenv("C2HLS_SYNTH_TIMEOUT", "1200")),
            "csim_s": int(os.getenv("C2HLS_CSIM_TIMEOUT", "180")),
            "llm_s": int(os.getenv("C2HLS_LLM_TIMEOUT", "900")),
        },
    }
    if SKILLS_90_JSON.is_file():
        snap["skills_json_count"] = count_skills(SKILLS_90_JSON)
    return snap
