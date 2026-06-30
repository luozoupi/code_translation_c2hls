"""Multistep fixed-cosim on ``benchmarks_cosim/``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

REPO = Path(__file__).resolve().parents[2]
BENCHMARKS_COSIM_DIR = REPO / "benchmarks_cosim"
SKILLS_PKG = REPO / "hls_full_optimization_skills_schema_1_1_package"
NEW_SKILLS_JSON_90 = SKILLS_PKG / "skills_ii_target_miss_solutions_added(90skills).json"

STAMP_ENV = "C2HLS_MULTISTEP_FIXED_COSIM_STAMP"
OUT_ENV = "C2HLS_MULTISTEP_FIXED_COSIM_OUT"
MATRIX_FAMILY = "multistep_fixed_cosim_benchmarks_cosim"

PILOT_BENCHES = [
    "hlsfactory_2mm",
    "hlsfactory_gemm",
    "hlsfactory_correlation",
    "hlsfactory_floyd-warshall",
    "hlsfactory_syrk",
]

InferenceKind = Literal["vllm"]


@dataclass(frozen=True)
class MultistepFixedCosimVariant:
    key: str
    label: str
    session_id: str
    artifact_prefix: str
    setup_tag: str
    skill_prompt_mode: str
    skills_json: Optional[Path]
    force_skill_prompts: bool
    skills_in_prompt: bool = True

    @property
    def stamp_env(self) -> str:
        return f"C2HLS_MULTISTEP_FIXED_{self.key.upper()}_STAMP"

    @property
    def out_env(self) -> str:
        return f"C2HLS_MULTISTEP_FIXED_{self.key.upper()}_OUT"


VARIANTS: dict[str, MultistepFixedCosimVariant] = {
    "aav_n": MultistepFixedCosimVariant(
        key="aav_n",
        label="All+avoids (new, 90 skills)",
        session_id="multistep_fixed_cosim_aav_n",
        artifact_prefix="multistep_fixed_cosim_aav_n",
        setup_tag="multistep__fixed_cosim__aav_n",
        skill_prompt_mode="all_skills_avoids_global",
        skills_json=NEW_SKILLS_JSON_90,
        force_skill_prompts=True,
    ),
    "nav_n": MultistepFixedCosimVariant(
        key="nav_n",
        label="No avoids (new, 90 skills)",
        session_id="multistep_fixed_cosim_nav_n",
        artifact_prefix="multistep_fixed_cosim_nav_n",
        setup_tag="multistep__fixed_cosim__nav_n",
        skill_prompt_mode="all_skills_no_avoids_global",
        skills_json=NEW_SKILLS_JSON_90,
        force_skill_prompts=True,
    ),
    "noskills": MultistepFixedCosimVariant(
        key="noskills",
        label="Noskills",
        session_id="multistep_fixed_cosim_noskills",
        artifact_prefix="multistep_fixed_cosim_noskills",
        setup_tag="multistep__fixed_cosim__noskills",
        skill_prompt_mode="bottleneck",
        skills_json=None,
        force_skill_prompts=False,
        skills_in_prompt=False,
    ),
}

VARIANT_ORDER = ["aav_n", "nav_n", "noskills"]


def count_skills_in_file(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    skills = data.get("skills") or []
    if not isinstance(skills, list):
        return 0
    return len(skills)


def verify_variant_skills(variant: MultistepFixedCosimVariant) -> dict:
    out: dict = {"variant": variant.key, "label": variant.label, "ok": True, "errors": []}
    if variant.skills_json is None:
        out["skills_json"] = None
        out["skill_count"] = 0
        out["expected_skill_count"] = 0
        return out
    path = variant.skills_json
    if not path.is_file():
        out["ok"] = False
        out["errors"].append(f"missing skills file: {path}")
        return out
    got = count_skills_in_file(path)
    want = got
    out["skills_json"] = str(path.resolve())
    out["skill_count"] = got
    out["expected_skill_count"] = want
    if got < 1:
        out["ok"] = False
        out["errors"].append(f"empty skills file: {path.name}")
    return out


def list_cosim_benches(bench_root: Path = BENCHMARKS_COSIM_DIR) -> list[str]:
    names: list[str] = []
    for meta_path in sorted(bench_root.glob("hlsfactory_*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not meta.get("supports_cosim"):
            continue
        bench_dir = meta_path.parent
        if not (bench_dir / "plain.cpp").is_file():
            continue
        names.append(meta.get("benchmark") or bench_dir.name)
    return names


def resolve_cosim_benches(
    requested: list[str],
    bench_root: Path = BENCHMARKS_COSIM_DIR,
) -> list[tuple[str, Path]]:
    available: dict[str, Path] = {}
    for meta_path in sorted(bench_root.glob("hlsfactory_*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not meta.get("supports_cosim"):
            continue
        name = meta.get("benchmark") or meta_path.parent.name
        available[name] = meta_path.parent
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"unknown or unsupported cosim benchmark(s): {missing}")
    return [(name, available[name]) for name in requested]


def configure_fixed_cosim_multistep_env(
    variant: MultistepFixedCosimVariant,
    *,
    inference: InferenceKind = "vllm",
) -> None:
    from c2hls_paths import apply_runtime_defaults
    from c2hls_temp import configure_temp_env

    apply_runtime_defaults(profile="sweep")
    configure_temp_env(create=True)

    os.environ["C2HLS_STRATEGY"] = "static"
    os.environ["C2HLS_DYNAMIC_ROUTING"] = "0"
    os.environ["C2HLS_RECORD_FLOW"] = "1"
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
    os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "3600")
    os.environ.setdefault("C2HLS_CSIM_TIMEOUT", "600")
    os.environ.setdefault("C2HLS_COSIM_TIMEOUT", "1200")
    os.environ.setdefault("C2HLS_LLM_TIMEOUT", "1800")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    if variant.force_skill_prompts and variant.skills_json is not None:
        os.environ["C2HLS_SKILL_MODE"] = "skill_on"
        os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
        os.environ["C2HLS_SKILL_PROMPT_MODE"] = variant.skill_prompt_mode
        os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(variant.skills_json.resolve())
        os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    else:
        os.environ["C2HLS_SKILL_MODE"] = "skill_off"
        os.environ["C2HLS_FORCE_SKILL_PROMPTS"] = "0"
        os.environ.pop("C2HLS_SKILL_PROMPT_MODE", None)
        os.environ.pop("C2HLS_PACKAGED_SKILLS_JSON", None)
        os.environ.pop("C2HLS_PACKAGED_SKILLS_ONLY", None)


def variant_env_snapshot(variant: MultistepFixedCosimVariant) -> dict:
    snap = {
        "matrix_family": MATRIX_FAMILY,
        "benchmarks_root": str(BENCHMARKS_COSIM_DIR.resolve()),
        "corpus": "benchmarks_cosim",
        "record_flow": True,
        "strategy": "static",
        "variant": variant.key,
        "label": variant.label,
        "skill_prompt_mode": variant.skill_prompt_mode,
        "force_skill_prompts": variant.force_skill_prompts,
        "skills_in_prompt": variant.skills_in_prompt,
        "skills_json": str(variant.skills_json.resolve()) if variant.skills_json else None,
        "skills_json_mode": "packaged_only" if variant.skills_json else None,
        "origin_meta": {
            "note": (
                "HLSFactory benchmarks_cosim has gold/baseline only; "
                "no per-step GT variants for step-level vs_ground_truth."
            ),
        },
    }
    if variant.skills_json and variant.skills_json.is_file():
        snap["skills_json_count"] = count_skills_in_file(variant.skills_json)
    return snap
