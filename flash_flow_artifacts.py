"""Flow pipeline artifact helpers (``C2HLS_RECORD_FLOW=1``)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional


_SKILL_ENV_KEYS = (
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_CURATION",
    "C2HLS_SKILL_CURATION_FOCUS",
    "C2HLS_SKILL_CURATION_SECTOR",
    "C2HLS_SKILL_CURATION_INCLUDE_AVOIDS",
    "C2HLS_BOTTLENECK_POSITIVE_SKILLS",
    "C2HLS_BOTTLENECK_AVOID_SKILLS",
)

MULTISTEP_OPT_STEPS = (
    "tiling",
    "pipeline",
    "unroll",
    "doublebuffer",
    "coalescing",
)


def record_flow_enabled() -> bool:
    return os.getenv("C2HLS_RECORD_FLOW", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def record_flow_legacy_steps_enabled() -> bool:
    return os.getenv("C2HLS_RECORD_FLOW_LEGACY_STEPS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def skill_env_snapshot() -> dict[str, str]:
    return {key: os.getenv(key, "") for key in _SKILL_ENV_KEYS}


def skills_source_snapshot() -> dict[str, Any]:
    """Metadata for the packaged skills JSON on disk."""
    from skill_library import _packaged_skills_path

    path = _packaged_skills_path()
    snap: dict[str, Any] = {
        "path": str(path.resolve()) if path.exists() else str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return snap
    text = path.read_text(encoding="utf-8")
    snap["sha256"] = sha256_text(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        snap["parse_error"] = True
        return snap
    snap["schema"] = data.get("schema")
    snap["saved_at"] = data.get("saved_at")
    skills = data.get("skills") or []
    if isinstance(skills, list):
        snap["skill_count_in_file"] = len(skills)
        snap["skill_ids_in_file"] = [
            entry.get("id") for entry in skills if isinstance(entry, dict) and entry.get("id")
        ]
    return snap


def skills_to_dicts(skills: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sk in skills:
        if sk is None:
            continue
        if hasattr(sk, "__dataclass_fields__"):
            out.append(asdict(sk))
        elif isinstance(sk, dict):
            out.append(dict(sk))
    return out


def _skills_from_ids(skill_library: Any, skill_ids: list[str]) -> list[dict[str, Any]]:
    if skill_library is None:
        return []
    resolved = []
    for sid in skill_ids:
        sk = skill_library.get(sid)
        if sk is not None:
            resolved.append(sk)
    return skills_to_dicts(resolved)


def capture_step_skills(
    *,
    step_name: str,
    skill_prompt_mode: str,
    skill_header: str,
    prompt_skills: list[Any],
    injected_prompt_text: str,
    top_bottleneck_kind: Optional[str],
    skill_id: Optional[str],
    skill_curation_record: Optional[dict[str, Any]],
    skill_library: Any,
) -> dict[str, Any]:
    """Build a full skills record for one optimization LLM call."""
    injected_skills = skills_to_dicts(prompt_skills)
    is_flash = step_name == "flash"
    schema = "flash_flow_skills_v1" if is_flash else "multistep_step_skills_v1"
    opt_block: dict[str, Any] = {
        "step_name": step_name,
        "skill_prompt_mode": skill_prompt_mode,
        "skill_header": skill_header,
        "top_bottleneck_kind": top_bottleneck_kind,
        "routed_skill_id": skill_id,
        "injected_skill_count": len(injected_skills),
        "injected_skills": injected_skills,
        "injected_prompt_text": injected_prompt_text,
    }
    if skill_curation_record:
        curated = dict(skill_curation_record)
        curated["selected_skills"] = _skills_from_ids(
            skill_library, curated.get("selected_skill_ids") or []
        )
        curated["avoid_skills"] = _skills_from_ids(
            skill_library, curated.get("avoid_skill_ids") or []
        )
        opt_block["skill_curation"] = curated

    record: dict[str, Any] = {
        "schema": schema,
        "step_name": step_name,
        "skills_source": skills_source_snapshot(),
        "skill_env": skill_env_snapshot(),
    }
    if is_flash:
        record["phase_b"] = {
            "skills_injected": False,
            "note": "Functional Phase B translate does not inject optimization skills.",
        }
        record["flash_opt"] = opt_block
    else:
        record["optimization"] = opt_block
    return record


def capture_flash_step_skills(
    *,
    step_name: str,
    skill_prompt_mode: str,
    skill_header: str,
    prompt_skills: list[Any],
    injected_prompt_text: str,
    top_bottleneck_kind: Optional[str],
    skill_id: Optional[str],
    skill_curation_record: Optional[dict[str, Any]],
    skill_library: Any,
) -> dict[str, Any]:
    """Backward-compatible alias for flash step skill capture."""
    return capture_step_skills(
        step_name=step_name,
        skill_prompt_mode=skill_prompt_mode,
        skill_header=skill_header,
        prompt_skills=prompt_skills,
        injected_prompt_text=injected_prompt_text,
        top_bottleneck_kind=top_bottleneck_kind,
        skill_id=skill_id,
        skill_curation_record=skill_curation_record,
        skill_library=skill_library,
    )


def write_flash_skills_record(
    output_dir: str | Path,
    bench_name: str,
    skills_context: Optional[dict[str, Any]],
) -> Optional[str]:
    """Write ``{bench}_flash_skills.json`` and copy ``skills_source.json``."""
    if not skills_context:
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel_skills = f"{bench_name}_flash_skills.json"
    (out / rel_skills).write_text(
        json.dumps(skills_context, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    rel_source: Optional[str] = None
    src_path = (skills_context.get("skills_source") or {}).get("path", "")
    if src_path:
        src = Path(src_path)
        if src.is_file():
            (out / "skills_source.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            rel_source = "skills_source.json"
    return rel_skills if rel_skills else None


def write_multistep_skills_record(
    output_dir: str | Path,
    bench_name: str,
    skills_records: list[dict[str, Any]] | None,
) -> Optional[str]:
    """Write ``{bench}_multistep_skills.json`` (list of per-step captures)."""
    if not skills_records:
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel_skills = f"{bench_name}_multistep_skills.json"
    payload = {
        "schema": "multistep_flow_skills_v1",
        "benchmark": bench_name,
        "steps": skills_records,
    }
    (out / rel_skills).write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    src_path = ""
    for rec in skills_records:
        src_path = (rec.get("skills_source") or {}).get("path", "")
        if src_path:
            break
    if src_path:
        src = Path(src_path)
        if src.is_file():
            (out / "skills_source.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return rel_skills


def _latency_from_report(report: dict[str, Any] | None) -> Any:
    if not isinstance(report, dict):
        return None
    return report.get("latency_cycles")


def infer_flow_selected_from(results: dict[str, Any]) -> str:
    """Return ``phase_b`` or ``flash_opt`` for the latency-selected kernel."""
    baseline_report = results.get("baseline_report") or {}
    final_report = results.get("final_report") or {}
    b_lat = _latency_from_report(baseline_report)
    f_lat = _latency_from_report(final_report)
    flash_step = next(
        (s for s in (results.get("steps") or []) if s.get("step_name") == "flash"),
        None,
    )
    o_lat = _latency_from_report((flash_step or {}).get("report"))
    if f_lat is not None and b_lat is not None and f_lat == b_lat:
        return "phase_b"
    if f_lat is not None and o_lat is not None and f_lat == o_lat:
        return "flash_opt"
    promo = results.get("best_so_far_promotion") or {}
    if promo.get("promoted"):
        name = promo.get("from_step_name")
        if name in ("baseline", "phase_b"):
            return "phase_b"
        if name == "flash":
            return "flash_opt"
    if flash_step and flash_step.get("success") and o_lat is not None and b_lat is not None:
        return "flash_opt" if o_lat < b_lat else "phase_b"
    return "phase_b"


def infer_multistep_selected_from(results: dict[str, Any]) -> str:
    """Return the step role that produced the selected kernel (multistep)."""
    baseline_report = results.get("baseline_report") or {}
    final_report = results.get("final_report") or {}
    b_lat = _latency_from_report(baseline_report)
    f_lat = _latency_from_report(final_report)

    step_lats: dict[str, Any] = {"phase_b": b_lat}
    for step in results.get("steps") or []:
        name = step.get("step_name")
        if not name or not step.get("success"):
            continue
        step_lats[name] = _latency_from_report(step.get("report"))

    if f_lat is not None:
        for role, lat in step_lats.items():
            if lat is not None and lat == f_lat:
                return role

    promo = results.get("best_so_far_promotion") or {}
    if promo.get("promoted"):
        name = promo.get("from_step_name")
        if name in ("baseline", "phase_b"):
            return "phase_b"
        if name in MULTISTEP_OPT_STEPS:
            return name

    best_role = "phase_b"
    best_lat = b_lat
    for role in MULTISTEP_OPT_STEPS:
        lat = step_lats.get(role)
        if lat is not None and (best_lat is None or lat < best_lat):
            best_lat = lat
            best_role = role
    return best_role


def resolve_cell_final_cpp(cell_dir: Path, bench: str) -> Optional[Path]:
    """Pick cosim kernel source: selected (record-flow) or legacy final."""
    selected = cell_dir / f"{bench}_selected.cpp"
    if selected.is_file():
        return selected
    finals = sorted(cell_dir.glob(f"{bench}_final.cpp"))
    if finals:
        return finals[0]
    return None


def resolve_cell_kernel_cpp(
    cell_dir: Path,
    bench: str,
    kernel_source: str = "selected",
) -> Optional[Path]:
    """Pick cosim kernel source by record-flow role."""
    role = (kernel_source or "selected").strip().lower().replace("-", "_")
    if role in ("phase_b", "translator", "translated"):
        path = cell_dir / f"{bench}_phase_b.cpp"
        return path if path.is_file() else None
    if role in ("flash_opt", "flash"):
        path = cell_dir / f"{bench}_flash_opt.cpp"
        return path if path.is_file() else None
    if role in MULTISTEP_OPT_STEPS:
        path = cell_dir / f"{bench}_{role}.cpp"
        return path if path.is_file() else None
    return resolve_cell_final_cpp(cell_dir, bench)


def write_flow_artifacts(
    output_dir: str | Path,
    bench_name: str,
    *,
    plain_code: str,
    phase_b_code: str,
    phase_b_report: dict[str, Any],
    flash_opt_code: str,
    flash_opt_report: dict[str, Any],
    selected_code: str,
    selected_report: dict[str, Any],
    results: dict[str, Any],
    skills_context: Optional[dict[str, Any]] = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_from = infer_flow_selected_from(results)

    if plain_code:
        (out / "plain.cpp").write_text(plain_code, encoding="utf-8")

    if phase_b_code:
        (out / f"{bench_name}_phase_b.cpp").write_text(phase_b_code, encoding="utf-8")
        (out / f"{bench_name}_phase_b_report.json").write_text(
            json.dumps(phase_b_report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    if flash_opt_code:
        (out / f"{bench_name}_flash_opt.cpp").write_text(flash_opt_code, encoding="utf-8")
        (out / f"{bench_name}_flash_opt_report.json").write_text(
            json.dumps(flash_opt_report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    if selected_code:
        (out / f"{bench_name}_selected.cpp").write_text(selected_code, encoding="utf-8")
        (out / f"{bench_name}_selected_report.json").write_text(
            json.dumps(selected_report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (out / f"{bench_name}_final.cpp").write_text(selected_code, encoding="utf-8")

    flash_step = next(
        (s for s in (results.get("steps") or []) if s.get("step_name") == "flash"),
        None,
    )
    skills_file = write_flash_skills_record(out, bench_name, skills_context)
    manifest = {
        "schema": "flash_flow_manifest_v1",
        "benchmark": bench_name,
        "selected_from": selected_from,
        "files": {
            "plain": "plain.cpp",
            "phase_b": f"{bench_name}_phase_b.cpp" if phase_b_code else None,
            "phase_b_report": f"{bench_name}_phase_b_report.json" if phase_b_code else None,
            "flash_opt": f"{bench_name}_flash_opt.cpp" if flash_opt_code else None,
            "flash_opt_report": f"{bench_name}_flash_opt_report.json" if flash_opt_code else None,
            "selected": f"{bench_name}_selected.cpp" if selected_code else None,
            "selected_report": f"{bench_name}_selected_report.json" if selected_code else None,
            "legacy_final": f"{bench_name}_final.cpp" if selected_code else None,
            "flash_skills": skills_file,
            "skills_source": "skills_source.json" if skills_file else None,
        },
        "latency_cycles": {
            "phase_b": phase_b_report.get("latency_cycles"),
            "flash_opt": flash_opt_report.get("latency_cycles"),
            "selected": selected_report.get("latency_cycles"),
        },
        "plain_c_sha256": sha256_text(plain_code) if plain_code else None,
        "phase_b_sha256": sha256_text(phase_b_code) if phase_b_code else None,
        "flash_opt_sha256": sha256_text(flash_opt_code) if flash_opt_code else None,
        "selected_sha256": sha256_text(selected_code) if selected_code else None,
        "flash_step_success": bool((flash_step or {}).get("success")),
        "best_so_far_promotion": results.get("best_so_far_promotion"),
    }
    (out / f"{bench_name}_flow_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_multistep_flow_artifacts(
    output_dir: str | Path,
    bench_name: str,
    *,
    plain_code: str,
    phase_b_code: str,
    phase_b_report: dict[str, Any],
    step_artifacts: list[dict[str, Any]],
    selected_code: str,
    selected_report: dict[str, Any],
    results: dict[str, Any],
    skills_records: list[dict[str, Any]] | None = None,
) -> None:
    """Persist multistep pipeline artifacts (``multistep_flow_manifest_v1``)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_from = infer_multistep_selected_from(results)

    if plain_code:
        (out / "plain.cpp").write_text(plain_code, encoding="utf-8")

    if phase_b_code:
        (out / f"{bench_name}_phase_b.cpp").write_text(phase_b_code, encoding="utf-8")
        (out / f"{bench_name}_phase_b_report.json").write_text(
            json.dumps(phase_b_report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    files: dict[str, Any] = {
        "plain": "plain.cpp" if plain_code else None,
        "phase_b": f"{bench_name}_phase_b.cpp" if phase_b_code else None,
        "phase_b_report": f"{bench_name}_phase_b_report.json" if phase_b_code else None,
    }
    latency_cycles: dict[str, Any] = {
        "phase_b": phase_b_report.get("latency_cycles"),
        "selected": selected_report.get("latency_cycles"),
    }
    step_success: dict[str, bool] = {}

    for entry in step_artifacts:
        step_name = entry.get("step_name")
        code = entry.get("code") or ""
        report = entry.get("report") or {}
        if not step_name:
            continue
        if code:
            cpp_name = f"{bench_name}_{step_name}.cpp"
            report_name = f"{bench_name}_{step_name}_report.json"
            (out / cpp_name).write_text(code, encoding="utf-8")
            (out / report_name).write_text(
                json.dumps(report, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            files[step_name] = cpp_name
            files[f"{step_name}_report"] = report_name
        latency_cycles[step_name] = report.get("latency_cycles")
        step_success[step_name] = bool(entry.get("success"))

    if selected_code:
        (out / f"{bench_name}_selected.cpp").write_text(selected_code, encoding="utf-8")
        (out / f"{bench_name}_selected_report.json").write_text(
            json.dumps(selected_report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (out / f"{bench_name}_final.cpp").write_text(selected_code, encoding="utf-8")
        files["selected"] = f"{bench_name}_selected.cpp"
        files["selected_report"] = f"{bench_name}_selected_report.json"
        files["legacy_final"] = f"{bench_name}_final.cpp"

    skills_file = write_multistep_skills_record(out, bench_name, skills_records)
    if skills_file:
        files["multistep_skills"] = skills_file
        files["skills_source"] = "skills_source.json"

    manifest = {
        "schema": "multistep_flow_manifest_v1",
        "benchmark": bench_name,
        "selected_from": selected_from,
        "origin_meta": {
            "note": (
                "HLSFactory benchmarks_cosim corpus has gold/baseline only; "
                "per-step vs_ground_truth compares against overall gold, not step-specific GT."
            ),
        },
        "files": files,
        "latency_cycles": latency_cycles,
        "step_success": step_success,
        "plain_c_sha256": sha256_text(plain_code) if plain_code else None,
        "phase_b_sha256": sha256_text(phase_b_code) if phase_b_code else None,
        "selected_sha256": sha256_text(selected_code) if selected_code else None,
        "best_so_far_promotion": results.get("best_so_far_promotion"),
    }
    (out / f"{bench_name}_flow_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
