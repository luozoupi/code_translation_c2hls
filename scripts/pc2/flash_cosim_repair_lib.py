"""Single-shot LLM cosim repair for championship-mode flash cosim failures (PC2)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from c2hls import extract_cpp_code
from c2hls_temp import join_temp_tag, temp_tag_scope
from skill_library import TIER_AVOID, SkillLibrary, render_skill_set_for_prompt

REPO = Path(__file__).resolve().parents[2]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PC2_ARTIFACTS = REPO / "artifacts" / "pc2"
DEFAULT_COSIM_RUN = PC2_ARTIFACTS / "flash_cosim" / "20260622_110920"
DEFAULT_REPAIR_ROOT = PC2_ARTIFACTS / "flash_cosim_repair"
DEFAULT_REPAIR_MULTILOOP_ROOT = PC2_ARTIFACTS / "flash_cosim_repair_multiloop"
DEFAULT_MAX_REPAIR_LOOPS = 1

NEW_SKILLS_JSON = (
    REPO
    / "hls_full_optimization_skills_schema_1_1_package"
    / "skills_ii_target_miss_solutions_added(90skills).json"
)
FLASH_SKILL_ENTRIES_JSON = (
    REPO
    / "hls_full_optimization_skills_schema_1_1_package"
    / "flash_no_RMW_m_axi_skill_entries.json"
)

REPAIR_VARIANTS = ("noskills", "all_skills_avoids", "all_skills_no_avoids")

# Three PC2 sessions: one championship flash mode + one repair prompt style each.
REPAIR_SESSIONS: dict[str, dict[str, str]] = {
    "all_avoids_new": {
        "session_id": "cosim_repair_all_avoids_new",
        "label": "All+avoids (new)",
        "artifact_basename": "flash_all_new_skills_avoids_global_20260621_020847",
        "repair_variant": "all_skills_avoids",
    },
    "no_avoids_old": {
        "session_id": "cosim_repair_no_avoids_old",
        "label": "No avoids (old)",
        "artifact_basename": "flash_all_skills_no_avoids_global_20260620_113247",
        "repair_variant": "all_skills_no_avoids",
    },
    "noskills_old": {
        "session_id": "cosim_repair_noskills_old",
        "label": "Noskills (old)",
        "artifact_basename": "flash_noskills_20260620_004507",
        "repair_variant": "noskills",
    },
}

REPAIR_SESSION_KEYS = tuple(REPAIR_SESSIONS.keys())

CHAMPIONSHIP_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("cur_all_json_bn", "flash_curated_all_avoids_json_bottleneck_20260621_104044"),
    ("no_avoids_old", "flash_all_skills_no_avoids_global_20260620_113247"),
    ("all_avoids_new", "flash_all_new_skills_avoids_global_20260621_020847"),
    ("noskills_old", "flash_noskills_20260620_004507"),
    ("all_avoids_old", "flash_all_skills_avoids_global_20260620_113247"),
)

CHAMPIONSHIP_BASENAMES = {basename for _, basename in CHAMPIONSHIP_ARTIFACTS}


@dataclass(frozen=True)
class RepairJob:
    index: int
    job_id: str
    repair_variant: str
    source_cell_id: str
    source_artifact_basename: str
    source_artifact_key: str
    bench: str
    setup_tag: str
    cell_dir: str
    final_cpp: str
    cosim_error: str
    source_cosim_work_dir: str
    matrix_family: str
    variant: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def repair_run_root(stamp: Optional[str] = None) -> Path:
    run_stamp = stamp or os.getenv("C2HLS_FLASH_COSIM_REPAIR_STAMP", "").strip() or _utc_stamp()
    root = Path(os.getenv("C2HLS_FLASH_COSIM_REPAIR_ROOT", str(DEFAULT_REPAIR_ROOT))) / run_stamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def manifest_path(run_root: Path) -> Path:
    return run_root / "manifest.json"


def job_dir(run_root: Path, job_id: str) -> Path:
    return run_root / "jobs" / job_id


def loop_dir(run_root: Path, job_id: str, loop_index: int) -> Path:
    return job_dir(run_root, job_id) / "loops" / f"loop_{loop_index:02d}"


def repair_loop_out_dir(run_root: Path, job_id: str, loop_index: int, *, max_loops: int) -> Path:
    if max_loops <= 1:
        return job_dir(run_root, job_id)
    return loop_dir(run_root, job_id, loop_index)


def make_job_id(source_cell_id: str, repair_variant: str) -> str:
    return join_temp_tag(source_cell_id, "repair", repair_variant)


def _load_skills_block(repair_variant: str) -> str:
    if repair_variant == "noskills":
        return ""
    os.environ.setdefault("C2HLS_PACKAGED_SKILLS_JSON", str(NEW_SKILLS_JSON))
    os.environ.setdefault("C2HLS_PACKAGED_SKILLS_ONLY", "1")
    os.environ.setdefault("C2HLS_FLASH_SKILL_ENTRIES_JSON", str(FLASH_SKILL_ENTRIES_JSON))
    from skill_library import make_default_library

    library = make_default_library(persist=False)
    all_skills = library.all()
    if repair_variant == "all_skills_no_avoids":
        skills = [sk for sk in all_skills if sk.confidence != TIER_AVOID]
    else:
        skills = list(all_skills)
    block = render_skill_set_for_prompt(skills, max_skills=len(skills))
    if not block or "No matching skills" in block:
        return ""
    header = (
        "OPTIONAL HLS OPTIMIZATION SKILLS (hints only — use your own judgment; "
        "correctness against the cosim testbench is the priority):\n\n"
    )
    return header + block


def build_diagnose_prompt(
    *,
    bench: str,
    top_function: str,
    hls_code: str,
    cosim_error: str,
    repair_variant: str,
) -> str:
    skills = _load_skills_block(repair_variant)
    skills_section = f"\n{skills}\n" if skills else ""
    return f"""You are an expert Xilinx Vitis HLS engineer. A generated kernel failed RTL co-simulation (cosim).

Benchmark: {bench}
Top function: {top_function}

Cosim / testbench error (truncated):
{cosim_error[:4000]}

Source code:
```cpp
{hls_code}
```
{skills_section}
Analyze why cosim failed (functional mismatch vs gold, missing array writes, wrong algorithm, stencil dependency bug, accumulation order, SIGSEGV/memory, etc.).

Do NOT output code yet. Respond with:
1. **Root cause** — 2-4 sentences
2. **Specific bugs** — bullet list with line/loop references where possible
3. **Fix plan** — numbered concrete steps to pass cosim while keeping HLS pragmas where sensible
"""


def build_repair_prompt(
    *,
    bench: str,
    top_function: str,
    hls_code: str,
    cosim_error: str,
    diagnosis: str,
    repair_variant: str,
) -> str:
    skills = _load_skills_block(repair_variant)
    skills_section = f"\n{skills}\n" if skills else ""
    return f"""Apply the fix plan so the kernel passes Vitis HLS cosim against the gold testbench.

Benchmark: {bench}
Top function: {top_function} (keep this symbol name)

Cosim error:
{cosim_error[:2000]}

Diagnosis and fix plan:
{diagnosis}

Original code:
```cpp
{hls_code}
```
{skills_section}
Output the COMPLETE fixed C++ translation unit in a single ```cpp fenced block.
Preserve `extern "C"` and the top function `{top_function}` unless the testbench requires otherwise.
Focus on functional correctness first; keep or simplify HLS pragmas as needed.
"""


def validation_error_text(validation: dict[str, Any]) -> str:
    """Pick the most relevant validation error for the next repair loop."""
    for phase in ("cosim", "csim", "synth"):
        block = validation.get(phase) or {}
        if block.get("passed"):
            continue
        err = (block.get("error") or "").strip()
        if err:
            return f"{phase}: {err}"
    return "validation failed (no error text)"


def build_followup_diagnose_prompt(
    *,
    bench: str,
    top_function: str,
    hls_code: str,
    error_text: str,
    repair_variant: str,
    loop_index: int,
    max_loops: int,
    prior_diagnosis: str,
) -> str:
    skills = _load_skills_block(repair_variant)
    skills_section = f"\n{skills}\n" if skills else ""
    return f"""You are an expert Xilinx Vitis HLS engineer. A prior cosim repair attempt still failed validation.

Benchmark: {bench}
Top function: {top_function}
Repair attempt: {loop_index} of {max_loops}

Latest validation error (truncated):
{error_text[:4000]}

Previous diagnosis:
{prior_diagnosis[:3000]}

Current code (after the last repair):
```cpp
{hls_code}
```
{skills_section}
Re-analyze why validation still fails. The prior fix was insufficient or introduced a new bug.

Do NOT output code yet. Respond with:
1. **Root cause** — what is still wrong
2. **Specific bugs** — bullet list with line/loop references
3. **Revised fix plan** — numbered steps for the next repair attempt
"""


def build_followup_repair_prompt(
    *,
    bench: str,
    top_function: str,
    hls_code: str,
    error_text: str,
    diagnosis: str,
    repair_variant: str,
    loop_index: int,
    max_loops: int,
) -> str:
    skills = _load_skills_block(repair_variant)
    skills_section = f"\n{skills}\n" if skills else ""
    return f"""Apply the revised fix plan so the kernel passes Vitis HLS cosim.

Benchmark: {bench}
Top function: {top_function} (keep this symbol name)
Repair attempt: {loop_index} of {max_loops}

Latest validation error:
{error_text[:2000]}

Revised diagnosis and fix plan:
{diagnosis}

Current code:
```cpp
{hls_code}
```
{skills_section}
Output the COMPLETE fixed C++ translation unit in a single ```cpp fenced block.
Preserve `extern "C"` and the top function `{top_function}` unless the testbench requires otherwise.
Focus on functional correctness first.
"""


def call_llm(messages: list[dict[str, str]], *, model: Optional[str] = None) -> str:
    from openai import OpenAI

    model = model or os.getenv("C2HLS_MODEL", "").strip() or os.getenv("PC2_LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError("C2HLS_MODEL or PC2_LLM_MODEL must be set")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL must be set (PC2 vLLM endpoint)")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    timeout = float(os.getenv("C2HLS_LLM_TIMEOUT", "900"))
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
        kwargs["max_completion_tokens"] = int(os.getenv("C2HLS_LLM_MAX_TOKENS", "8192"))
    else:
        kwargs["max_tokens"] = int(os.getenv("C2HLS_LLM_MAX_TOKENS", "8192"))
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def session_run_root(run_root: Path, session_key: str) -> Path:
    path = run_root / session_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_session_config(session_key: str) -> dict[str, str]:
    if session_key not in REPAIR_SESSIONS:
        raise KeyError(f"unknown repair session: {session_key}")
    return REPAIR_SESSIONS[session_key]


def discover_failures_for_artifact(
    artifact_basename: str,
    cosim_run_root: Path = DEFAULT_COSIM_RUN,
) -> list[dict[str, Any]]:
    """Load cosim failures for one championship artifact."""
    cells_dir = cosim_run_root / "cells"
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"missing cosim cells dir: {cells_dir}")

    key_by_basename = {basename: key for key, basename in CHAMPIONSHIP_ARTIFACTS}
    failures: list[dict[str, Any]] = []
    for result_path in sorted(cells_dir.glob("*/cosim_result.json")):
        result = json.loads(result_path.read_text())
        if result.get("passed"):
            continue
        prov = result.get("provenance") or {}
        if prov.get("artifact_basename") != artifact_basename:
            continue
        failures.append(
            {
                "source_cell_id": prov.get("cell_id", result_path.parent.name),
                "source_artifact_basename": artifact_basename,
                "source_artifact_key": key_by_basename.get(artifact_basename, ""),
                "bench": prov.get("bench", ""),
                "setup_tag": prov.get("setup_tag", ""),
                "cell_dir": prov.get("cell_dir", ""),
                "final_cpp": prov.get("final_cpp", ""),
                "cosim_error": result.get("error") or "",
                "source_cosim_work_dir": result.get("work_dir") or "",
                "matrix_family": prov.get("matrix_family", ""),
                "variant": prov.get("variant", ""),
                "mode": prov.get("mode", ""),
                "source_cosim_result": str(result_path),
            }
        )
    failures.sort(key=lambda row: row["bench"])
    return failures


def discover_championship_failures(
    cosim_run_root: Path = DEFAULT_COSIM_RUN,
) -> list[dict[str, Any]]:
    """Load failed cosim cells from championship artifact dirs only."""
    cells_dir = cosim_run_root / "cells"
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"missing cosim cells dir: {cells_dir}")

    key_by_basename = {basename: key for key, basename in CHAMPIONSHIP_ARTIFACTS}
    failures: list[dict[str, Any]] = []
    for result_path in sorted(cells_dir.glob("*/cosim_result.json")):
        result = json.loads(result_path.read_text())
        if result.get("passed"):
            continue
        prov = result.get("provenance") or {}
        artifact = prov.get("artifact_basename", "")
        if artifact not in CHAMPIONSHIP_BASENAMES:
            continue
        failures.append(
            {
                "source_cell_id": prov.get("cell_id", result_path.parent.name),
                "source_artifact_basename": artifact,
                "source_artifact_key": key_by_basename.get(artifact, ""),
                "bench": prov.get("bench", ""),
                "setup_tag": prov.get("setup_tag", ""),
                "cell_dir": prov.get("cell_dir", ""),
                "final_cpp": prov.get("final_cpp", ""),
                "cosim_error": result.get("error") or "",
                "source_cosim_work_dir": result.get("work_dir") or "",
                "matrix_family": prov.get("matrix_family", ""),
                "variant": prov.get("variant", ""),
                "mode": prov.get("mode", ""),
                "source_cosim_result": str(result_path),
            }
        )
    failures.sort(key=lambda row: (row["source_artifact_basename"], row["bench"]))
    return failures


def build_session_jobs(
    failures: list[dict[str, Any]],
    *,
    repair_variant: str,
    session_key: str,
) -> list[RepairJob]:
    jobs: list[RepairJob] = []
    for index, row in enumerate(failures):
        job_id = row["source_cell_id"]
        jobs.append(
            RepairJob(
                index=index,
                job_id=job_id,
                repair_variant=repair_variant,
                source_cell_id=row["source_cell_id"],
                source_artifact_basename=row["source_artifact_basename"],
                source_artifact_key=row.get("source_artifact_key", session_key),
                bench=row["bench"],
                setup_tag=row["setup_tag"],
                cell_dir=row["cell_dir"],
                final_cpp=row["final_cpp"],
                cosim_error=row["cosim_error"],
                source_cosim_work_dir=row["source_cosim_work_dir"],
                matrix_family=row["matrix_family"],
                variant=row["variant"],
                mode=row["mode"],
            )
        )
    return jobs


def build_repair_jobs(
    failures: list[dict[str, Any]],
) -> list[RepairJob]:
    jobs: list[RepairJob] = []
    index = 0
    for row in failures:
        for repair_variant in REPAIR_VARIANTS:
            job_id = make_job_id(row["source_cell_id"], repair_variant)
            jobs.append(
                RepairJob(
                    index=index,
                    job_id=job_id,
                    repair_variant=repair_variant,
                    source_cell_id=row["source_cell_id"],
                    source_artifact_basename=row["source_artifact_basename"],
                    source_artifact_key=row["source_artifact_key"],
                    bench=row["bench"],
                    setup_tag=row["setup_tag"],
                    cell_dir=row["cell_dir"],
                    final_cpp=row["final_cpp"],
                    cosim_error=row["cosim_error"],
                    source_cosim_work_dir=row["source_cosim_work_dir"],
                    matrix_family=row["matrix_family"],
                    variant=row["variant"],
                    mode=row["mode"],
                )
            )
            index += 1
    return jobs


def write_session_manifest(
    session_root: Path,
    session_key: str,
    jobs: list[RepairJob],
    *,
    cosim_run_root: Path,
) -> Path:
    cfg = get_session_config(session_key)
    payload = {
        "schema": "flash_cosim_repair_session_v1",
        "session_key": session_key,
        "session_id": cfg["session_id"],
        "label": cfg["label"],
        "artifact_basename": cfg["artifact_basename"],
        "repair_variant": cfg["repair_variant"],
        "cosim_run_root": str(cosim_run_root),
        "session_root": str(session_root),
        "failure_count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }
    path = session_root / "session_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_run_manifest(
    run_root: Path,
    *,
    cosim_run_root: Path,
    sessions: dict[str, list[RepairJob]],
) -> Path:
    payload = {
        "schema": "flash_cosim_repair_run_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repair_run_root": str(run_root),
        "cosim_run_root": str(cosim_run_root),
        "session_count": len(sessions),
        "total_jobs": sum(len(j) for j in sessions.values()),
        "sessions": {
            key: {
                **get_session_config(key),
                "session_root": str(session_run_root(run_root, key)),
                "failure_count": len(jobs),
                "jobs": [j.to_dict() for j in jobs],
            }
            for key, jobs in sessions.items()
        },
    }
    path = manifest_path(run_root)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _array_slices(job_count: int) -> dict[str, str]:
    """Deprecated — session batch mode does not use Slurm arrays."""
    return {}


def write_manifest(
    run_root: Path,
    jobs: list[RepairJob],
    *,
    cosim_run_root: Path,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema": "flash_cosim_repair_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repair_run_root": str(run_root),
        "cosim_run_root": str(cosim_run_root),
        "championship_artifacts": [
            {"key": key, "basename": basename} for key, basename in CHAMPIONSHIP_ARTIFACTS
        ],
        "repair_variants": list(REPAIR_VARIANTS),
        "source_failure_count": len({j.source_cell_id for j in jobs}),
        "job_count": len(jobs),
        "array_slices": _array_slices(len(jobs)),
        "jobs": [job.to_dict() for job in jobs],
    }
    if extra:
        payload.update(extra)
    path = manifest_path(run_root)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(run_root: Path) -> dict[str, Any]:
    return json.loads(manifest_path(run_root).read_text())


def find_job(manifest: dict[str, Any], index: int) -> RepairJob:
    for raw in manifest.get("jobs", []):
        if int(raw.get("index", -1)) == index:
            return RepairJob(**raw)
    raise KeyError(f"repair manifest index not found: {index}")


def validate_repaired_code(
    bench: str,
    hls_code: str,
    *,
    repair_tag: str,
) -> dict[str, Any]:
    from scripts.pc2.flash_cosim_lib import load_cosim_inputs

    import hls_eval

    bench_dir = REPO / "benchmarks" / bench
    inputs = load_cosim_inputs(bench_dir)
    top = inputs["top_function"]

    with temp_tag_scope(bench, repair_tag, "repair_val"):
        csim = hls_eval.run_csim(
            hls_code,
            inputs["testbench_code"],
            inputs["header_code"],
            header_name=inputs["header_name"],
            top_function=top,
            part=inputs["part"],
            clock_ns=inputs["clock_ns"],
            extra_files=inputs["extra_files"],
        )
        synth = hls_eval.run_hls_synthesis(
            hls_code,
            inputs["header_code"],
            header_name=inputs["header_name"],
            top_function=top,
            part=inputs["part"],
            clock_ns=inputs["clock_ns"],
            extra_files=inputs["extra_files"],
        )
        cosim = None
        if synth.get("success"):
            cosim = hls_eval.run_cosim(
                hls_code,
                inputs["testbench_code"],
                inputs["header_code"],
                header_name=inputs["header_name"],
                top_function=top,
                part=inputs["part"],
                clock_ns=inputs["clock_ns"],
                extra_files=inputs["extra_files"],
                interface_depths=inputs["cosim_depths"],
            )

    return {
        "csim": {
            "success": bool(csim.get("success")),
            "passed": bool(csim.get("passed")),
            "error": (csim.get("error") or "")[:500],
        },
        "synth": {
            "success": bool(synth.get("success")),
            "error": (synth.get("error") or "")[:500],
            "latency_cycles": (synth.get("report") or {}).get("latency_cycles"),
        },
        "cosim": {
            "success": bool(cosim and cosim.get("success")),
            "passed": bool(cosim and cosim.get("passed")),
            "error": (cosim.get("error") or "")[:500] if cosim else "skipped (synth failed)",
            "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles") if cosim else None,
        },
    }


def _execute_repair_loop(
    *,
    loop_index: int,
    max_loops: int,
    bench: str,
    top: str,
    hls_code: str,
    error_text: str,
    repair_variant: str,
    loop_out: Path,
    prior_diagnosis: str = "",
) -> dict[str, Any]:
    loop_out.mkdir(parents=True, exist_ok=True)
    (loop_out / "input.cpp").write_text(hls_code, encoding="utf-8")
    (loop_out / "error.txt").write_text(error_text, encoding="utf-8")

    t0 = time.time()
    if loop_index == 1:
        diagnose_prompt = build_diagnose_prompt(
            bench=bench,
            top_function=top,
            hls_code=hls_code,
            cosim_error=error_text,
            repair_variant=repair_variant,
        )
    else:
        diagnose_prompt = build_followup_diagnose_prompt(
            bench=bench,
            top_function=top,
            hls_code=hls_code,
            error_text=error_text,
            repair_variant=repair_variant,
            loop_index=loop_index,
            max_loops=max_loops,
            prior_diagnosis=prior_diagnosis,
        )
    (loop_out / "diagnose_prompt.txt").write_text(diagnose_prompt, encoding="utf-8")
    diagnosis = call_llm([{"role": "user", "content": diagnose_prompt}])
    (loop_out / "diagnose_response.txt").write_text(diagnosis, encoding="utf-8")

    if loop_index == 1:
        repair_prompt = build_repair_prompt(
            bench=bench,
            top_function=top,
            hls_code=hls_code,
            cosim_error=error_text,
            diagnosis=diagnosis,
            repair_variant=repair_variant,
        )
    else:
        repair_prompt = build_followup_repair_prompt(
            bench=bench,
            top_function=top,
            hls_code=hls_code,
            error_text=error_text,
            diagnosis=diagnosis,
            repair_variant=repair_variant,
            loop_index=loop_index,
            max_loops=max_loops,
        )
    (loop_out / "repair_prompt.txt").write_text(repair_prompt, encoding="utf-8")
    repair_reply = call_llm([{"role": "user", "content": repair_prompt}])
    (loop_out / "repair_response.txt").write_text(repair_reply, encoding="utf-8")

    repaired = extract_cpp_code(repair_reply)
    if not repaired:
        return {
            "loop": loop_index,
            "status": "fail",
            "phase": "repair_extract",
            "error": "LLM repair turn did not return a ```cpp code block",
            "diagnosis": diagnosis,
            "runtime_seconds": round(time.time() - t0, 3),
        }

    (loop_out / "repaired.cpp").write_text(repaired, encoding="utf-8")
    repair_tag = join_temp_tag(bench, repair_variant, f"repair_l{loop_index:02d}")
    validation = validate_repaired_code(bench, repaired, repair_tag=repair_tag)
    (loop_out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    cosim_ok = bool(validation.get("cosim", {}).get("passed"))
    return {
        "loop": loop_index,
        "status": "ok" if cosim_ok else "fail",
        "cosim_passed": cosim_ok,
        "csim_passed": validation.get("csim", {}).get("passed"),
        "synth_passed": validation.get("synth", {}).get("success"),
        "validation": validation,
        "diagnosis": diagnosis,
        "repaired_cpp": repaired,
        "runtime_seconds": round(time.time() - t0, 3),
    }


def run_repair_job(
    job: RepairJob,
    run_root: Path,
    *,
    max_loops: int = DEFAULT_MAX_REPAIR_LOOPS,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if max_loops < 1:
        raise ValueError(f"max_loops must be >= 1, got {max_loops}")

    out = job_dir(run_root, job.job_id)
    result_path = out / "repair_result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text())

    out.mkdir(parents=True, exist_ok=True)
    source_cpp = Path(job.final_cpp).read_text()
    (out / "source_final.cpp").write_text(source_cpp, encoding="utf-8")

    from scripts.pc2.flash_cosim_lib import load_cosim_inputs

    bench_dir = REPO / "benchmarks" / job.bench
    inputs = load_cosim_inputs(bench_dir)
    top = inputs["top_function"]

    provenance = {
        **job.to_dict(),
        "repair_run_root": str(run_root),
        "skills_json": str(NEW_SKILLS_JSON),
        "max_loops": max_loops,
    }

    if dry_run:
        result = {
            "status": "dry_run",
            "provenance": provenance,
            "top_function": top,
            "max_loops": max_loops,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    t0 = time.time()
    current_code = source_cpp
    error_text = job.cosim_error
    prior_diagnosis = ""
    loop_results: list[dict[str, Any]] = []
    cosim_ok = False

    for loop_index in range(1, max_loops + 1):
        loop_out = repair_loop_out_dir(run_root, job.job_id, loop_index, max_loops=max_loops)
        loop_row = _execute_repair_loop(
            loop_index=loop_index,
            max_loops=max_loops,
            bench=job.bench,
            top=top,
            hls_code=current_code,
            error_text=error_text,
            repair_variant=job.repair_variant,
            loop_out=loop_out,
            prior_diagnosis=prior_diagnosis,
        )
        loop_results.append(loop_row)

        if loop_row.get("phase") == "repair_extract":
            break

        current_code = loop_row.get("repaired_cpp", current_code)
        prior_diagnosis = loop_row.get("diagnosis", prior_diagnosis)
        cosim_ok = bool(loop_row.get("cosim_passed"))
        if cosim_ok:
            break
        error_text = validation_error_text(loop_row.get("validation") or {})

    loops_used = len(loop_results)
    if max_loops == 1 and loop_results:
        single = loop_results[0]
        if single.get("phase") == "repair_extract":
            result = {
                "status": "fail",
                "phase": "repair_extract",
                "error": single.get("error"),
                "provenance": provenance,
                "runtime_seconds": round(time.time() - t0, 3),
            }
        else:
            result = {
                "status": single.get("status", "fail"),
                "cosim_passed": bool(single.get("cosim_passed")),
                "csim_passed": single.get("csim_passed"),
                "synth_passed": single.get("synth_passed"),
                "validation": single.get("validation"),
                "provenance": provenance,
                "runtime_seconds": round(time.time() - t0, 3),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            if single.get("repaired_cpp"):
                (out / "repaired.cpp").write_text(single["repaired_cpp"], encoding="utf-8")
                (out / "validation.json").write_text(
                    json.dumps(single.get("validation") or {}, indent=2) + "\n",
                    encoding="utf-8",
                )
    else:
        final_loop = loop_results[-1] if loop_results else {}
        result = {
            "status": "ok" if cosim_ok else "fail",
            "cosim_passed": cosim_ok,
            "csim_passed": final_loop.get("csim_passed"),
            "synth_passed": final_loop.get("synth_passed"),
            "validation": final_loop.get("validation"),
            "max_loops": max_loops,
            "loops_used": loops_used,
            "loops": loop_results,
            "provenance": provenance,
            "runtime_seconds": round(time.time() - t0, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        if final_loop.get("repaired_cpp"):
            (out / "repaired.cpp").write_text(final_loop["repaired_cpp"], encoding="utf-8")
            (out / "validation.json").write_text(
                json.dumps(final_loop.get("validation") or {}, indent=2) + "\n",
                encoding="utf-8",
            )

    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_repair_batch(
    session_key: str,
    run_root: Path,
    cosim_run_root: Path,
    *,
    max_loops: int = DEFAULT_MAX_REPAIR_LOOPS,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repair all cosim failures for one session (sequential, one GPU+compute pair)."""
    cfg = get_session_config(session_key)
    session_root = session_run_root(run_root, session_key)
    failures = discover_failures_for_artifact(cfg["artifact_basename"], cosim_run_root)
    jobs = build_session_jobs(
        failures,
        repair_variant=cfg["repair_variant"],
        session_key=session_key,
    )
    write_session_manifest(session_root, session_key, jobs, cosim_run_root=cosim_run_root)

    if dry_run:
        return {
            "session_key": session_key,
            "label": cfg["label"],
            "artifact_basename": cfg["artifact_basename"],
            "repair_variant": cfg["repair_variant"],
            "max_loops": max_loops,
            "total": len(jobs),
            "cosim_passed": 0,
            "cosim_failed": len(jobs),
            "dry_run": True,
            "benches": [job.bench for job in jobs],
        }

    results: list[dict[str, Any]] = []
    for job in jobs:
        row = run_repair_job(
            job,
            session_root,
            max_loops=max_loops,
            force=force,
            dry_run=False,
        )
        results.append({"job_id": job.job_id, "bench": job.bench, **row})

    ok = sum(1 for r in results if r.get("cosim_passed"))
    summary = {
        "session_key": session_key,
        "label": cfg["label"],
        "artifact_basename": cfg["artifact_basename"],
        "repair_variant": cfg["repair_variant"],
        "max_loops": max_loops,
        "total": len(jobs),
        "cosim_passed": ok,
        "cosim_failed": len(jobs) - ok,
        "dry_run": dry_run,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    (session_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
