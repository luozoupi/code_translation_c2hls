"""Post-flash DSE step: LLM nest rewrite into compact multi-PE x SIMD GEMM.

Runs **after** flash succeeds (csim + csynth). Unlike pragma-opt, this step
**may rewrite the compute nest**. It must keep the top-level ABI and INTERFACE
pragmas. Skills live in ``post_flash_dse_pe_skill_entries.json`` and are not
mixed into the flash 84/90-skill dump.

Validation: compile + csim + csynth (cosim off). Architecture gate: DSP must
exceed a few units (flash leftover) or the attempt is treated as a miss.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from autosa_flow_gates import compute_architecture_ok
from c2hls_paths import POST_FLASH_DSE_SKILL_ENTRIES_JSON
from flash_flow_artifacts import sha256_text
from post_flash_dataflow import extract_kernel_block, sanitize_kernel_source
from post_flash_mem_parallel import discover_matrix_cells, resolve_selected_kernel
from post_flash_pragma_opt import resolve_source_kernel, summarize_synth_report
from post_flash_pe_recipe import format_recipe_prompt, recipe_for
from skill_library import (
    Skill,
    _coerce_skill_entry,
    render_skill_set_for_prompt_full,
)

STEP_TAG = "dse"
DEFAULT_REPAIR_ROUNDS = 4
DEFAULT_MIN_DSP = 32
DEFAULT_MAX_TOKENS = 65536
DEFAULT_EMPTY_REPLY_RETRIES = 3

_LOG = logging.getLogger(__name__)

_SYSTEM = """You are an expert Xilinx Vitis HLS 2023.2 engineer running a **DSE** step after flash.

Flash already produced a **working** load-compute-store kernel (csim + csynth pass).
Your job is **not** pragma-only tuning and **not** cloning an AutoSA PE/IO netlist.
Rewrite the **compute nest** into a compact multi-PE GEMM with latency hiding.

## Architecture (mandatory)
1. Keep the exact top-level `extern "C"` signature, parameter list, array ranks/shapes, and every existing `#pragma HLS INTERFACE` line.
2. Keep bulk on-chip staging (load A/B/C, compute, store C). No m_axi RMW inside compute.
3. Tile rows by **PE** from the mandatory recipe (not blindly 16). Use I_P/J_P/K_P when kernel.h defines them.
4. **Pipeline the independent j loop** (C columns) at II=1 — this is latency hiding.
5. Fully **UNROLL** the PE loop inside that pipelined j loop. Each PE has private `Crow[p][j]`.
6. Factor k by **SIMD** from the recipe. Fully UNROLL SIMD into an adder tree: `partial = a0*b0+...` then `Crow[p][j] += partial`.
7. Match ARRAY_PARTITION to PE/SIMD. Do not complete-partition the full `C[I][J]`.
8. Index B using the existing layout (`B[J][K]` → `local_B[j][k]`; `B[K][J]` → `local_B[k][j]`).

## Forbidden
- Changing top-level ports, bundles, or function name.
- Pipelining a k-loop that updates one `C[i][j]` / `crow[j]` (flash leftover; DSP stays ~3–8, II=4).
- Emitting AutoSA `kernel0`, PE/IO interconnect, or `hls::stream` of PE structs. Do not emit AutoSA netlists.
- `malloc`, system calls, or non-synthesizable constructs.

## Success check before you submit
- Pipelined compute loop is `compute_j` (independent columns), not `compute_k`.
- PE UNROLL + SIMD UNROLL are present.
- Expected DSP matches the recipe (float PE×SIMD×5; uint16 PE×SIMD; uint32 ~PE×SIMD×3). Single-digit DSP means you still have the flash nest — revise.

## Output
Start with ```kernel immediately. Close the fence. Do not spend the token budget on analysis.
```kernel
... full kernel source ...
```
"""

_INITIAL_USER = """{seed_intro}

Keep the top signature and INTERFACE pragmas. Return a single ```kernel``` block.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## {seed_heading}
```cpp
{kernel_code}
```

## Prior csynth summary
{synth_summary}
{skills_section}"""

_SEED_INTRO = {
    "flash_final": (
        "Run the **DSE** step on this flash-final kernel.\n\n"
        "Flash already passed csim + csynth. Rewrite the compute nest{skills_clause}."
    ),
    "baseline": (
        "Run the **multi-PE** step on the autosa_mm **HLS baseline**. Flash was **not** run.\n\n"
        "The seed is the naive triple loop (`hls_baseline.cpp` / `plain.cpp`). "
        "Add legal `#pragma HLS INTERFACE` if missing. Rewrite the compute nest{skills_clause}."
    ),
}

_SEED_HEADING = {
    "flash_final": "Flash kernel (seed)",
    "baseline": "Baseline kernel (seed, no flash)",
}

_SYSTEM_BASELINE = (
    _SYSTEM.replace(
        "running a **DSE** step after flash.",
        "running a **multi-PE** rewrite on the HLS baseline (flash skipped).",
    ).replace(
        "Flash already produced a **working** load-compute-store kernel (csim + csynth pass).\n"
        "Your job is **not** pragma-only tuning and **not** cloning an AutoSA PE/IO netlist.\n"
        "Rewrite the **compute nest** into a compact multi-PE GEMM with latency hiding.",
        "The seed is the **HLS baseline** (naive ijk GEMM). Flash was skipped. "
        "The nest may have no INTERFACE pragmas and is not a PE array.\n"
        "Your job is **not** pragma-only tuning and **not** cloning an AutoSA PE/IO netlist.\n"
        "Rewrite the **compute nest** into a compact multi-PE GEMM with latency hiding.",
    )
)

_REPAIR_USER = """Repair the **DSE** kernel after a validation or architecture failure.

Keep the exact top-level signature and `#pragma HLS INTERFACE` pragmas.
Keep the PE×SIMD from the recipe below; pipeline independent `j`; do not fall back to flash `crow[j] +=` over k.
Start the reply with ```kernel and close the fence. Do not spend the token budget on analysis.

{recipe_block}

## Failure
Stage: {stage}
```
{error}
```

## Header ({header_name})
```cpp
{header_code}
```

## Current / seed kernel
```cpp
{kernel_code}
```

Return a corrected single ```kernel``` block.
"""


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "on"}


def post_flash_no_skills() -> bool:
    return _env_flag("C2HLS_POST_FLASH_NO_SKILLS") is True


def dse_source_role() -> str:
    raw = os.getenv("C2HLS_DSE_SOURCE_ROLE", "flash_final").strip().lower()
    if raw in {"baseline", "plain", "hls_baseline"}:
        return "baseline"
    return "flash_final"


def dse_enabled() -> bool:
    return _env_flag("C2HLS_POST_FLASH_DSE") is True


def chain_after_flash() -> bool:
    chained = _env_flag("C2HLS_DSE_CHAIN_FLASH")
    if chained is not None:
        return chained
    return dse_enabled()


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_DSE_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def dse_max_tokens() -> int:
    """Completion budget. Flash uses >=16384; 8192 left DeepSeek-v4-flash empty."""
    for key in ("C2HLS_DSE_MAX_TOKENS", "C2HLS_FLASH_MAX_TOKENS", "C2HLS_LLM_MAX_TOKENS"):
        raw = os.getenv(key, "").strip()
        if raw.isdigit():
            return max(int(raw), 8192)
    return DEFAULT_MAX_TOKENS


def empty_reply_retry_limit() -> int:
    try:
        return max(
            1,
            int(os.getenv("C2HLS_LLM_EMPTY_RETRIES", str(DEFAULT_EMPTY_REPLY_RETRIES))),
        )
    except ValueError:
        return DEFAULT_EMPTY_REPLY_RETRIES


def _invoke_llm(orchestrator: Any, messages: list[dict[str, str]], max_tokens: int) -> str:
    try:
        raw = orchestrator._call_llm(messages, max_tokens=max_tokens)
    except TypeError:
        raw = orchestrator._call_llm(messages)
    if raw is None:
        return ""
    return raw if isinstance(raw, str) else str(raw)


def call_dse_llm(
    orchestrator: Any,
    messages: list[dict[str, str]],
    *,
    purpose: str = "dse",
) -> str:
    """Call the LLM with the DSE token floor; retry empty/length-starved replies."""
    from post_flash_dataflow import is_empty_llm_reply

    tokens = dse_max_tokens()
    retries = empty_reply_retry_limit()
    last = ""
    for attempt in range(retries):
        last = _invoke_llm(orchestrator, messages, tokens)
        if not is_empty_llm_reply(last):
            if attempt:
                _LOG.info(
                    "[%s] non-empty LLM reply on retry %d/%d (len=%d, max_tokens=%d)",
                    purpose,
                    attempt + 1,
                    retries,
                    len(last),
                    tokens,
                )
            return last
        _LOG.warning(
            "[%s] empty LLM reply attempt %d/%d max_tokens=%d; retrying",
            purpose,
            attempt + 1,
            retries,
            tokens,
        )
    return last


def min_dsp_for_architecture(bench: Optional[str] = None) -> int:
    raw = os.getenv("C2HLS_DSE_MIN_DSP", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if bench:
        return recipe_for(bench).min_dsp
    return DEFAULT_MIN_DSP


# A 13160-cycle kernel with DSP 352 is a compute PASS; I/O is the next step.
def architecture_ok(report: Optional[dict[str, Any]], bench: Optional[str] = None) -> bool:
    if not report:
        return False
    min_dsp = min_dsp_for_architecture(bench)
    _LOG.info(compute_architecture_ok(report, min_dsp=min_dsp).reason)
    try:
        dsp = int(report.get("dsp") or 0)
    except (TypeError, ValueError):
        return False
    return dsp >= min_dsp


def resolve_dse_skills_path() -> Path:
    raw = os.getenv("C2HLS_DSE_SKILL_ENTRIES_JSON", "").strip()
    if raw:
        return Path(raw)
    return POST_FLASH_DSE_SKILL_ENTRIES_JSON


def validate_dse_skill_entries(path: Path) -> list[str]:
    if not path.is_file():
        return [f"missing skill file: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in {path.name}: {exc}"]
    if not isinstance(data, dict):
        return [f"{path.name}: root must be a JSON object"]
    skills_raw = data.get("skills")
    if not isinstance(skills_raw, list) or not skills_raw:
        return [f"{path.name}: skills must be a non-empty array"]
    errors: list[str] = []
    for entry in skills_raw:
        if _coerce_skill_entry(entry) is None:
            sid = entry.get("id") if isinstance(entry, dict) else "?"
            errors.append(f"{path.name}: malformed skill entry {sid}")
    return errors


def load_dse_skills(path: Optional[Path] = None) -> list[Skill]:
    skill_path = path or resolve_dse_skills_path()
    errors = validate_dse_skill_entries(skill_path)
    if errors:
        raise ValueError("; ".join(errors))
    data = json.loads(skill_path.read_text(encoding="utf-8"))
    skills: list[Skill] = []
    for entry in data["skills"]:
        skill = _coerce_skill_entry(entry)
        if skill is not None:
            skills.append(skill)
    return skills


def build_dse_skills_prompt_block(
    path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    if post_flash_no_skills():
        return "", {"skills_path": "", "skill_count": 0, "skill_ids": []}
    skill_path = path or resolve_dse_skills_path()
    skills = load_dse_skills(skill_path)
    block = render_skill_set_for_prompt_full(skills)
    meta: dict[str, Any] = {
        "skills_path": str(skill_path),
        "skill_count": len(skills),
        "skill_ids": [sk.id for sk in skills],
    }
    return block, meta


def format_dse_initial_user(
    *,
    skills_block: str,
    benchmark_context: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
    synth_summary: str,
    source_role: str = "flash_final",
) -> str:
    block = (skills_block or "").strip()
    if block:
        skills_clause = " using the DSE skills below"
        skills_section = (
            "\n## DSE skills (follow these; copy the PE template)\n" + block + "\n"
        )
    else:
        skills_clause = ""
        skills_section = ""
    role = source_role if source_role in _SEED_INTRO else "flash_final"
    seed_intro = _SEED_INTRO[role].format(skills_clause=skills_clause)
    return _INITIAL_USER.format(
        seed_intro=seed_intro,
        seed_heading=_SEED_HEADING[role],
        skills_section=skills_section,
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code,
        kernel_code=kernel_code,
        synth_summary=synth_summary,
    )


def resolve_baseline_kernel(cell_dir: Path, bench: str, bench_dir: Path) -> Optional[Path]:
    for path in (
        cell_dir / f"{bench}_baseline.cpp",
        bench_dir / "hls_baseline.cpp",
        bench_dir / "plain.cpp",
    ):
        if path.is_file():
            return path
    return None


def artifact_paths(cell_dir: Path, bench: str) -> dict[str, Path]:
    base = f"{bench}_{STEP_TAG}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
    }


def load_existing_result(result_path: Path) -> Optional[dict[str, Any]]:
    if not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("success") is True:
        return data
    return None


def discover_dse_cells(matrix_root: Path) -> list[dict[str, Any]]:
    """Find flash cells under a campaign root (matrix.json or *_selected.cpp)."""
    cells = discover_matrix_cells(matrix_root)
    if cells:
        return cells
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for selected in sorted(matrix_root.rglob("*_selected.cpp")):
        if not selected.is_file():
            continue
        name = selected.name
        if not name.endswith("_selected.cpp"):
            continue
        bench = name[: -len("_selected.cpp")]
        cell_dir = selected.parent
        if cell_dir in seen:
            continue
        seen.add(cell_dir)
        found.append({
            "bench": bench,
            "cell_dir": cell_dir,
            "status": "unknown",
            "model": "",
        })
    return found


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def should_promote_dse(
    *,
    success: bool,
    latency_cycles: Any,
    baseline_latency: Any,
    dsp: Any,
    bench: Optional[str] = None,
) -> bool:
    if not success:
        return False
    lat = _as_int(latency_cycles)
    dsp_n = _as_int(dsp)
    if lat is None or dsp_n is None:
        return False
    if dsp_n < min_dsp_for_architecture(bench):
        return False
    baseline = _as_int(baseline_latency)
    if baseline is None:
        return True
    return lat < baseline


def promote_dse_as_selected(
    *,
    cell_dir: Path,
    bench: str,
    code: str,
    report: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Overwrite selected pointers with a passing DSE kernel; keep flash seed."""
    paths = artifact_paths(cell_dir, bench)
    lat = result_payload.get("latency_cycles")
    promotion: dict[str, Any] = {
        "source_role": "flash_final",
        "latency_cycles": lat,
        "kernel": paths["kernel"].name,
        "selected_stage": "dse",
    }
    selected_cpp = cell_dir / f"{bench}_selected.cpp"
    selected_report = cell_dir / f"{bench}_selected_report.json"
    seed_cpp = cell_dir / f"{bench}_flash_seed.cpp"
    seed_report = cell_dir / f"{bench}_flash_seed_report.json"
    if not seed_cpp.is_file():
        if selected_cpp.is_file() and selected_cpp.read_text(encoding="utf-8").strip():
            shutil.copy2(selected_cpp, seed_cpp)
            if selected_report.is_file():
                shutil.copy2(selected_report, seed_report)
        else:
            final_cpp = cell_dir / f"{bench}_final.cpp"
            if final_cpp.is_file() and final_cpp.read_text(encoding="utf-8").strip():
                shutil.copy2(final_cpp, seed_cpp)
        if seed_cpp.is_file():
            promotion["flash_seed_kernel"] = seed_cpp.name
            if seed_report.is_file():
                promotion["flash_seed_report"] = seed_report.name
    selected_cpp.write_text(code, encoding="utf-8")
    selected_report.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    promotion["selected_kernel"] = selected_cpp.name
    promotion["selected_report"] = selected_report.name

    manifest_path = cell_dir / f"{bench}_flow_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except json.JSONDecodeError:
            manifest = {}
    lat_map = manifest.get("latency_cycles")
    if not isinstance(lat_map, dict):
        lat_map = {}
    lat_map = dict(lat_map)
    if lat is not None:
        lat_map["selected"] = lat
        lat_map["dse"] = lat
    manifest["latency_cycles"] = lat_map
    manifest["selected_from"] = "dse"
    manifest["selected_sha256"] = sha256_text(code)
    if "files" not in manifest or not isinstance(manifest["files"], dict):
        manifest["files"] = {}
    files = dict(manifest["files"])
    files["selected"] = selected_cpp.name
    files["selected_report"] = selected_report.name
    files["dse"] = paths["kernel"].name
    files["dse_report"] = paths["report"].name
    manifest["files"] = files
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    promotion["flow_manifest"] = manifest_path.name
    return promotion


@dataclass
class DseOutcome:
    bench: str
    source_role: str
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def run_dse_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> DseOutcome:
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag

    if source_role not in {"flash_final", "baseline"}:
        return DseOutcome(bench, source_role, False, str(cell_dir), "dse is flash-final or baseline only")

    prior_report: Optional[dict[str, Any]] = None
    if source_role == "baseline":
        kernel_path = resolve_baseline_kernel(cell_dir, bench, bench_dir)
        kernel_role = "baseline"
    else:
        kernel_path, kernel_role, prior_report = resolve_source_kernel(
            cell_dir, bench, "flash_final"
        )
    if kernel_path is None:
        return DseOutcome(
            bench,
            source_role,
            False,
            str(cell_dir),
            "no selected/final kernel cpp" if source_role == "flash_final" else "no baseline/plain.cpp",
        )

    inputs = _load_benchmark_inputs(str(bench_dir))
    source_kernel = kernel_path.read_text(encoding="utf-8")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or "workload"
    )
    benchmark_context = format_recipe_prompt(bench, step="dse") + (
        inputs.get("benchmark_context", "") or ""
    )
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    paths = artifact_paths(cell_dir, bench)
    if skip_existing:
        existing = load_existing_result(paths["result"])
        if existing is not None:
            return DseOutcome(bench, source_role, True, str(cell_dir), result=existing)

    skills_block, skills_meta = build_dse_skills_prompt_block()
    synth_summary = summarize_synth_report(prior_report)
    baseline_latency = (prior_report or {}).get("latency_cycles")
    token_floor = dse_max_tokens()
    current = getattr(orchestrator, "max_completion_tokens", 0) or 0
    if current < token_floor:
        orchestrator.max_completion_tokens = token_floor

    history: list[dict[str, str]] = []
    attempts: list[dict[str, Any]] = []
    kernel_code = ""
    success = False
    last_error = ""
    synth_report: dict[str, Any] = {}
    csim_summary: Optional[dict[str, Any]] = None

    system = _SYSTEM_BASELINE if source_role == "baseline" else _SYSTEM
    user = format_dse_initial_user(
        skills_block=skills_block,
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code[:12000],
        kernel_code=source_kernel[:120000],
        synth_summary=synth_summary,
        source_role=source_role,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reply = call_dse_llm(orchestrator, messages, purpose="dse_initial")
    history.extend([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reply},
    ])
    kernel_code = extract_kernel_block(reply)

    tag_base = f"{STEP_TAG}_{source_role}"
    for attempt in range(repair_round_limit()):
        attempt_error = ""
        stage = "extract"
        if not kernel_code:
            attempt_error = "LLM response missing ```kernel``` fenced block"
        else:
            ok, err = compile_check_cpp(
                kernel_code,
                header_code,
                header_name,
                extra_files=extra_files,
            )
            stage = "compile_kernel"
            if not ok:
                attempt_error = err
            elif not testbench_code:
                attempt_error = "benchmark has no testbench for csim"
                stage = "testbench"
            else:
                tag = join_temp_tag(bench, tag_base, f"a{attempt}")
                outcome = _run_synth_csim_cosim(
                    kernel_code,
                    header_code=header_code,
                    header_name=header_name,
                    top_function=top_function,
                    part=part,
                    clock_ns=clock_ns,
                    extra_files=extra_files,
                    testbench_code=testbench_code,
                    run_csim_check=True,
                    run_cosim_check=False,
                    log_prefix=f"[{STEP_TAG}]",
                    temp_tag=tag,
                )
                synth = outcome.get("synth") or {}
                csim_summary = outcome.get("csim")
                stage = "csynth"
                if synth.get("success"):
                    csim_pass = (
                        csim_summary is None
                        or csim_summary.get("passed")
                        or csim_summary.get("status") == "passed"
                    )
                    if not csim_pass:
                        attempt_error = (csim_summary or {}).get("error") or "csim failed"
                        stage = "csim"
                    else:
                        synth_report = synth.get("report") or {}
                        if architecture_ok(synth_report, bench):
                            success = True
                        else:
                            dsp = synth_report.get("dsp")
                            attempt_error = (
                                f"architecture miss: DSP={dsp} "
                                f"(need PE×SIMD nest, DSP>={min_dsp_for_architecture(bench)}; "
                                "do not keep flash crow[j]+= over k)"
                            )
                            stage = "architecture"
                else:
                    attempt_error = synth.get("error") or "csynth failed"

        attempts.append({
            "attempt": attempt,
            "stage": stage,
            "success": success,
            "error": attempt_error[:4000],
            "dsp": (synth_report or {}).get("dsp") if synth_report else None,
            "latency_cycles": (synth_report or {}).get("latency_cycles") if synth_report else None,
        })
        last_error = attempt_error
        if success:
            break
        if attempt >= repair_round_limit() - 1:
            break

        repair_kernel = kernel_code if kernel_code.strip() else source_kernel
        repair_user = _REPAIR_USER.format(
            recipe_block=format_recipe_prompt(bench, step="dse"),
            stage=stage,
            error=attempt_error[:8000],
            header_name=header_name,
            header_code=header_code[:12000],
            kernel_code=repair_kernel[:120000],
        )
        repair_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": repair_user},
        ]
        reply = call_dse_llm(orchestrator, repair_messages, purpose="dse_repair")
        history.extend([
            {"role": "user", "content": repair_user},
            {"role": "assistant", "content": reply},
        ])
        extracted = extract_kernel_block(reply)
        if extracted:
            kernel_code = extracted

    kernel_code = sanitize_kernel_source(kernel_code)
    dsp = (synth_report or {}).get("dsp")
    latency_cycles = (synth_report or {}).get("latency_cycles")
    promoted = False
    promotion: Optional[dict[str, Any]] = None
    if should_promote_dse(
        success=success,
        latency_cycles=latency_cycles,
        baseline_latency=baseline_latency,
        dsp=dsp,
        bench=bench,
    ) and kernel_code:
        promotion = promote_dse_as_selected(
            cell_dir=cell_dir,
            bench=bench,
            code=kernel_code,
            report=synth_report,
            result_payload={
                "success": True,
                "latency_cycles": latency_cycles,
                "dsp": dsp,
            },
        )
        promoted = True

    result_payload: dict[str, Any] = {
        "schema": "post_flash_dse_v1",
        "benchmark": bench,
        "source_role": source_role,
        "step": STEP_TAG,
        "success": success,
        "error": last_error if not success else "",
        "source_kernel": str(kernel_path.name),
        "source_kernel_role": kernel_role,
        "testbench": "benchmark_original",
        "attempts": attempts,
        "synth_report": synth_report,
        "csim": csim_summary,
        "latency_cycles": latency_cycles,
        "dsp": dsp,
        "baseline_latency_cycles": baseline_latency,
        "architecture_ok": architecture_ok(synth_report, bench) if synth_report else False,
        "promoted": promoted,
        "promotion": promotion,
        "skills": skills_meta,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    artifacts: dict[str, str] = {}
    if kernel_code:
        paths["kernel"].write_text(kernel_code, encoding="utf-8")
        artifacts["kernel"] = paths["kernel"].name
        result_payload["kernel_sha256"] = sha256_text(kernel_code)
    if synth_report:
        paths["report"].write_text(json.dumps(synth_report, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts["report"] = paths["report"].name
    if artifacts:
        result_payload["artifacts"] = artifacts

    paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
    paths["history"].write_text(json.dumps({
        "model": getattr(orchestrator, "gpt_model", ""),
        "source_role": source_role,
        "messages": history,
    }, indent=2), encoding="utf-8")
    _write_cell_manifest(cell_dir, bench, kernel_path, result_payload)
    return DseOutcome(bench, source_role, success, str(cell_dir), last_error, result_payload)


def _write_cell_manifest(
    cell_dir: Path,
    bench: str,
    kernel_path: Path,
    result_payload: dict[str, Any],
) -> None:
    manifest_path = cell_dir / f"{bench}_post_flash_{STEP_TAG}.json"
    manifest_path.write_text(json.dumps({
        "schema": "post_flash_dse_manifest_v1",
        "benchmark": bench,
        "source_role": "flash_final",
        "source_kernel": str(kernel_path),
        "success": result_payload.get("success"),
        "promoted": result_payload.get("promoted"),
        "result": artifact_paths(cell_dir, bench)["result"].name,
    }, indent=2) + "\n", encoding="utf-8")


def maybe_chain_dse(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> Optional[DseOutcome]:
    """Run DSE when enabled for flash-final; swallow errors."""
    # DSE 2.0 PE×SIMD sweep replaces v1 when C2HLS_DSE_V2=1.
    # DSE v4 is a separate AutoSA-config harness; it also replaces v1.
    if _env_flag("C2HLS_DSE_V2") is True or _env_flag("C2HLS_DSE_V4") is True:
        return None
    if source_role != "flash_final" or not chain_after_flash() or not dse_enabled():
        return None
    try:
        outcome = run_dse_for_cell(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orchestrator,
            source_role=source_role,
            skip_existing=skip_existing,
        )
        if outcome.success:
            _LOG.info("[dse] %s passed", bench)
        else:
            _LOG.warning("[dse] %s failed: %s", bench, outcome.error[:200])
        return outcome
    except Exception as exc:
        _LOG.exception("[dse] %s error: %s", bench, exc)
        try:
            (cell_dir / f"{bench}_dse_chain_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        except OSError:
            pass
        return None


def configure_post_flash_dse_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")


def prompt_text_for_docs() -> dict[str, str]:
    skills_block, meta = build_dse_skills_prompt_block()
    return {
        "system": _SYSTEM,
        "initial_user": format_dse_initial_user(
            skills_block=skills_block,
            benchmark_context="...",
            header_name="kernel.h",
            header_code="...",
            kernel_code="...",
            synth_summary="...",
        ),
        "repair_user": _REPAIR_USER.format(
            recipe_block=format_recipe_prompt("autosa_mm", step="dse"),
            stage="architecture",
            error="...",
            header_name="kernel.h",
            header_code="...",
            kernel_code="...",
        ),
        "skills_path": meta["skills_path"],
        "skill_ids": meta["skill_ids"],
    }


__all__ = [
    "DseOutcome",
    "POST_FLASH_DSE_SKILL_ENTRIES_JSON",
    "STEP_TAG",
    "architecture_ok",
    "artifact_paths",
    "build_dse_skills_prompt_block",
    "call_dse_llm",
    "chain_after_flash",
    "dse_max_tokens",
    "configure_post_flash_dse_env",
    "discover_dse_cells",
    "dse_enabled",
    "dse_source_role",
    "resolve_baseline_kernel",
    "load_dse_skills",
    "load_existing_result",
    "maybe_chain_dse",
    "min_dsp_for_architecture",
    "post_flash_no_skills",
    "promote_dse_as_selected",
    "prompt_text_for_docs",
    "repair_round_limit",
    "resolve_dse_skills_path",
    "resolve_selected_kernel",
    "run_dse_for_cell",
    "should_promote_dse",
    "validate_dse_skill_entries",
]
