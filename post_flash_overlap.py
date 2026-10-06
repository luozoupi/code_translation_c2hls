"""Post-multi-PE overlap ablation: fuse Crow init/drain + ping-pong DATAFLOW.

Seed is the passing ``*_dse.cpp`` PE array. This step must NOT emit the stream
PE-FIFO pack (``mm_pe`` + packed SIMD FIFOs). Artifacts use the ``overlap`` tag
and never overwrite ``*_selected.cpp``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from c2hls_paths import POST_FLASH_OVERLAP_SKILL_ENTRIES_JSON
from flash_flow_artifacts import sha256_text
from post_flash_dataflow import extract_kernel_block, sanitize_kernel_source
from post_flash_dse import load_existing_result
from post_flash_pe_recipe import format_recipe_prompt, recipe_for
from post_flash_pragma_opt import summarize_synth_report
from post_flash_stream import (
    _compute_pipeline_ii_ok,
    _crow_not_complete_partitioned,
    _kj_not_unflattened,
    call_stream_llm,
    min_dsp_for_architecture,
    repair_round_limit,
    resolve_stream_source_kernel,
    stream_max_tokens,
)
from skill_library import Skill, _coerce_skill_entry, render_skill_set_for_prompt_full

STEP_TAG = "overlap"
_LOG = logging.getLogger(__name__)

_SYSTEM = """You are an expert Xilinx Vitis HLS 2023.2 engineer running an **overlap** ablation.

The seed is a **working multi-PE × SIMD** GEMM (bulk local_A/B/C, PE UNROLL, SIMD
adder tree, II=1 on j). Load, compute, and store are still **sequential**.

Your job is only:
1. **Fuse** Crow init + writeback into the MAC loop (no 64-beat init_pe_j / wb_pe_j).
2. **Overlap** load/store with compute using `#pragma HLS DATAFLOW` and **ping-pong
   buffers** (two A-row tiles, two C-row tiles). Outer `i0 += PE` wraps DATAFLOW.

## Keep
- Exact `extern "C"` top name, ports, array ranks, existing INTERFACE pragmas
  (you may split bundle=gmem → gmem0/gmem1/gmem2).
- `#define PE` / SIMD from the recipe. One `compute_tile` with `Crow[PE][J]` and
  UNROLL PE + UNROLL SIMD. This is still the multi-PE nest, not a new array.

## Forbidden (this is NOT the stream column)
- PE_NUM explicit `mm_pe()` calls with `fifo_A` / `fifo_B` packed `ap_uint` streams
- Systolic `fifo_B[p] → fifo_B[p+1]`
- AutoSA kernel0 / A_t16 netlists
- DATAFLOW on the original body while still sharing one `local_A[I][K]` without ping-pong

## Success
- DATAFLOW around the i-tile loop, ping-pong (or depth>=2) so load(tile n+1) can
  overlap compute(tile n).
- Fused MAC loop, II=1. DSP stays near the recipe (not flash leftover).
- Start with ```kernel immediately.
```kernel
... full kernel source ...
```
"""

_INITIAL_USER = """Run the **overlap** ablation on this multi-PE kernel.

Keep the PE×SIMD nest. Fuse Crow init/drain. Add ping-pong DATAFLOW so load/store
overlap compute. Do **not** emit the stream `mm_pe` FIFO pack.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Multi-PE kernel (seed)
```cpp
{kernel_code}
```

## Multi-PE csynth summary
{synth_summary}

## Overlap skills
{skills_block}
"""

_REPAIR_USER = """Repair the **overlap** kernel.

Keep PE×SIMD. Fuse init/wb. Ping-pong DATAFLOW around i-tiles.
Do not emit mm_pe + packed FIFOs (that is the stream pack).

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


def resolve_overlap_skills_path() -> Path:
    raw = os.getenv("C2HLS_OVERLAP_SKILL_ENTRIES_JSON", "").strip()
    if raw:
        return Path(raw)
    return POST_FLASH_OVERLAP_SKILL_ENTRIES_JSON


def load_overlap_skills(path: Optional[Path] = None) -> list[Skill]:
    skill_path = path or resolve_overlap_skills_path()
    data = json.loads(skill_path.read_text(encoding="utf-8"))
    skills: list[Skill] = []
    for entry in data["skills"]:
        skill = _coerce_skill_entry(entry)
        if skill is not None:
            skills.append(skill)
    return skills


def build_overlap_skills_prompt_block(
    path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    skill_path = path or resolve_overlap_skills_path()
    skills = load_overlap_skills(skill_path)
    meta = {
        "skills_path": str(skill_path),
        "skill_count": len(skills),
        "skill_ids": [sk.id for sk in skills],
    }
    return render_skill_set_for_prompt_full(skills), meta


def artifact_paths(cell_dir: Path, bench: str) -> dict[str, Path]:
    base = f"{bench}_{STEP_TAG}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
    }


def _defines_pe(code: str, pe: int) -> bool:
    return re.search(rf"#\s*define\s+PE(_NUM)?\s+{pe}\b", code) is not None


def is_stream_pe_fifo_pack(code: str) -> bool:
    """True when the kernel is the stream column, not ping-pong overlap."""
    if not code:
        return False
    n_mm_pe = len(re.findall(r"\bmm_pe\s*\(", code))
    n_def = len(re.findall(r"void\s+mm_pe\s*\(", code))
    calls = n_mm_pe - n_def
    has_fifo_b = "fifo_B" in code or "fifo_b" in code.lower()
    has_stream = "hls::stream" in code
    return calls >= 8 and has_fifo_b and has_stream


def architecture_ok(
    code: str,
    report: Optional[dict[str, Any]],
    bench: Optional[str] = None,
) -> bool:
    rec = recipe_for(bench)
    if not code or "DATAFLOW" not in code.upper():
        return False
    if is_stream_pe_fifo_pack(code):
        return False
    if not _defines_pe(code, rec.pe):
        return False
    if not _crow_not_complete_partitioned(code):
        return False
    if not _kj_not_unflattened(code):
        return False
    if not report:
        return False
    try:
        dsp = int(report.get("dsp") or 0)
    except (TypeError, ValueError):
        return False
    if dsp < min_dsp_for_architecture(bench):
        return False
    return _compute_pipeline_ii_ok(report)


def architecture_miss_message(
    code: str,
    report: Optional[dict[str, Any]],
    bench: Optional[str] = None,
) -> str:
    rec = recipe_for(bench)
    dsp = (report or {}).get("dsp")
    return (
        f"overlap miss: keep PE={rec.pe} SIMD={rec.simd} nest, fuse init/wb, "
        "DATAFLOW + ping-pong around i-tiles. Do not emit mm_pe FIFO pack. "
        f"Need DSP>={min_dsp_for_architecture(bench)}, compute II=1. "
        f"Got DSP={dsp}, stream_pack={is_stream_pe_fifo_pack(code)}, "
        f"pe_defined={_defines_pe(code, rec.pe)}, dataflow={'DATAFLOW' in (code or '').upper()}."
    )


@dataclass
class OverlapOutcome:
    bench: str
    source_role: str
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def run_overlap_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    skip_existing: bool = True,
) -> OverlapOutcome:
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag

    kernel_path, kernel_role, prior_report = resolve_stream_source_kernel(cell_dir, bench)
    if kernel_path is None:
        return OverlapOutcome(bench, "missing", False, str(cell_dir), "no dse/selected kernel")
    source_role = kernel_role or "dse"

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
    benchmark_context = format_recipe_prompt(bench, step="overlap") + (
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
            return OverlapOutcome(bench, source_role, True, str(cell_dir), result=existing)

    skills_block, skills_meta = build_overlap_skills_prompt_block()
    synth_summary = summarize_synth_report(prior_report)
    token_floor = stream_max_tokens()
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

    user = _INITIAL_USER.format(
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code[:12000],
        kernel_code=source_kernel[:120000],
        synth_summary=synth_summary,
        skills_block=skills_block,
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    reply = call_stream_llm(orchestrator, messages, purpose="overlap_initial")
    history.extend([
        {"role": "system", "content": _SYSTEM},
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
                        if architecture_ok(kernel_code, synth_report, bench):
                            success = True
                        else:
                            attempt_error = architecture_miss_message(
                                kernel_code, synth_report, bench
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
            recipe_block=format_recipe_prompt(bench, step="overlap"),
            stage=stage,
            error=attempt_error[:8000],
            header_name=header_name,
            header_code=header_code[:12000],
            kernel_code=repair_kernel[:120000],
        )
        repair_messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": repair_user},
        ]
        reply = call_stream_llm(orchestrator, repair_messages, purpose="overlap_repair")
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
    result_payload: dict[str, Any] = {
        "schema": "post_flash_overlap_v1",
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
        "baseline_latency_cycles": (prior_report or {}).get("latency_cycles"),
        "architecture_ok": architecture_ok(kernel_code, synth_report, bench) if synth_report else False,
        "promoted": False,
        "skills": skills_meta,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if kernel_code:
        paths["kernel"].write_text(kernel_code, encoding="utf-8")
        result_payload["kernel_sha256"] = sha256_text(kernel_code)
        result_payload["artifacts"] = {"kernel": paths["kernel"].name}
    if synth_report:
        paths["report"].write_text(json.dumps(synth_report, indent=2, default=str) + "\n", encoding="utf-8")
        result_payload.setdefault("artifacts", {})["report"] = paths["report"].name
    paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
    paths["history"].write_text(json.dumps({
        "model": getattr(orchestrator, "gpt_model", ""),
        "source_role": source_role,
        "messages": history,
    }, indent=2), encoding="utf-8")
    return OverlapOutcome(bench, source_role, success, str(cell_dir), last_error, result_payload)


def configure_post_flash_overlap_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
