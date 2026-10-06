"""Post-DSE stream/I/O step: LLM rewrite into PE modules + hls::stream DATAFLOW.

Runs **after** DSE (or on a DSE-promoted selected kernel). Unlike DSE, this step
**streamifies I/O**: PE tasks, systolic B, burst DRAM loaders. It must keep the
top-level ABI and INTERFACE pragmas. Skills live in
``post_flash_stream_pe_io_skill_entries.json`` and are not mixed into flash or DSE.

Validation: compile + csim + csynth (cosim off). Architecture gate: DATAFLOW +
``hls::stream`` + DSP above the flash leftover.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import autosa_flow_gates
from c2hls_paths import POST_FLASH_STREAM_SKILL_ENTRIES_JSON
from flash_flow_artifacts import sha256_text
from post_flash_dataflow import extract_kernel_block, sanitize_kernel_source
from post_flash_dse import (
    DEFAULT_EMPTY_REPLY_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPAIR_ROUNDS,
    _env_flag,
    _invoke_llm,
    empty_reply_retry_limit,
    load_existing_result,
    post_flash_no_skills,
)
from post_flash_mem_parallel import discover_matrix_cells
from post_flash_pragma_opt import resolve_source_kernel, summarize_synth_report
from post_flash_pe_recipe import (
    RECIPE_REFERENCE_N,
    PeRecipe,
    format_recipe_prompt,
    onchip_tiles_active,
    recipe_for,
    resolve_stream_recipe,
)
from skill_library import (
    Skill,
    _coerce_skill_entry,
    render_skill_set_for_prompt_full,
)

STEP_TAG = "stream"
DEFAULT_STREAM_MIN_DSP = 200

_LOG = logging.getLogger(__name__)

_SYSTEM = """You are an expert Xilinx Vitis HLS 2023.2 engineer running a **stream** step after DSE.

DSE already produced a **working** multi-PE x SIMD GEMM (csim + csynth pass) with
bulk on-chip `local_A` / `local_B` / `local_C`. Latency is still high because
load, compute, and store cannot overlap.

Your job is **I/O construction + streamifying** into compact PE modules, not another PE/SIMD search
and **not** cloning an AutoSA PE/IO netlist.

## Architecture (mandatory)
1. Keep the exact top-level `extern "C"` signature, parameter list, array ranks/shapes, and every existing `#pragma HLS INTERFACE` line.
2. Include `<hls_stream.h>`. Use `hls::stream` + `#pragma HLS DATAFLOW`.
3. Split into INLINE-off tasks: `load_A`, `load_B`, PE_NUM x `mm_pe`, `drain_B`, `store_C`.
4. DATAFLOW region = stream declarations + function calls only. No shared `local_A[I][K]` / `local_B[J][K]` / `local_C[I][J]` across tasks.
5. **PE_NUM, SIMD, and pack width come from the mandatory recipe.** Do not default to PE=16 SIMD=4 ap_uint<128> float union if the recipe says otherwise. Unpack SIMD into scalars. **`pe_kj` must stay II=1**.
6. Name the PE function **`mm_pe`**, not `PE` (`#define PE` would eat the identifier).
7. B is systolic: mm_pe p reads `fifo_B[p]`, forwards to `fifo_B[p+1]`. A is per-PE `fifo_A[p]`.
8. **`Crow[J]` is a BRAM, not complete-partitioned.** `#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram`. Complete-partition of Crow muxes `Crow[j]` into one `mux_case_0` register → **II=4, DSP=5/PE**. Do not DEPENDENCE inter false: Crow[j] has a real dep at distance J; ram_2p already allows II=1.
9. **Flatten k×j into one II=1 pipeline** (`pe_kj`, trip=(K/SIMD)*J from kernel.h — e.g. 1024 at SIMD=4 when J=K=64). Do not assume the matrix is 64³. Do not use LOOP_FLATTEN off on a nested pe_k0/compute_j. Fuse init/drain into pe_kj: first k-tile writes `partial`, last k-tile writes C. Do not flatten pe_i0 (Crow holds one row).
10. Index B as `B[J][K]` → `B[j][k0+s]`. Catapult uses I_P/J_P/K_P.

## Forbidden
- Changing top-level ports, bundles, or function name.
- Slapping DATAFLOW on the DSE load-compute-store body.
- Emitting AutoSA `kernel0`, `A_t16`/`B_t16`, or a 1000+ line PE/IO netlist. Do not emit AutoSA.
- Streaming SIMD as `struct {float,float,float,float}` or `hls::stream<data_t>` four times.
- **`#pragma HLS ARRAY_PARTITION variable=Crow complete`** on the pipelined J index (mux_case_0 II=4).
- PE unroll inside one function as the only "array" (that is DSE, not this step).
- `malloc`, system calls, or non-synthesizable constructs.

## Success check before you submit
- `#pragma HLS DATAFLOW`, packed `hls::stream<ap_uint<pack_bits>>`, PE_NUM `mm_pe` calls from the recipe.
- Crow is ram_2p, not complete-partitioned. One flattened pe_kj, II=1. DSP near the recipe, not a leftover II=4 (DSP~5/PE).
- No `local_A[I][K]` feeding compute.
- Copy the packed PE/stream template from the skills, then change PE_NUM/SIMD/pack to the recipe. Start with ```kernel immediately.

## Output
Start with ```kernel immediately. Close the fence. Do not spend the token budget on analysis.
```kernel
... full kernel source ...
```
"""

_INITIAL_USER = """Run the **stream** step on this DSE (or selected) kernel.

DSE already passed csim + csynth with PE x SIMD compute. Streamify I/O{skills_clause}:
PE modules, `hls::stream`, DATAFLOW, systolic B, burst DRAM loaders.
Keep the top signature and INTERFACE pragmas. Return a single ```kernel``` block.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## DSE kernel (seed)
```cpp
{kernel_code}
```

## DSE csynth summary
{synth_summary}
{skills_section}"""

_REPAIR_USER = """Repair the **stream** kernel after a validation or architecture failure.

Keep the exact top-level signature and `#pragma HLS INTERFACE` pragmas.
Keep PE_NUM/SIMD/pack_bits from the recipe below, DATAFLOW, packed ap_uint streams, mm_pe tasks; do not fall back to shared local_A[I][K] or struct-of-floats FIFOs.
If compute_j II=4 with Memory Dependency on Crow/mux_case_0: remove ARRAY_PARTITION complete on Crow, use ram_2p BRAM, unpack SIMD to scalars. Do not DEPENDENCE inter false.
If PE latency is ~7k with nested pe_k0/compute_j: flatten into one pe_kj pipeline of (K/SIMD)*J beats, fuse init/drain, do not use LOOP_FLATTEN off.
Handshake per chapter (tile or stream beat). Do not PIPO the whole matrix after the whole loop.
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


def stream_enabled() -> bool:
    return _env_flag("C2HLS_POST_FLASH_STREAM") is True


def chain_after_flash() -> bool:
    chained = _env_flag("C2HLS_STREAM_CHAIN_FLASH")
    if chained is not None:
        return chained
    return stream_enabled()


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_STREAM_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def stream_max_tokens() -> int:
    """Completion budget. Flash uses >=16384; 8192 left DeepSeek-v4-flash empty."""
    for key in (
        "C2HLS_STREAM_MAX_TOKENS",
        "C2HLS_DSE_MAX_TOKENS",
        "C2HLS_FLASH_MAX_TOKENS",
        "C2HLS_LLM_MAX_TOKENS",
    ):
        raw = os.getenv(key, "").strip()
        if raw.isdigit():
            return max(int(raw), 8192)
    return DEFAULT_MAX_TOKENS


def call_stream_llm(
    orchestrator: Any,
    messages: list[dict[str, str]],
    *,
    purpose: str = "stream",
) -> str:
    from post_flash_dataflow import is_empty_llm_reply

    tokens = stream_max_tokens()
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
    raw = os.getenv("C2HLS_STREAM_MIN_DSP", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if bench:
        return recipe_for(bench).min_dsp
    return DEFAULT_STREAM_MIN_DSP


def _code_has_streams(code: str) -> bool:
    if not code:
        return False
    low = code.lower()
    has_dataflow = "dataflow" in low
    has_stream = "hls::stream" in code or "hls_stream" in low
    return has_dataflow and has_stream


def _code_has_packed_simd_streams(code: str, pack_bits: int = 128) -> bool:
    if not code:
        return False
    if re.search(rf"ap_uint\s*<\s*{int(pack_bits)}\s*>", code) is not None:
        return True
    # Wider packing of several SIMD groups is acceptable (e.g. 8xu16 in 128).
    if pack_bits < 128 and re.search(r"ap_uint\s*<\s*128\s*>", code) is not None:
        return True
    return False


def _compute_pipeline_ii_ok(report: dict[str, Any]) -> bool:
    feedback = report.get("feedback") if isinstance(report.get("feedback"), dict) else {}
    scopes = feedback.get("scopes") if isinstance(feedback.get("scopes"), list) else []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        blob = f"{scope.get('name') or ''} {scope.get('scope_id') or ''}".lower()
        if not any(tok in blob for tok in ("compute_j", "k0_compute_j", "pe_k0_compute", "pe_kj")):
            continue
        ii = scope.get("pipeline_ii")
        if ii is None:
            continue
        try:
            if int(ii) > 1:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _crow_not_complete_partitioned(code: str) -> bool:
    """Complete-partition of Crow[J] muxes Crow[j] into mux_case_0 → II=4."""
    if not code:
        return True
    return re.search(
        r"ARRAY_PARTITION\s+variable\s*=\s*Crow\s+complete",
        code,
        flags=re.IGNORECASE,
    ) is None


def _kj_not_unflattened(code: str) -> bool:
    """Nested pe_k0 + LOOP_FLATTEN off restarts compute_j 16 times (~7k cycles)."""
    if not code:
        return True
    return re.search(
        r"#\s*pragma\s+HLS\s+LOOP_FLATTEN\s+off",
        code,
        flags=re.IGNORECASE,
    ) is None


def _extract_fn_body(code: str, name: str) -> str:
    match = re.search(rf"(?:static\s+)?void\s+{re.escape(name)}\s*\([^;{{]*\)\s*\{{", code)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    for i, ch in enumerate(code[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return code[start : i + 1]
    return code[start:]


def _i_tile_loop_around_dataflow(code: str) -> bool:
    """True when #pragma HLS DATAFLOW sits inside a for-loop (ping-pong tiles)."""
    if not code:
        return False
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if re.search(r"#\s*pragma\s+HLS\s+DATAFLOW", line, flags=re.IGNORECASE) is None:
            continue
        window = "\n".join(lines[max(0, idx - 12) : idx + 1])
        if re.search(r"\bfor\s*\(", window) is None:
            continue
        if window.count("{") >= window.count("}"):
            return True
    return False


def _tasks_replay_i_tiles(code: str) -> bool:
    """True if mm_pe / load_B / drain_B still loop ROWS_PER_PE or i0 += PE_NUM."""
    tile_re = re.compile(
        r"ROWS_PER_PE|i0\s*\+=\s*PE_NUM|tile\s*<\s*ROWS_PER_PE",
        flags=re.IGNORECASE,
    )
    for name in ("mm_pe", "load_B", "drain_B"):
        body = _extract_fn_body(code, name)
        if body and tile_re.search(body):
            return True
    return False


def architecture_ok(
    code: str,
    report: Optional[dict[str, Any]],
    bench: Optional[str] = None,
) -> bool:
    rec = recipe_for(bench)
    pack_bits = rec.pack_bits if bench else 128
    if not _code_has_streams(code) or not _code_has_packed_simd_streams(code, pack_bits):
        return False
    if not _crow_not_complete_partitioned(code):
        return False
    if not _kj_not_unflattened(code):
        return False
    if rec.tile_loop == "around_dataflow":
        if not _i_tile_loop_around_dataflow(code) or _tasks_replay_i_tiles(code):
            return False
    if not report:
        return False
    try:
        dsp = int(report.get("dsp") or 0)
    except (TypeError, ValueError):
        return False
    if dsp < min_dsp_for_architecture(bench):
        return False
    if not _compute_pipeline_ii_ok(report):
        return False
    if not autosa_flow_gates.io_overlap_ok(
        report, max_latency_over_interval=1.15
    ).ok:
        return False
    return True


def architecture_ok_for_recipe(
    code: str,
    report: Optional[dict[str, Any]],
    rec: PeRecipe,
) -> bool:
    """Same gates as architecture_ok, keyed off ``rec`` (not C2HLS_PE_RECIPE)."""
    if rec.layout == "pack":
        pe_i = rec.pe_i if rec.pe_i else rec.pe
        pe_j = rec.pe_j
        if code.count("pack_pe(") != 1 + pe_i * pe_j:
            return False
        if "autosa_mm_pack(" not in code or "void kernel0(" in code:
            return False
        if "ap_uint<512>" not in code:
            return False
        if _i_tile_loop_around_dataflow(code):
            return False
    elif rec.layout in ("io4", "io5"):
        pe_i = rec.pe_i if rec.pe_i else rec.pe
        pe_j = rec.pe_j
        if code.count("io_pe(") != 1 + pe_i * pe_j:
            return False
        if "autosa_mm_pack(" not in code or "void kernel0(" in code:
            return False
        if "ap_uint<512>" not in code:
            return False
        if "ping" not in code or "pong" not in code:
            return False
        if "K_PART" not in code and "k_part" not in code:
            return False
        if rec.layout == "io5":
            if "fifo_C_in" not in code or "fifo_C_out" not in code:
                return False
        else:
            if "Crow" not in code or "fifo_C_in" in code:
                return False
        if _i_tile_loop_around_dataflow(code):
            return False
    elif rec.layout == "mesh":
        pe_i = rec.pe_i if rec.pe_i else rec.pe
        pe_j = rec.pe_j
        if code.count("mesh_pe(") != 1 + pe_i * pe_j:
            return False
        if _i_tile_loop_around_dataflow(code):
            return False
    if not _code_has_streams(code) or not _code_has_packed_simd_streams(
        code, rec.pack_bits
    ):
        return False
    if not _crow_not_complete_partitioned(code):
        return False
    if not _kj_not_unflattened(code):
        return False
    if rec.tile_loop == "around_dataflow":
        if not _i_tile_loop_around_dataflow(code) or _tasks_replay_i_tiles(code):
            return False
    if not report:
        return False
    try:
        dsp = int(report.get("dsp") or 0)
    except (TypeError, ValueError):
        return False
    if dsp < rec.min_dsp:
        return False
    if not _compute_pipeline_ii_ok(report):
        return False
    if not autosa_flow_gates.io_overlap_ok(
        report, max_latency_over_interval=1.15
    ).ok:
        return False
    return True


def architecture_miss_message(
    code: str,
    report: Optional[dict[str, Any]],
    bench: Optional[str] = None,
    rec: Optional[PeRecipe] = None,
) -> str:
    if rec is None:
        rec = recipe_for(bench)
        dsp_floor = min_dsp_for_architecture(bench)
    else:
        dsp_floor = rec.min_dsp
    dsp = (report or {}).get("dsp")
    packed = _code_has_packed_simd_streams(code, rec.pack_bits if bench else 128)
    ii_ok = _compute_pipeline_ii_ok(report or {})
    crow_ok = _crow_not_complete_partitioned(code)
    flat_ok = _kj_not_unflattened(code)
    wrap_ok = (
        rec.tile_loop != "around_dataflow"
        or (_i_tile_loop_around_dataflow(code) and not _tasks_replay_i_tiles(code))
    )
    extra = ""
    if rec.tile_loop == "around_dataflow":
        extra = (
            f" i-tile loop must WRAP DATAFLOW ({rec.i_tiles} tiles); "
            "mm_pe/load_B/drain_B must not loop ROWS_PER_PE or i0 += PE_NUM. "
        )
    overlap = autosa_flow_gates.io_overlap_ok(
        report, max_latency_over_interval=1.15
    )
    overlap_txt = "" if overlap.ok else f" {overlap.reason}"
    if onchip_tiles_active(rec):
        return (
            f"architecture miss: DATAFLOW ping-pong inside i0 += TI ({rec.ti}) and "
            f"j0 += TJ ({rec.tj}). load_B element trip per tile TJ*TK={rec.load_trip}. "
            f"compute trip per tile (TK/SIMD)*TJ={rec.compute_trip} at II=1. "
            f"store trip PE*TJ={rec.store_trip}. "
            f"PE_NUM={rec.pe} SIMD={rec.simd}, packed ap_uint<{rec.pack_bits}>, "
            f"DSP>={dsp_floor} (expect ~{rec.expected_dsp}). "
            "Crow must be ram_2p on the pipelined column index, not ARRAY_PARTITION complete. "
            "Do not replay B for every PE-row step and every SIMD slice of the full K extent "
            "across all columns. Do not use LOOP_FLATTEN off. "
            f"Got DSP={dsp}, packed={packed}, compute_ii_ok={ii_ok}, "
            f"crow_not_complete={crow_ok}, kj_flattened={flat_ok}, tile_wrap_ok={wrap_ok}."
            f"{overlap_txt} "
            "Do not keep shared local_A[I][K] or AutoSA kernel0."
        )
    return (
        f"architecture miss: need DATAFLOW + packed ap_uint<{rec.pack_bits}> SIMD FIFOs "
        f"(not struct-of-floats), flattened pe_kj of {rec.pe_kj} beats at II=1, "
        f"PE_NUM={rec.pe} SIMD={rec.simd}, DSP>={dsp_floor} "
        f"(expect ~{rec.expected_dsp}). Crow[J] must be ram_2p BRAM, not ARRAY_PARTITION complete "
        f"(that muxes Crow[j] into mux_case_0 → II=4). "
        "Do not use LOOP_FLATTEN off; do not DEPENDENCE inter false. "
        f"{extra}"
        f"Got DSP={dsp}, packed={packed}, compute_ii_ok={ii_ok}, "
        f"crow_not_complete={crow_ok}, kj_flattened={flat_ok}, tile_wrap_ok={wrap_ok}."
        f"{overlap_txt} "
        "Do not keep shared local_A[I][K] or AutoSA kernel0."
    )


def resolve_stream_skills_path() -> Path:
    raw = os.getenv("C2HLS_STREAM_SKILL_ENTRIES_JSON", "").strip()
    if raw:
        return Path(raw)
    return POST_FLASH_STREAM_SKILL_ENTRIES_JSON


def validate_stream_skill_entries(path: Path) -> list[str]:
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


def load_stream_skills(path: Optional[Path] = None) -> list[Skill]:
    skill_path = path or resolve_stream_skills_path()
    errors = validate_stream_skill_entries(skill_path)
    if errors:
        raise ValueError("; ".join(errors))
    data = json.loads(skill_path.read_text(encoding="utf-8"))
    skills: list[Skill] = []
    for entry in data["skills"]:
        skill = _coerce_skill_entry(entry)
        if skill is not None:
            skills.append(skill)
    return skills


def build_stream_skills_prompt_block(
    path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    if post_flash_no_skills():
        return "", {"skills_path": "", "skill_count": 0, "skill_ids": []}
    skill_path = path or resolve_stream_skills_path()
    skills = load_stream_skills(skill_path)
    block = render_skill_set_for_prompt_full(skills)
    meta: dict[str, Any] = {
        "skills_path": str(skill_path),
        "skill_count": len(skills),
        "skill_ids": [sk.id for sk in skills],
    }
    return block, meta


def format_stream_initial_user(
    *,
    skills_block: str,
    benchmark_context: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
    synth_summary: str,
) -> str:
    block = (skills_block or "").strip()
    if block:
        skills_clause = " using the skills below"
        skills_section = (
            "\n## Stream skills (follow these; copy the PE/stream template)\n"
            + block
            + "\n"
        )
    else:
        skills_clause = ""
        skills_section = ""
    return _INITIAL_USER.format(
        skills_clause=skills_clause,
        skills_section=skills_section,
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code,
        kernel_code=kernel_code,
        synth_summary=synth_summary,
    )


def artifact_paths(cell_dir: Path, bench: str) -> dict[str, Path]:
    base = f"{bench}_{STEP_TAG}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
    }


def discover_stream_cells(matrix_root: Path) -> list[dict[str, Any]]:
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


def _load_json_dict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def resolve_stream_source_kernel(
    cell_dir: Path,
    bench: str,
) -> tuple[Optional[Path], str, Optional[dict[str, Any]]]:
    """Prefer a passing DSE kernel; else flash-selected."""
    dse_cpp = cell_dir / f"{bench}_dse.cpp"
    dse_result = _load_json_dict(cell_dir / f"{bench}_dse_result.json")
    if dse_cpp.is_file() and dse_result and dse_result.get("success") is True:
        report = _load_json_dict(cell_dir / f"{bench}_dse_report.json")
        if report is None:
            synth = dse_result.get("synth_report")
            report = synth if isinstance(synth, dict) else {
                "latency_cycles": dse_result.get("latency_cycles"),
                "dsp": dse_result.get("dsp"),
            }
        return dse_cpp, "dse", report
    return resolve_source_kernel(cell_dir, bench, "flash_final")


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def should_promote_stream(
    *,
    success: bool,
    latency_cycles: Any,
    baseline_latency: Any,
    dsp: Any,
    code: str,
    bench: Optional[str] = None,
) -> bool:
    if not success:
        return False
    if not _code_has_streams(code):
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


def promote_stream_as_selected(
    *,
    cell_dir: Path,
    bench: str,
    code: str,
    report: dict[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Overwrite selected pointers with a passing stream kernel; keep DSE + flash seeds."""
    paths = artifact_paths(cell_dir, bench)
    lat = result_payload.get("latency_cycles")
    promotion: dict[str, Any] = {
        "source_role": "dse",
        "latency_cycles": lat,
        "kernel": paths["kernel"].name,
        "selected_stage": "stream",
    }
    selected_cpp = cell_dir / f"{bench}_selected.cpp"
    selected_report = cell_dir / f"{bench}_selected_report.json"
    dse_cpp = cell_dir / f"{bench}_dse.cpp"
    if dse_cpp.is_file():
        promotion["dse_kernel"] = dse_cpp.name
    seed_cpp = cell_dir / f"{bench}_flash_seed.cpp"
    if seed_cpp.is_file():
        promotion["flash_seed_kernel"] = seed_cpp.name
    elif selected_cpp.is_file() and selected_cpp.read_text(encoding="utf-8").strip():
        shutil.copy2(selected_cpp, seed_cpp)
        promotion["flash_seed_kernel"] = seed_cpp.name
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
        lat_map["stream"] = lat
    manifest["latency_cycles"] = lat_map
    manifest["selected_from"] = "stream"
    manifest["selected_sha256"] = sha256_text(code)
    if "files" not in manifest or not isinstance(manifest["files"], dict):
        manifest["files"] = {}
    files = dict(manifest["files"])
    files["selected"] = selected_cpp.name
    files["selected_report"] = selected_report.name
    files["stream"] = paths["kernel"].name
    files["stream_report"] = paths["report"].name
    manifest["files"] = files
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    promotion["flow_manifest"] = manifest_path.name
    return promotion


@dataclass
class StreamOutcome:
    bench: str
    source_role: str
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def problem_size_from_bench(
    header_code: str,
    meta: Optional[dict[str, Any]] = None,
) -> tuple[int, int, int]:
    """I, J, K from kernel.h, then bench metadata ``ijk`` if the header has none."""
    from post_flash_dse_v2 import parse_ijk_from_header

    try:
        return parse_ijk_from_header(header_code or "")
    except ValueError:
        raw = (meta or {}).get("ijk")
        if isinstance(raw, int) and raw > 0:
            return raw, raw, raw
        if isinstance(raw, dict):
            i, j, k = raw.get("I"), raw.get("J"), raw.get("K")
            if all(isinstance(v, int) and v > 0 for v in (i, j, k)):
                return int(i), int(j), int(k)
        raise


def stream_recipe_for_header(
    bench: str,
    header_code: str,
    meta: Optional[dict[str, Any]] = None,
    seed_code: Optional[str] = None,
) -> tuple[PeRecipe, tuple[int, int, int]]:
    """Size from kernel.h. PE×SIMD from the seed unless an env override is set."""
    i, j, k = problem_size_from_bench(header_code, meta)
    return resolve_stream_recipe(bench, i, j, k, seed_code=seed_code), (i, j, k)


def problem_size_override_block(i: int, j: int, k: int, rec: PeRecipe) -> str:
    """Tell the stream prompt the live size when it is not the authored 64³."""
    if onchip_tiles_active(rec):
        return (
            "## Problem size override (authoritative)\n"
            f"This benchmark is I={i}, J={j}, K={k} from kernel.h / bench metadata.\n"
            f"On-chip tiles from the seed: TI={rec.ti} TJ={rec.tj} TK={rec.tk}.\n"
            "Skill examples that stream B across the full matrix are the wrong nest for this seed.\n"
            f"- PE_NUM={rec.pe}\n"
            f"- SIMD={rec.simd}\n"
            f"- TI={rec.ti}\n"
            f"- TJ={rec.tj}\n"
            f"- TK={rec.tk}\n"
            f"- outer tile pairs (I/TI)*(J/TJ) = {rec.outer_tiles}\n"
            f"- load_B element trip per tile TJ*TK = {rec.load_trip}\n"
            f"- compute trip per tile (TK/SIMD)*TJ = {rec.compute_trip}\n"
            f"- store trip PE*TJ = {rec.store_trip}\n"
            "- i0 steps by TI, j0 steps by TJ, k only inside TK\n"
        )
    if (i, j, k) == (RECIPE_REFERENCE_N, RECIPE_REFERENCE_N, RECIPE_REFERENCE_N):
        return ""
    return (
        "## Problem size override (authoritative)\n"
        f"This benchmark is I={i}, J={j}, K={k} from kernel.h / bench metadata.\n"
        "Skill examples written for a 64^3 matrix are the 64^3 illustration only. "
        "Do not copy those literals.\n"
        f"- PE_NUM={rec.pe}\n"
        f"- SIMD={rec.simd}\n"
        f"- pe_kj trip = (K/SIMD)*J = {rec.pe_kj}\n"
        f"- i_tiles = I/PE_NUM = {rec.i_tiles}\n"
        f"- Crow reuse distance is J={j}; fifo_C STREAM depth >= {j}\n"
        "- array extents and loop trips use kernel.h macros I, J, K\n"
    )


def _replace_pe_kj_literal(text: str, old: int, new: int) -> str:
    if old == new:
        return text
    text = text.replace(f"pe_kj={old}", f"pe_kj={new}")
    text = text.replace(f"pe_kj of {old} beats", f"pe_kj of {new} beats")
    text = text.replace(f"pipeline of {old} beats", f"pipeline of {new} beats")
    text = text.replace(f"of {old} beats", f"of {new} beats")
    return text


def specialize_skill_block_for_problem(
    block: str,
    rec: PeRecipe,
    i: int,
    j: int,
    k: int,
) -> str:
    """Rewrite 64³ trip/extent phrases in a rendered skill block.

    The stock skill file stays the 64³ template (unit tests read it as-is).
    A run pointed at another size, or at a seed PE×SIMD other than 16×4, must
    not keep those literals.
    """
    pair_is_locked_mm = rec.pe == 16 and rec.simd == 4
    size_is_ref = (i, j, k) == (
        RECIPE_REFERENCE_N,
        RECIPE_REFERENCE_N,
        RECIPE_REFERENCE_N,
    )
    if not block or (size_is_ref and pair_is_locked_mm and not onchip_tiles_active(rec)):
        return block
    text = block
    if not size_is_ref:
        text = re.sub(r"(?<![A-Za-z])J=64(?!\d)", f"J={j}", text)
        text = re.sub(r"J \(64\)(?!\d)", f"J ({j})", text)
        text = re.sub(r"depth=64(?!\d)", f"depth={j}", text)
        text = text.replace("when J=K=64", f"when J={j} K={k}")
    if rec.simd > 0:
        ref_pe_kj = (RECIPE_REFERENCE_N // rec.simd) * RECIPE_REFERENCE_N
        text = _replace_pe_kj_literal(text, ref_pe_kj, rec.pe_kj)
        text = text.replace(
            f"{ref_pe_kj} at SIMD={rec.simd}",
            f"{rec.pe_kj} at SIMD={rec.simd}",
        )
    # The skill file's copy-paste template is the 16×4 / pe_kj=1024 illustration.
    if not pair_is_locked_mm:
        text = text.replace("#define PE_NUM 16", f"#define PE_NUM {rec.pe}")
        text = text.replace("#define SIMD 4", f"#define SIMD {rec.simd}")
        text = text.replace(
            "Default PE=16 SIMD=4",
            f"Default PE={rec.pe} SIMD={rec.simd}",
        )
        text = text.replace(
            "Do not default to PE=16 SIMD=4 ap_uint<128> float union if the recipe says otherwise.",
            (
                f"Use PE={rec.pe} SIMD={rec.simd} ap_uint<{rec.pack_bits}> from the recipe. "
                "Do not keep a different pair from a skill example."
            ),
        )
        if rec.pe_kj != 1024:
            text = text.replace("1024 at SIMD=4", f"{rec.pe_kj} at SIMD={rec.simd}")
            text = _replace_pe_kj_literal(text, 1024, rec.pe_kj)
    if onchip_tiles_active(rec):
        text = _rewrite_prompt_for_onchip_tiles(text, rec)
    return text


_FULL_MATRIX_B_REPLAY = re.compile(
    r"for\s*\(\s*int\s+i0\s*=\s*0\s*;\s*i0\s*<\s*I\s*;\s*i0\s*\+=\s*PE_NUM\s*\)"
    r"[\s\S]{0,800}?"
    r"for\s*\(\s*int\s+k0\s*=\s*0\s*;\s*k0\s*<\s*K\s*;\s*k0\s*\+=\s*SIMD\s*\)"
    r"[\s\S]{0,800}?"
    r"for\s*\(\s*int\s+j\s*=\s*0\s*;\s*j\s*<\s*J\b",
)


def prompt_instructs_full_matrix_b_replay(text: str) -> bool:
    """True when the prompt tells load_B to walk full K and all J for every i0."""
    if not text:
        return False
    if "(K/SIMD)*J" in text or "(K / SIMD) * J" in text:
        return True
    return _FULL_MATRIX_B_REPLAY.search(text) is not None


def _replace_static_function(text: str, name: str, new_body: str) -> str:
    pattern = re.compile(rf"static void {re.escape(name)}\b")
    parts: list[str] = []
    pos = 0
    for match in pattern.finditer(text):
        parts.append(text[pos:match.start()])
        brace = text.find("{", match.end())
        if brace < 0:
            parts.append(text[match.start():])
            return "".join(parts)
        depth = 0
        end = None
        for index in range(brace, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            parts.append(text[match.start():])
            return "".join(parts)
        parts.append(new_body.rstrip() + "\n")
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def _onchip_tile_task_sources(rec: PeRecipe) -> dict[str, str]:
    pack = f"ap_uint<{rec.pack_bits}>"
    load_a = f"""static void load_A(data_t A[I][K], hls::stream<{pack} > fifo_A[PE_NUM], int i0) {{
#pragma HLS INLINE off
  // TI x TK tile. k only while k0 < TK. Do not walk the full K extent.
  load_i: for (int i = 0; i < TI; ++i) {{
    load_k0: for (int k0 = 0; k0 < TK; k0 += SIMD) {{
#pragma HLS PIPELINE II=1
      fifo_A[0].write(/* pack SIMD lanes of A[i0 + i][k0 + s], s = 0..SIMD-1 */);
    }}
  }}
}}"""
    load_b = f"""static void load_B(data_t B[J][K], hls::stream<{pack} > &fifo_B, int j0) {{
#pragma HLS INLINE off
  // One TJ x TK tile. Element trip {rec.load_trip}. Not under i0, and k stays inside TK.
  load_j: for (int j = 0; j < TJ; ++j) {{
    load_k0: for (int k0 = 0; k0 < TK; k0 += SIMD) {{
#pragma HLS PIPELINE II=1
      fifo_B.write(/* pack SIMD lanes of B[j0 + j][k0 + s], s = 0..SIMD-1 */);
    }}
  }}
}}"""
    mm_pe = f"""static void mm_pe(
    hls::stream<{pack} > &fifo_A,
    hls::stream<{pack} > &fifo_B_in,
    hls::stream<{pack} > &fifo_B_out,
    hls::stream<data_t> &fifo_C) {{
#pragma HLS INLINE off
  data_t Crow[TJ];
#pragma HLS BIND_STORAGE variable=Crow type=ram_2p impl=bram
  // One PE row. B is the shared TJ x TK tile. Compute trip {rec.compute_trip}.
  compute_k0: for (int k0 = 0; k0 < TK; k0 += SIMD) {{
    compute_j: for (int j = 0; j < TJ; ++j) {{
#pragma HLS PIPELINE II=1
      // unpack SIMD lanes, MAC into Crow[j], forward this B beat to the next PE
    }}
  }}
  write_j: for (int j = 0; j < TJ; ++j) {{
#pragma HLS PIPELINE II=1
    fifo_C.write(Crow[j]);
  }}
}}"""
    drain_b = f"""static void drain_B(hls::stream<{pack} > &fifo_B) {{
#pragma HLS INLINE off
  drain_j: for (int j = 0; j < TJ; ++j) {{
    drain_k0: for (int k0 = 0; k0 < TK; k0 += SIMD) {{
#pragma HLS PIPELINE II=1
      (void)fifo_B.read();
    }}
  }}
}}"""
    store_c = f"""static void store_C(data_t C[I][J], hls::stream<data_t> fifo_C[PE_NUM], int i0, int j0) {{
#pragma HLS INLINE off
  // Seed stores i < PE, not i < TI. Trip {rec.store_trip}.
  store_p: for (int p = 0; p < PE_NUM; ++p) {{
    store_j: for (int j = 0; j < TJ; ++j) {{
#pragma HLS PIPELINE II=1
      C[i0 + p][j0 + j] = fifo_C[p].read();
    }}
  }}
}}"""
    return {
        "load_A": load_a,
        "load_B": load_b,
        "mm_pe": mm_pe,
        "drain_B": drain_b,
        "store_C": store_c,
    }


def _rewrite_prompt_for_onchip_tiles(text: str, rec: PeRecipe) -> str:
    """Replace the systolic full-J copy-paste with the seed's on-chip tile nest."""
    text = text.replace(
        "3. Split into INLINE-off tasks: `load_A`, `load_B`, PE_NUM x `mm_pe`, `drain_B`, `store_C`.",
        "3. Split into INLINE-off tasks: `load_A`, `load_B`, PE_NUM x `mm_pe` on the tile, "
        "`drain_B`, `store_C`. DATAFLOW/ping-pong overlaps load, compute, and store. "
        "Do not give those mm_pe calls a trip over every SIMD slice of the full K extent and every column.",
    )
    text = text.replace(
        "7. B is systolic: mm_pe p reads `fifo_B[p]`, forwards to `fifo_B[p+1]`. A is per-PE `fifo_A[p]`.",
        "7. B is one TJ×TK tile reused by the PEs (forward that tile on `fifo_B[p]` or broadcast it). "
        "A is per-PE for the current i0 tile. Do not stream a new B beat for every column on every "
        "PE-row step and every SIMD slice of the full K extent.",
    )
    text = re.sub(
        r"\*\*Flatten k×j into one II=1 pipeline\*\* \(.*?Do not flatten pe_i0 \(Crow holds one row\)\.",
        (
            f"**Keep the seed tile nest.** i0 steps by TI={rec.ti}, j0 steps by TJ={rec.tj}. "
            f"k only while k0 < TK={rec.tk}. Do not add a loop over full K, and do not widen the partial GEMM. "
            f"Compute is the PE-row × SIMD-lane nest inside the tile, trip {rec.compute_trip}, II=1. "
            f"load_B moves one TJ×TK tile, element trip {rec.load_trip}, reused by the PEs. "
            "Do not replay B on every PE-row step and every SIMD slice of the full K extent across all columns. "
            "Ping-pong DATAFLOW so load, compute, and store overlap."
        ),
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        "10. Index B as `B[J][K]` → `B[j][k0+s]`. Catapult uses I_P/J_P/K_P.",
        "10. Index the B tile as `B[j0 + j][k]` with j in 0..TJ-1 and k in 0..TK-1. "
        "Catapult uses I_P/J_P/K_P.",
    )
    text = text.replace(
        "Crow is ram_2p, not complete-partitioned. One flattened pe_kj, II=1.",
        "Crow is ram_2p, not complete-partitioned. Tile compute II=1.",
    )
    text = text.replace(
        "If PE latency is ~7k with nested pe_k0/compute_j: flatten into one pe_kj pipeline of (K/SIMD)*J beats, fuse init/drain, do not use LOOP_FLATTEN off.",
        "If load_B or compute walks the full K extent or every column: cut it back to one TJ×TK B tile "
        "and the seed nest k0 < TK step SIMD, j < TJ. Do not use LOOP_FLATTEN off.",
    )
    text = text.replace(
        "wrap #pragma HLS DATAFLOW in the top i-tile loop (i0 += PE_NUM). Pass i0 into load_A/store_C. "
        "mm_pe, load_B, and drain_B have NO tile/i0/ROWS_PER_PE loop — one pe_kj of (K/SIMD)*J beats per firing.",
        "wrap #pragma HLS DATAFLOW in the top tile loops (i0 += TI, j0 += TJ). "
        f"load_B emits one TJ×TK tile (element trip {rec.load_trip}). "
        f"Each mm_pe covers compute trip {rec.compute_trip} inside TK and TJ.",
    )
    text = text.replace(
        "top: for (int i0 = 0; i0 < I; i0 += PE_NUM) { #pragma HLS DATAFLOW ... }",
        "top: for (int i0 = 0; i0 < I; i0 += TI) { for (int j0 = 0; j0 < J; j0 += TJ) { #pragma HLS DATAFLOW ... } }",
    )
    text = text.replace(
        "mm_pe / load_B / drain_B contain only pe_kj (or KTILES*J) — delete for tile < ROWS_PER_PE and for i0 += PE_NUM inside those tasks",
        "mm_pe / load_B / drain_B contain one TJ x TK tile. "
        "Delete a loop over the full K extent or over every column. "
        "Outer loops are i0 += TI and j0 += TJ.",
    )
    text = text.replace(
        "pe_i0 wraps DATAFLOW in the TOP; mm_pe/load_B/drain_B have only pe_kj",
        "i0 += TI and j0 += TJ wrap DATAFLOW in the TOP; mm_pe/load_B/drain_B see one TJ x TK tile",
    )
    for name, body in _onchip_tile_task_sources(rec).items():
        text = _replace_static_function(text, name, body)
    text = re.sub(
        r"tile_i0:\s*for\s*\(int i0 = 0; i0 < I; i0 \+= PE_NUM\)\s*\{.*?"
        r"store_C\(C, fifo_C, i0\);\s*\n\s*\}",
        (
            "tile_i0: for (int i0 = 0; i0 < I; i0 += TI) {\n"
            "    tile_j0: for (int j0 = 0; j0 < J; j0 += TJ) {\n"
            "#pragma HLS DATAFLOW\n"
            "      load_A(A, fifo_A, i0);\n"
            "      load_B(B, fifo_B[0], j0);\n"
            f"      mm_pe(/* PE_NUM calls; compute trip {rec.compute_trip} inside the tile */);\n"
            "      drain_B(fifo_B[PE_NUM]);\n"
            "      store_C(C, fifo_C, i0, j0);\n"
            "    }\n"
            "  }"
        ),
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"flatten k×j into ONE pe_kj pipeline of \d+ beats\.",
        f"keep compute inside the tile (trip {rec.compute_trip}), not a full-matrix sweep.",
        text,
    )
    text = re.sub(
        r"one pe_kj loop of [^.;\n]+",
        f"one tile compute of {rec.compute_trip} beats inside TK and TJ",
        text,
    )
    text = text.replace("(K/SIMD)*J", f"(TK/SIMD)*TJ={rec.compute_trip}")
    text = text.replace("(K / SIMD) * J", "(TK/SIMD)*TJ")
    text = re.sub(
        r"SIMD=4 pe_kj=\d+",
        f"SIMD={rec.simd} tile_compute={rec.compute_trip}",
        text,
    )
    return text


def audit_resolved_stream_recipe(
    bench: str,
    *,
    cell_dir: Path,
    bench_dir: Path,
) -> dict[str, Any]:
    """Resolve the recipe the stream job will actually prompt with."""
    header = ""
    header_path = bench_dir / "kernel.h"
    if header_path.is_file():
        header = header_path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    meta_path = bench_dir / "metadata.json"
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except json.JSONDecodeError:
            meta = {}
    kpath, role, _report = resolve_stream_source_kernel(cell_dir, bench)
    seed = kpath.read_text(encoding="utf-8") if kpath is not None and kpath.is_file() else ""
    rec, size = stream_recipe_for_header(bench, header, meta, seed_code=seed)
    prompt = format_recipe_prompt(
        bench, step="stream", i=size[0], j=size[1], k=size[2], recipe=rec
    )
    override = problem_size_override_block(size[0], size[1], size[2], rec)
    skills, _skills_meta = build_stream_skills_prompt_block()
    skills = specialize_skill_block_for_problem(skills, rec, *size)
    system = specialize_skill_block_for_problem(_SYSTEM, rec, *size)
    repair = specialize_skill_block_for_problem(
        _REPAIR_USER.format(
            recipe_block=prompt + override,
            stage="architecture",
            error="dry-run",
            header_name="kernel.h",
            header_code="",
            kernel_code="",
        ),
        rec,
        *size,
    )
    full = system + prompt + override + skills + repair
    return {
        "bench": bench,
        "kernel_role": role,
        "pe": rec.pe,
        "simd": rec.simd,
        "ti": rec.ti,
        "tj": rec.tj,
        "tk": rec.tk,
        "load_trip": rec.load_trip,
        "compute_trip": rec.compute_trip,
        "store_trip": rec.store_trip,
        "outer_tiles": rec.outer_tiles,
        "pe_kj": rec.pe_kj,
        "i_tiles": rec.i_tiles,
        "I": size[0],
        "J": size[1],
        "K": size[2],
        "pack_bits": rec.pack_bits,
        "full_kj_b_replay": prompt_instructs_full_matrix_b_replay(full),
        "has_define_simd4": "#define SIMD 4" in full,
        "has_pe_kj_1024": "pe_kj=1024" in full,
        "has_simd_assign_4": "\n- SIMD=4\n" in prompt,
        "has_j64": re.search(r"(?<![A-Za-z])J=64(?!\d)", full) is not None,
        "has_depth64": "depth=64" in full,
    }


def apply_stream_bench_timeouts(meta: Optional[dict[str, Any]]) -> None:
    """Honor staged csim/synth timeouts on the bench (n1024 is 86400s)."""
    if not isinstance(meta, dict):
        return
    if meta.get("csim_timeout_s") is not None:
        os.environ["C2HLS_CSIM_TIMEOUT"] = str(int(meta["csim_timeout_s"]))
    if meta.get("synth_timeout_s") is not None:
        os.environ["C2HLS_SYNTH_TIMEOUT"] = str(int(meta["synth_timeout_s"]))
    try:
        import hls_eval
    except ImportError:
        return
    raw = os.environ.get("C2HLS_CSIM_TIMEOUT", "").strip()
    if raw.isdigit():
        hls_eval.CSIM_TIMEOUT = int(raw)


def resolve_stream_bench_dir(bench: str, *, repo: Optional[Path] = None) -> Path:
    """Bench directory for stream csim/csynth.

    ``C2HLS_AUTOSA_READY_ROOT`` wins so a run can point at n1024 without
    rewriting the live 64³ ``related_work/benchmarks/autosa_ready`` tree.
    """
    from c2hls_paths import BENCHMARKS_DIR, REPO_ROOT

    root = repo or REPO_ROOT
    ready = os.getenv("C2HLS_AUTOSA_READY_ROOT", "").strip()
    candidates: list[Path] = []
    if ready:
        candidates.append(Path(ready) / bench)
    candidates += [
        BENCHMARKS_DIR / bench,
        root / "related_work/benchmarks/autosa_ready" / bench,
        root / "benchmarks_autosa_dse" / bench,
        root / "benchmarks_autosa" / bench,
        root / "related_work/benchmarks/HLSFactory_benchmarks/chathls_ready" / bench,
        root / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready" / bench,
        root / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready" / bench,
    ]
    for path in candidates:
        if (path / "metadata.json").is_file():
            return path
    raise ValueError(f"unknown benchmark: {bench}")


def run_stream_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> StreamOutcome:
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag

    if source_role != "flash_final":
        return StreamOutcome(bench, source_role, False, str(cell_dir), "stream is flash-final only")

    kernel_path, kernel_role, prior_report = resolve_stream_source_kernel(cell_dir, bench)
    if kernel_path is None:
        return StreamOutcome(bench, source_role, False, str(cell_dir), "no dse/selected kernel cpp")

    inputs = _load_benchmark_inputs(str(bench_dir))
    source_kernel = kernel_path.read_text(encoding="utf-8")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    apply_stream_bench_timeouts(meta)
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or "workload"
    )
    sized_recipe, (prob_i, prob_j, prob_k) = stream_recipe_for_header(
        bench, header_code, meta, seed_code=source_kernel
    )
    benchmark_context = (
        format_recipe_prompt(
            bench,
            step="stream",
            i=prob_i,
            j=prob_j,
            k=prob_k,
            recipe=sized_recipe,
        )
        + problem_size_override_block(prob_i, prob_j, prob_k, sized_recipe)
        + (inputs.get("benchmark_context", "") or "")
    )
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    paths = artifact_paths(cell_dir, bench)
    if skip_existing:
        existing = load_existing_result(paths["result"])
        if existing is not None:
            return StreamOutcome(bench, source_role, True, str(cell_dir), result=existing)

    skills_block, skills_meta = build_stream_skills_prompt_block()
    skills_block = specialize_skill_block_for_problem(
        skills_block, sized_recipe, prob_i, prob_j, prob_k
    )
    synth_summary = summarize_synth_report(prior_report)
    baseline_latency = (prior_report or {}).get("latency_cycles")
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

    system = specialize_skill_block_for_problem(
        _SYSTEM, sized_recipe, prob_i, prob_j, prob_k
    )
    user = format_stream_initial_user(
        skills_block=skills_block,
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code[:12000],
        kernel_code=source_kernel[:120000],
        synth_summary=synth_summary,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reply = call_stream_llm(orchestrator, messages, purpose="stream_initial")
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
                        if architecture_ok_for_recipe(
                            kernel_code, synth_report, sized_recipe
                        ):
                            success = True
                        else:
                            dsp = synth_report.get("dsp")
                            attempt_error = architecture_miss_message(
                                kernel_code, synth_report, bench, sized_recipe
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
        repair_template = specialize_skill_block_for_problem(
            _REPAIR_USER, sized_recipe, prob_i, prob_j, prob_k
        )
        repair_user = repair_template.format(
            recipe_block=(
                format_recipe_prompt(
                    bench,
                    step="stream",
                    i=prob_i,
                    j=prob_j,
                    k=prob_k,
                    recipe=sized_recipe,
                )
                + problem_size_override_block(prob_i, prob_j, prob_k, sized_recipe)
            ),
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
        reply = call_stream_llm(orchestrator, repair_messages, purpose="stream_repair")
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
    if should_promote_stream(
        success=success,
        latency_cycles=latency_cycles,
        baseline_latency=baseline_latency,
        dsp=dsp,
        code=kernel_code,
        bench=bench,
    ) and kernel_code:
        promotion = promote_stream_as_selected(
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
        "schema": "post_flash_stream_v1",
        "benchmark": bench,
        "problem_size": {"I": prob_i, "J": prob_j, "K": prob_k},
        "pe": sized_recipe.pe,
        "simd": sized_recipe.simd,
        "ti": sized_recipe.ti,
        "tj": sized_recipe.tj,
        "tk": sized_recipe.tk,
        "load_trip": sized_recipe.load_trip,
        "compute_trip": sized_recipe.compute_trip,
        "store_trip": sized_recipe.store_trip,
        "outer_tiles": sized_recipe.outer_tiles,
        "pack_bits": sized_recipe.pack_bits,
        "pe_kj": sized_recipe.pe_kj,
        "i_tiles": sized_recipe.i_tiles,
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
        "architecture_ok": (
            architecture_ok_for_recipe(kernel_code, synth_report, sized_recipe)
            if synth_report
            else False
        ),
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
    return StreamOutcome(bench, source_role, success, str(cell_dir), last_error, result_payload)


def _write_cell_manifest(
    cell_dir: Path,
    bench: str,
    kernel_path: Path,
    result_payload: dict[str, Any],
) -> None:
    manifest_path = cell_dir / f"{bench}_post_flash_{STEP_TAG}.json"
    manifest_path.write_text(json.dumps({
        "schema": "post_flash_stream_manifest_v1",
        "benchmark": bench,
        "source_role": "flash_final",
        "source_kernel": str(kernel_path),
        "success": result_payload.get("success"),
        "promoted": result_payload.get("promoted"),
        "result": artifact_paths(cell_dir, bench)["result"].name,
    }, indent=2) + "\n", encoding="utf-8")


def maybe_chain_stream(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> Optional[StreamOutcome]:
    """Run stream when enabled for flash-final; swallow errors."""
    if source_role != "flash_final" or not chain_after_flash() or not stream_enabled():
        return None
    try:
        outcome = run_stream_for_cell(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orchestrator,
            source_role=source_role,
            skip_existing=skip_existing,
        )
        if outcome.success:
            _LOG.info("[stream] %s passed", bench)
        else:
            _LOG.warning("[stream] %s failed: %s", bench, outcome.error[:200])
        return outcome
    except Exception as exc:
        _LOG.exception("[stream] %s error: %s", bench, exc)
        try:
            (cell_dir / f"{bench}_stream_chain_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        except OSError:
            pass
        return None


def configure_post_flash_stream_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")


def prompt_text_for_docs() -> dict[str, str]:
    skills_block, meta = build_stream_skills_prompt_block()
    return {
        "system": _SYSTEM,
        "initial_user": format_stream_initial_user(
            skills_block=skills_block,
            benchmark_context="...",
            header_name="kernel.h",
            header_code="...",
            kernel_code="...",
            synth_summary="...",
        ),
        "repair_user": _REPAIR_USER.format(
            recipe_block=format_recipe_prompt("autosa_mm", step="stream"),
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
    "POST_FLASH_STREAM_SKILL_ENTRIES_JSON",
    "STEP_TAG",
    "StreamOutcome",
    "architecture_ok",
    "artifact_paths",
    "build_stream_skills_prompt_block",
    "call_stream_llm",
    "chain_after_flash",
    "configure_post_flash_stream_env",
    "discover_stream_cells",
    "load_stream_skills",
    "maybe_chain_stream",
    "min_dsp_for_architecture",
    "promote_stream_as_selected",
    "prompt_text_for_docs",
    "repair_round_limit",
    "resolve_stream_skills_path",
    "resolve_stream_source_kernel",
    "run_stream_for_cell",
    "should_promote_stream",
    "stream_enabled",
    "stream_max_tokens",
    "validate_stream_skill_entries",
]
