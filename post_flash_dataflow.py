"""Optional post-flash DATAFLOW step (task functions + #pragma HLS DATAFLOW).

Runs **after** flash has selected a final kernel (``*_final.cpp`` or record-flow
``*_selected.cpp``). The LLM refactors load / compute / store into separate
functions under ``#pragma HLS DATAFLOW``. Validation uses the **original**
benchmark testbench internally for csim only — the LLM never sees it. Hybrid contract
check (static + LLM auditor) runs before compile/csim/csynth (up to four contract-fix
rounds, separate budget from csynth repair). csim + csynth; cosim off by default.
Up to four repair rounds per cell on validation failure.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flash_flow_artifacts import sha256_text
from dataflow_contract_check import (
    contract_audit_system_prompt,
    contract_check_enabled,
    contract_failure_message,
    contract_round_limit,
    format_contract_audit_user,
    format_contract_breaches_json,
    format_contract_fix_user,
    hybrid_contract_check,
    parse_llm_contract_report,
    static_contract_check,
)
from post_flash_mem_parallel import (
    discover_matrix_cells,
    extract_labeled_cpp_blocks,
    resolve_selected_kernel,
)

DEFAULT_REPAIR_ROUNDS = 4
STEP_TAG = "dataflow"
DEFAULT_PROMPT_POLICY = "system_skills"
PROMPT_POLICIES = ("system_skills", "user_skills")

_SYSTEM = """You are an expert Xilinx Vitis HLS engineer specializing in **task-level pipelining** with `#pragma HLS DATAFLOW`.

Refactor the given flash-optimized HLS kernel into **separate synthesizable task functions** and connect them under a top-level `#pragma HLS DATAFLOW` region so independent tasks can overlap in time.

## MANDATORY — always use `#pragma HLS DATAFLOW` (this step)
This is the **DATAFLOW refactor step**. The top kernel body **must** contain exactly one `#pragma HLS DATAFLOW` and call static task functions inside it.

**Never** omit `#pragma HLS DATAFLOW`.
**Never** run load → compute → store **sequentially** without DATAFLOW because overlap is unclear.
**Never** add a comment like “sequential execution to avoid DATAFLOW conflicts” and skip the pragma.

Skills that warn about “decorative DATAFLOW” mean: **fix overlap** (distinct `gmemN`, `hls::stream`, ping-pong buffers, single-writer/single-reader locals) — **not** remove DATAFLOW.

If overlap is not yet provable, **still keep DATAFLOW** and fix structure until csynth passes:
1. Rebundle every `m_axi` port to `gmem0`, `gmem1`, …
2. Split load / compute / store into separate tasks with clean buffer handoff
3. Add `hls::stream` or ping-pong (`buf0`/`buf1`) where tasks exchange tiles

## Top-level kernel (must not change)
- Keep the **exact** `extern "C"` top function named in benchmark context (`metadata.json` → `translated_hls_top`: `kernel_*` for HLSFactory, `workload` for Rodinia/MachSuite).
- Keep parameter list, array shapes, and all `#pragma HLS INTERFACE` pragmas on that single top (you **may** change only `bundle=gmem` → `bundle=gmemN` per port for DATAFLOW overlap).
- Do not add a second top-level export (no extra `workload()` forwarding to `kernel_*`, no duplicate `extern "C"` blocks).
- Only refactor the **body** of the top function: call task functions inside `#pragma HLS DATAFLOW`.

## Task function signatures
- Declare tasks as **`static` functions** at file scope (before the top kernel).
- Mark each task with `#pragma HLS INLINE off` so DATAFLOW instantiates separate processes.
- Pass data through **function arguments** only — array parameters with **fixed bounds** using the same macros/constants as the header (e.g. `NI`, `NJ`, `NK`, `M`, `N`). Do not use runtime sizes or VLAs.
- **Tile sizes / constants used in task signatures** must be visible at file scope: `#define TILE_SIZE ...` or `static const int TILE_SIZE = ...;` **before** task declarations. Never use a `const int` declared only inside the top function in task parameter arrays.
- Match types exactly between top-level locals and task parameters.
- Tasks must not declare `#pragma HLS INTERFACE` and must not be alternate top functions.
- **Declare every loop index** (`int i`, `int j`, `int k`, …). Undeclared loop variables fail g++ compile.
- **Label every `for` loop** with a descriptive name before the loop header (e.g. `load_A: for (...)`, `compute_tile: for (...)`). Labels are required for audit and HLS log readability.

## MANDATORY — `m_axi` bundles for DATAFLOW overlap (skill: distinct gmemN per port)
**Before writing tasks**, read every top-level `#pragma HLS INTERFACE m_axi` and note each port's `bundle=` name.

**When refactoring for DATAFLOW**, assign **distinct** `bundle=gmem0`, `gmem1`, `gmem2`, … on the **top-level** `m_axi` pragmas — one unique bundle per pointer port. This enables parallel load/compute/store tasks without serializing on one `gmem`.

**Rule:** across **concurrent** DATAFLOW processes, each `m_axi` bundle allows **at most one reader** and **at most one writer**.

### If flash kernel still has all ports on shared `bundle=gmem`
You **must rebundle** top-level pragmas to `gmem0`, `gmem1`, … (you may change only the `bundle=` suffix on existing `m_axi` INTERFACE lines). Then use **separate** load/store tasks per port when overlap is intended.

Do **not** keep shared `gmem` and add parallel per-port loaders — that fails csynth (HLS 200-1013 / 200-984).

### Fallback when rebundling cannot enable overlap
Use the fused pattern:
1. **One fused read task** — sequentially copies every input port into distinct local buffers inside a **single** task (pipelined loops OK).
2. **One or more compute tasks** — **locals/streams only**, **no** top-level port pointers.
3. **One fused write task** — writes every output port from locals inside a **single** task.

### Other bus rules (non-negotiable)
- **Never** pass top-level `m_axi` port arrays into compute tasks.
- **Never** have two concurrent tasks both read (or both write) the same port on the same bundle.
- When a port is both read and written: read into `local_in`, compute into `local_out`, store writes `local_out` back — **never** let load and compute both write the same local array.

## MANDATORY — on-chip buffer ownership (prevents HLS 200-979 / 200-779)
Every local array crossing concurrent DATAFLOW tasks must have **exactly one writer** and **exactly one reader**:
- One task writes a buffer; a different task reads it. **No third task** may access it.
- **Fan-out forbidden:** if `compute_A` writes `local_tmp`, only **one** downstream task may read `local_tmp`. If both `compute_B` and `store_tmp` need it, **merge** `compute_B` + any other consumers of the same local into **one** `compute_fused_task`, or use `hls::stream` — never `compute_B(..., local_tmp)` and `store_tmp(..., local_tmp)` concurrently.
- **Shared matrix across compute phases:** if two compute tasks both read `local_A` (dual matvec, chained GEMM, mean+stddev+normalize), merge them into **one** compute task with sequential loops — not separate tasks.
- **No inline copy loops inside DATAFLOW** to work around fan-out — an unlabeled `for` copying `local_in` → `local_copy` inside the DATAFLOW region is another reader/writer process and still fails (HLS 200-779).
- If load initializes data and compute also updates it → use **two arrays** (`buf_in` / `buf_out` or ping-pong banks). **Never** share one array between a writer in load and a writer in compute.
- Use **ping-pong** (`buf0`/`buf1`) for streaming tiles.
- Inner loops inside each task may use `#pragma HLS PIPELINE II=1`.

## MANDATORY — temporal & tile loops (prevents HLS 200-979 / 200-1013)
- **Time-step / iteration loops** (`for (t = 0; t < TSTEPS; t++)`, stencil phases): put the **entire** t-loop inside **one** `compute_*` task. DATAFLOW top level is only `load → compute_all_timesteps → store`.
- **Never** place a `for (tile…)` loop **inside** `#pragma HLS DATAFLOW` if the body calls tasks that read/write `m_axi` ports — HLS creates parallel readers on one bundle (HLS 200-1013).
- Tiled kernels: use **fused load** (one task, all inputs), **compute on locals**, **fused store** — OR `hls::stream` pipeline `load_tile → compute_tile → store_tile` with FIFO depth ≥ 2.
- **Dual-pass matrix kernels** (two phases over same `A` with different layout): **one** `fused_load_task` reads `A` once and fills row + transposed locals; **no** `for (i0 += TILE)` inside DATAFLOW calling `load_tile_A`.

## Performance — prefer parallelism & resource use (when csynth-legal)
- `#pragma HLS ARRAY_PARTITION` on hot locals (cyclic/block, factor = UNROLL factor).
- `#pragma HLS UNROLL` on innermost dot-product / reduction loops (factor 4–8 typical).
- Run **parallel** `load_*` tasks when each touches a **different** `m_axi` port (`gmem0`, `gmem1`, …).
- Use `hls::stream` between tile stages for load(t+1) ‖ compute(t) overlap.
- Target **lower latency** even at higher BRAM/DSP — do not minimize resources at the cost of leaving DATAFLOW idle.

## DATAFLOW structure (required)
- Place **one** `#pragma HLS DATAFLOW` at the top kernel body — **required, not optional**.
- Call **at least three** static tasks: load (read `m_axi` → locals), compute (locals only), store (locals → `m_axi`). More tasks are fine when bundles and buffer ownership allow overlap.
- **Do not** put DATAFLOW inside an outer tile loop unless each iteration uses private ping-pong buffers (see skills).
- Dependences flow through data (locals/streams); tasks run in parallel when bundle and buffer rules above are satisfied.

## Forbidden patterns (guaranteed csynth failure or wrong step)
- **Omitting** `#pragma HLS DATAFLOW` or running tasks sequentially without DATAFLOW.
- **Fan-out:** one local array with one writer and **two+ reader** tasks under concurrent DATAFLOW (includes `compute_x` + `compute_w` both reading `local_A_out`, or `compute_D` + `store_tmp` both reading `local_tmp`).
- **Inline copy loop** inside DATAFLOW to duplicate a local for another task.
- **Separate compute tasks** that both read the same `local_A` or `local_data_in` (use fused compute task instead).
- **Temporal t-loop** inside DATAFLOW calling multiple compute tasks that alternate writes on the same locals.
- **Tile for-loop** inside DATAFLOW calling `load_*` tasks on the same `m_axi` port each iteration.
- Parallel `load_A_task` + `load_B_task` + … when those ports share one `m_axi` bundle **without** rebundling to distinct `gmemN` first.
- Parallel `store_X_task` + `store_Y_task` on the same bundle.
- `compute_task(..., local_C, ...)` where `load_task` also wrote `local_C`.
- Top-level port arrays passed into compute while another concurrent task reads/writes those ports.
- Tile constants in task prototypes but defined only inside the top function.
- Undeclared loop variables.

## Pre-output checklist — verify every item before returning code
0. Top kernel body contains `#pragma HLS DATAFLOW` with ≥3 static task calls (load, compute, store).
1. Top-level `m_axi` ports use distinct `gmem0`, `gmem1`, … **or** fused single load / single store on shared bundle.
2. For each `m_axi` bundle: ≤1 concurrent reader task, ≤1 concurrent writer task.
3. Every local array: exactly one writer, exactly one reader among concurrent tasks (no fan-out).
4. No `for (t…)` or `for (tile…)` inside DATAFLOW that violates rules above.
5. No compute task takes top-level `m_axi` port pointers.
6. Hot locals have ARRAY_PARTITION + inner UNROLL where applicable.
7. All task array bounds use file-scope macros/constants; all loop indices declared.
8. Every `for` loop has a descriptive label (`name: for (...)`).
9. Top function signature unchanged except allowed `bundle=gmemN` rebinding on `m_axi` pragmas.

## Output
Return **one** fenced block only:
```kernel
... full kernel source ...
```
"""

_INITIAL_USER_LEGACY = """Refactor this flash-selected kernel with **`#pragma HLS DATAFLOW`** and static task functions.

**Mandatory:** `#pragma HLS DATAFLOW` always. Distinct `gmemN` per port. One writer + one reader per local (use `hls::stream` for fan-out). Put time-step loops inside one compute task. No tile/m_axi loops inside DATAFLOW. ARRAY_PARTITION + UNROLL on hot loops.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Selected flash kernel (preserve top-level signature and interfaces)
```cpp
{kernel_code}
```

Return a single ```kernel``` block. Verify the pre-output checklist in the system prompt before responding.
"""

_INITIAL_USER_RICH = """## Task — DATAFLOW refactor

Refactor the flash-selected kernel below into **static task functions** connected under **`#pragma HLS DATAFLOW`**.

Your goal is **real stage overlap** when legal (parallel loads on distinct `gmemN`, stream/pipeline tiles), not decorative DATAFLOW that only serializes load → compute → store.

### What you must deliver
1. Keep the exact top-level `extern "C"` signature and INTERFACE pragmas (rebundling `bundle=gmem` → `gmem0`, `gmem1`, … is allowed).
2. Exactly one `#pragma HLS DATAFLOW` in the top kernel body with ≥3 task calls (load, compute, store).
3. Distinct `gmemN` per `m_axi` port when concurrent load/store tasks should overlap.
4. **Single writer + single reader** per local array across concurrent DATAFLOW tasks — merge shared consumers into one `compute_fused_task` or use `hls::stream`.
5. Put **all time-step loops** (`for (t …)`) inside **one** compute task — never multiple alternating compute tasks under DATAFLOW.
6. **No tile `for` loops inside DATAFLOW** that call `m_axi` load/store tasks — fuse loads, stream tiles, or move the tile loop outside.
7. Apply **ARRAY_PARTITION** + inner **UNROLL** on hot compute loops.
8. Apply **every matching FLASH skill** below; **avoid** skills are hard rejections.

Work through the **pre-output checklist** in the system message before returning code.

{skills_block}

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Selected flash kernel (preserve top-level signature and interfaces)
```cpp
{kernel_code}
```

Return a single ```kernel``` block only.
"""

# Back-compat alias used by older tests/docs.
_INITIAL_USER = _INITIAL_USER_LEGACY

_REPAIR_USER_LEGACY = """Repair the **DATAFLOW** kernel after a validation failure.

Keep the **exact** top-level `extern "C"` kernel signature and interface pragmas.
**Keep `#pragma HLS DATAFLOW`** — fix task structure; do not remove DATAFLOW or fall back to sequential execution.

## Failure
Stage: {stage}
```
{error}
```

Apply the **mandatory** fix for the failure class:

**HLS 200-979 / 200-779 / single reader/writer / only be written in one process:**
- **First choice:** merge all compute phases that share a local into **one** `compute_fused_task` (chained GEMMs, dual matvec on same `A`, mean→stddev→normalize, `compute_x`+`compute_w` on same `local_A_out`).
- If `compute_D` and `store_tmp` both need `local_tmp`: fused compute does both GEMMs then writes `local_tmp_out` for store, OR store runs inside fused compute — never two concurrent readers on `local_tmp`.
- Remove any **inline for-loop copy** inside DATAFLOW (use fused compute instead).
- Split load/compute handoff buffers (`buf_in` / `buf_out`, ping-pong) only when load and compute both touch the same array.
- Move `for (t…)` timestep loops into a **single** compute task (not multiple tasks under DATAFLOW).

**HLS 200-1013 / 200-984 with distinct gmemN already assigned:**
- A **tile for-loop inside DATAFLOW** calling load tasks is the usual cause — fuse loads into one task or use stream tile pipeline.
- For **two-pass matrix** kernels: `fused_load_task` reads `A` once into row + transposed locals; phase compute tasks use locals only.
- Never have multiple concurrent readers on the same port even with gmemN rebundling.

**HLS 200-1013 / 200-984 / gmem / bundled bus interface:**
- **First:** rebundle top-level `m_axi` to distinct `gmem0`, `gmem1`, … and use separate load/store tasks per port.
- **Only if rebundling cannot help:** collapse to **one fused read task** and **one fused write task** on that bundle (still inside `#pragma HLS DATAFLOW`).
- Remove top-level port pointers from all compute tasks.

**Compile errors (undeclared variable, TILE_SIZE, task prototype):**
- Declare all loop indices; move tile constants to file scope before task declarations.

Re-verify the pre-output checklist (including `#pragma HLS DATAFLOW` present) before returning code.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Current kernel
```cpp
{kernel_code}
```

Return a corrected single ```kernel``` block.
"""

_REPAIR_USER_RICH = """## Task — repair DATAFLOW kernel after validation failure

Keep the **exact** top-level `extern "C"` kernel signature and interface pragmas.
**Keep `#pragma HLS DATAFLOW`** — fix task structure; do not remove DATAFLOW or fall back to sequential execution.

Re-apply every matching FLASH skill below when fixing the failure.

{skills_block}

## Failure
Stage: {stage}
```
{error}
```

Apply the **mandatory** fix for the failure class (see system message for full rules):

**HLS 200-979 / 200-779 / single reader/writer / only be written in one process:**
- **First choice:** merge all compute phases that share a local into **one** `compute_fused_task` (chained GEMMs, dual matvec on same `A`, mean→stddev→normalize, `compute_x`+`compute_w` on same `local_A_out`).
- If `compute_D` and `store_tmp` both need `local_tmp`: fused compute does both GEMMs then writes `local_tmp_out` for store, OR store runs inside fused compute — never two concurrent readers on `local_tmp`.
- Remove any **inline for-loop copy** inside DATAFLOW (use fused compute instead).
- Split load/compute handoff buffers (`buf_in` / `buf_out`, ping-pong) only when load and compute both touch the same array.
- Move `for (t…)` timestep loops into a **single** compute task (not multiple tasks under DATAFLOW).

**HLS 200-1013 / 200-984 with distinct gmemN already assigned:**
- A **tile for-loop inside DATAFLOW** calling load tasks is the usual cause — fuse loads into one task or use stream tile pipeline.
- For **two-pass matrix** kernels: `fused_load_task` reads `A` once into row + transposed locals; phase compute tasks use locals only.
- Never have multiple concurrent readers on the same port even with gmemN rebundling.

**HLS 200-1013 / 200-984 / gmem / bundled bus interface:**
- **First:** rebundle top-level `m_axi` to distinct `gmem0`, `gmem1`, … and use separate load/store tasks per port.
- **Only if rebundling cannot help:** collapse to **one fused read task** and **one fused write task** on that bundle (still inside `#pragma HLS DATAFLOW`).
- Remove top-level port pointers from all compute tasks.

**Compile errors (undeclared variable, TILE_SIZE, task prototype):**
- Declare all loop indices; move tile constants to file scope before task declarations.

Re-verify the pre-output checklist (including `#pragma HLS DATAFLOW` present) before returning code.

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Current kernel
```cpp
{kernel_code}
```

Return a corrected single ```kernel``` block.
"""

_REPAIR_USER = _REPAIR_USER_LEGACY


def dataflow_enabled() -> bool:
    return os.getenv("C2HLS_POST_FLASH_DATAFLOW", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_DATAFLOW_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def resolve_dataflow_skills_path() -> Path:
    """Skill file for DATAFLOW refactor (default: flash overlay JSON)."""
    from c2hls_paths import FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON

    for key in ("C2HLS_DATAFLOW_SKILL_ENTRIES_JSON", "C2HLS_FLASH_SKILL_ENTRIES_JSON"):
        raw = os.getenv(key, "").strip()
        if raw:
            return Path(raw)
    return FLASH_NO_RMW_M_AXI_SKILL_ENTRIES_JSON


def validate_dataflow_skill_entries(path: Path) -> list[str]:
    """Return validation errors; empty list means the file is loadable."""
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
    from skill_library import _coerce_skill_entry

    errors: list[str] = []
    for entry in skills_raw:
        if _coerce_skill_entry(entry) is None:
            sid = entry.get("id") if isinstance(entry, dict) else "?"
            errors.append(f"{path.name}: malformed skill entry {sid}")
    return errors


def build_dataflow_skills_prompt_block(
    path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    """Load flash skill entries and render them for the DATAFLOW system prompt."""
    from skill_library import (
        SkillLibrary,
        _load_packaged_skills,
        global_skills_for_prompt,
        render_skill_for_prompt,
    )

    skills_path = path or resolve_dataflow_skills_path()
    errors = validate_dataflow_skill_entries(skills_path)
    if errors:
        raise ValueError("; ".join(errors))

    lib = SkillLibrary()
    for sk in _load_packaged_skills(skills_path):
        lib.add(sk, overwrite=True)
    skills = global_skills_for_prompt(lib, include_avoids=True)
    header = (
        "## FLASH HLS OPTIMIZATION SKILLS (mandatory for DATAFLOW refactor)\n\n"
        "**DATAFLOW step rule:** always emit `#pragma HLS DATAFLOW` with static load/compute/store "
        "tasks. Skills that mention avoiding 'decorative DATAFLOW' mean fix overlap "
        "(distinct `gmemN` per port, `hls::stream`, ping-pong, single-writer/single-reader "
        "locals) — **never** omit DATAFLOW or run tasks sequentially without it.\n\n"
        "Apply every matching skill below when refactoring loops, pipelines, m_axi "
        "access, tiling, and DATAFLOW task structure. Avoid rules are mandatory "
        "rejections — do not emit code that violates them.\n"
        f"Source: {skills_path.name} ({len(skills)} skills)\n\n"
    )
    body = "\n\n".join(render_skill_for_prompt(sk) for sk in skills)
    meta: dict[str, Any] = {
        "skills_path": str(skills_path),
        "skill_count": len(skills),
        "skill_ids": [sk.id for sk in skills],
    }
    return header + body, meta


def resolve_prompt_policy(raw: Optional[str] = None) -> str:
    """Return a validated DATAFLOW prompt policy name."""
    value = (raw if raw is not None else os.getenv("C2HLS_POST_FLASH_PROMPT_POLICY", "")).strip()
    if not value:
        value = DEFAULT_PROMPT_POLICY
    if value not in PROMPT_POLICIES:
        allowed = ", ".join(PROMPT_POLICIES)
        raise ValueError(f"unknown DATAFLOW prompt policy {value!r} (expected one of: {allowed})")
    return value


def kernel_bundle_dir_name(prompt_policy: str) -> str:
    return f"kernel_bundle_pp-{resolve_prompt_policy(prompt_policy)}"


def format_results_root_name(
    stamp: str,
    *,
    results_suffix: str = "",
    prompt_policy: str = DEFAULT_PROMPT_POLICY,
) -> str:
    """Directory name under matrix root for a packaged DATAFLOW run."""
    policy = resolve_prompt_policy(prompt_policy)
    parts = [f"post_flash_dataflow_results_{stamp}", f"pp-{policy}"]
    suffix = results_suffix.strip()
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


@dataclass(frozen=True)
class DataflowPromptBundle:
    prompt_policy: str
    system_prompt: str
    initial_user_template: str
    repair_user_template: str
    skills_meta: dict[str, Any]


def compose_dataflow_prompts(
    prompt_policy: Optional[str] = None,
) -> DataflowPromptBundle:
    """Build system/user templates for the selected prompt policy."""
    policy = resolve_prompt_policy(prompt_policy)
    skills_block, skills_meta = build_dataflow_skills_prompt_block()
    if policy == "system_skills":
        system_prompt = f"{_SYSTEM.rstrip()}\n\n{skills_block}"
        initial_user_template = _INITIAL_USER_LEGACY
        repair_user_template = _REPAIR_USER_LEGACY
    elif policy == "user_skills":
        system_prompt = _SYSTEM.rstrip()
        initial_user_template = _INITIAL_USER_RICH
        repair_user_template = _REPAIR_USER_RICH
    else:  # pragma: no cover
        raise ValueError(f"unhandled prompt policy: {policy}")
    meta = dict(skills_meta)
    meta["prompt_policy"] = policy
    return DataflowPromptBundle(
        prompt_policy=policy,
        system_prompt=system_prompt,
        initial_user_template=initial_user_template,
        repair_user_template=repair_user_template,
        skills_meta=meta,
    )


def format_dataflow_initial_user(
    bundle: DataflowPromptBundle,
    *,
    benchmark_context: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
) -> str:
    skills_block, _ = build_dataflow_skills_prompt_block()
    kwargs: dict[str, str] = {
        "benchmark_context": benchmark_context,
        "header_name": header_name,
        "header_code": header_code[:12000],
        "kernel_code": kernel_code[:120000],
    }
    if bundle.prompt_policy == "user_skills":
        kwargs["skills_block"] = skills_block
    return bundle.initial_user_template.format(**kwargs)


def format_dataflow_repair_user(
    bundle: DataflowPromptBundle,
    *,
    stage: str,
    error: str,
    benchmark_context: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
) -> str:
    skills_block, _ = build_dataflow_skills_prompt_block()
    kwargs: dict[str, str] = {
        "stage": stage,
        "error": error[:8000],
        "benchmark_context": benchmark_context,
        "header_name": header_name,
        "header_code": header_code[:12000],
        "kernel_code": kernel_code[:120000],
    }
    if bundle.prompt_policy == "user_skills":
        kwargs["skills_block"] = skills_block
    return bundle.repair_user_template.format(**kwargs)


def compose_dataflow_system_prompt(
    prompt_policy: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    bundle = compose_dataflow_prompts(prompt_policy)
    return bundle.system_prompt, bundle.skills_meta


def sanitize_kernel_source(code: str) -> str:
    """Strip stray fence labels the generic parser may leave behind."""
    code = (code or "").strip()
    if code.lower() == "kernel":
        return ""
    if code.lower().startswith("kernel\n"):
        code = code.split("\n", 1)[1].lstrip()
    return code


def extract_kernel_block(text: str) -> str:
    """Parse ```kernel ...``` or first generic cpp fence."""
    if not text:
        return ""
    blocks = extract_labeled_cpp_blocks(text)
    if blocks.get("kernel"):
        return sanitize_kernel_source(blocks["kernel"])
    generic = re.findall(r"```(?:cpp|c\+\+|c|hls)\s*(.*?)```", text, flags=re.DOTALL)
    if generic:
        return sanitize_kernel_source(generic[0])
    labeled_only = re.findall(
        r"```\s*kernel\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if labeled_only:
        return sanitize_kernel_source(labeled_only[0])
    return ""


def artifact_paths(cell_dir: Path, bench: str) -> dict[str, Path]:
    base = f"{bench}_{STEP_TAG}"
    return {
        "kernel": cell_dir / f"{base}.cpp",
        "report": cell_dir / f"{base}_report.json",
        "result": cell_dir / f"{base}_result.json",
        "history": cell_dir / f"{base}_history.json",
    }


VALIDATE_SCHEMA = "post_flash_dataflow_validate_v1"


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


def load_existing_validate_result(result_path: Path) -> Optional[dict[str, Any]]:
    if not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        isinstance(data, dict)
        and data.get("schema") == VALIDATE_SCHEMA
        and data.get("success") is True
    ):
        return data
    return None


def _find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    for idx in range(open_index, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_workload_extern_block(code: str) -> Optional[tuple[int, int, list[str]]]:
    """Return (start, end_exclusive, interface_pragmas) for a workload wrapper block."""
    for match in re.finditer(r'extern\s+"C"\s*\{', code):
        block_start = match.start()
        brace_open = match.end() - 1
        block_end = _find_matching_brace(code, brace_open)
        if block_end < 0:
            continue
        block = code[block_start:block_end + 1]
        if not re.search(r"\bvoid\s+workload\s*\(", block):
            continue
        pragmas = re.findall(r"#pragma\s+HLS\s+INTERFACE[^\n]*", block)
        return block_start, block_end + 1, pragmas
    return None


def _top_function_has_interface(code: str, top_function: str) -> bool:
    match = re.search(rf"\bvoid\s+{re.escape(top_function)}\s*\(", code)
    if not match:
        return False
    brace = code.find("{", match.end())
    if brace < 0:
        return False
    chunk = code[brace:brace + 4000]
    return bool(re.search(r"#pragma\s+HLS\s+INTERFACE", chunk))


def _inject_pragmas_after_top_opening(code: str, top_function: str, pragmas: list[str]) -> str:
    match = re.search(
        rf"\bvoid\s+{re.escape(top_function)}\s*\([^)]*\)\s*\{{",
        code,
        flags=re.DOTALL,
    )
    if not match or not pragmas:
        return code
    insert_at = match.end()
    block = "\n" + "\n".join(pragmas) + "\n"
    return code[:insert_at] + block + code[insert_at:]


def prepare_recovered_kernel_for_validate(
    code: str,
    top_function: str,
) -> tuple[str, dict[str, Any]]:
    """Strip duplicate workload() wrapper and move INTERFACE pragmas onto metadata top."""
    code = sanitize_kernel_source(code)
    meta: dict[str, Any] = {
        "prepared": False,
        "stripped_workload": False,
        "moved_interface_pragmas": False,
    }
    if top_function == "workload":
        return code, meta
    extracted = _extract_workload_extern_block(code)
    if extracted is None:
        return code, meta
    start, end, pragmas = extracted
    new_code = code[:start].rstrip() + "\n"
    meta["stripped_workload"] = True
    meta["prepared"] = True
    if pragmas and not _top_function_has_interface(new_code, top_function):
        new_code = _inject_pragmas_after_top_opening(new_code, top_function, pragmas)
        meta["moved_interface_pragmas"] = True
    return new_code.rstrip() + "\n", meta


@dataclass
class DataflowOutcome:
    bench: str
    success: bool
    cell_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def run_dataflow_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    skip_existing: bool = True,
    prompt_policy: Optional[str] = None,
) -> DataflowOutcome:
    from c2hls import (
        _load_benchmark_inputs,
        _run_synth_csim_cosim,
        compile_check_cpp,
    )
    from c2hls_temp import join_temp_tag

    inputs = _load_benchmark_inputs(str(bench_dir))
    kernel_path, kernel_role = resolve_selected_kernel(cell_dir, bench)
    if kernel_path is None:
        return DataflowOutcome(bench, False, str(cell_dir), "no selected/final kernel cpp")

    selected_kernel = kernel_path.read_text(encoding="utf-8")
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = meta.get("translated_hls_top") or meta.get("hls_top") or meta.get("kernel_top") or "workload"
    benchmark_context = inputs.get("benchmark_context", "")
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    paths = artifact_paths(cell_dir, bench)
    if skip_existing:
        existing = load_existing_result(paths["result"])
        if existing is not None:
            return DataflowOutcome(bench, True, str(cell_dir), result=existing)

    history: list[dict[str, str]] = []
    attempts: list[dict[str, Any]] = []
    kernel_code = ""
    success = False
    last_error = ""
    synth_report: dict[str, Any] = {}
    csim_summary: Optional[dict[str, Any]] = None

    prompt_bundle = compose_dataflow_prompts(prompt_policy)
    system = prompt_bundle.system_prompt
    skills_meta = prompt_bundle.skills_meta
    user = format_dataflow_initial_user(
        prompt_bundle,
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code,
        kernel_code=selected_kernel,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    reply = orchestrator._call_llm(messages)
    history.extend([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reply},
    ])
    kernel_code = extract_kernel_block(reply)

    contract_attempts: list[dict[str, Any]] = []
    contract_passed = True
    last_contract_report = None
    if kernel_code and contract_check_enabled():
        for c_round in range(contract_round_limit()):
            llm_report = None
            audit_user = format_contract_audit_user(kernel_code)
            audit_messages = [
                {"role": "system", "content": contract_audit_system_prompt()},
                {"role": "user", "content": audit_user},
            ]
            audit_reply = orchestrator._call_llm(audit_messages)
            history.extend([
                {"role": "system", "content": contract_audit_system_prompt()},
                {"role": "user", "content": audit_user},
                {"role": "assistant", "content": audit_reply},
            ])
            llm_report = parse_llm_contract_report(audit_reply)
            report = hybrid_contract_check(
                kernel_code,
                top_function=top_function,
                llm_report=llm_report,
            )
            contract_attempts.append({
                "round": c_round,
                "passed": report.passed,
                "report": report.to_dict(),
            })
            contract_passed = report.passed
            last_contract_report = report
            if report.passed:
                break
            if c_round >= contract_round_limit() - 1:
                break

            skills_block, _ = build_dataflow_skills_prompt_block()
            fix_user = format_contract_fix_user(
                prompt_policy=prompt_bundle.prompt_policy,
                breaches_json=format_contract_breaches_json(report),
                benchmark_context=benchmark_context,
                header_name=header_name,
                header_code=header_code,
                kernel_code=kernel_code,
                skills_block=skills_block,
            )
            fix_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": fix_user},
            ]
            fix_reply = orchestrator._call_llm(fix_messages)
            history.extend([
                {"role": "user", "content": fix_user},
                {"role": "assistant", "content": fix_reply},
            ])
            extracted = extract_kernel_block(fix_reply)
            if extracted:
                kernel_code = extracted

    if not contract_passed and contract_attempts:
        last_error = contract_failure_message(last_contract_report) if last_contract_report else "DATAFLOW contract check failed"
        attempts.append({
            "attempt": 0,
            "stage": "contract",
            "success": False,
            "error": last_error[:4000],
        })
    else:
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
                    tag = join_temp_tag(bench, STEP_TAG, f"a{attempt}")
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
                        log_prefix="[dataflow]",
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
                        if csim_pass:
                            success = True
                            synth_report = synth.get("report") or {}
                        else:
                            attempt_error = (csim_summary or {}).get("error") or "csim failed"
                            stage = "csim"
                    else:
                        attempt_error = synth.get("error") or "csynth failed"

            attempts.append({
                "attempt": attempt,
                "stage": stage,
                "success": success,
                "error": attempt_error[:4000],
            })
            last_error = attempt_error
            if success:
                break
            if attempt >= repair_round_limit() - 1:
                break

            repair_user = format_dataflow_repair_user(
                prompt_bundle,
                stage=stage,
                error=attempt_error,
                benchmark_context=benchmark_context,
                header_name=header_name,
                header_code=header_code,
                kernel_code=kernel_code,
            )
            repair_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": repair_user},
            ]
            reply = orchestrator._call_llm(repair_messages)
            history.extend([
                {"role": "user", "content": repair_user},
                {"role": "assistant", "content": reply},
            ])
            extracted = extract_kernel_block(reply)
            if extracted:
                kernel_code = extracted

    kernel_code = sanitize_kernel_source(kernel_code)

    result_payload: dict[str, Any] = {
        "schema": "post_flash_dataflow_v2",
        "benchmark": bench,
        "success": success,
        "error": last_error if not success else "",
        "source_kernel": str(kernel_path.name),
        "source_kernel_role": kernel_role,
        "testbench": "benchmark_original",
        "attempts": attempts,
        "synth_report": synth_report,
        "csim": csim_summary,
        "latency_cycles": synth_report.get("latency_cycles"),
        "selected_from_flash": kernel_role,
        "skills": skills_meta,
        "prompt_policy": prompt_bundle.prompt_policy,
        "contract_check_enabled": contract_check_enabled(),
        "contract_rounds": contract_round_limit(),
        "contract_attempts": contract_attempts,
        "contract_passed": contract_passed,
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
        "model": orchestrator.gpt_model,
        "prompt_policy": prompt_bundle.prompt_policy,
        "messages": history,
    }, indent=2), encoding="utf-8")
    _write_cell_manifest(cell_dir, bench, kernel_path, result_payload)
    if success:
        from post_flash_pragma_opt import maybe_chain_pragma_opt

        pragma_outcome = maybe_chain_pragma_opt(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orchestrator,
            source_role="dataflow",
            skip_existing=True,
        )
        if pragma_outcome is not None:
            result_payload["pragma_opt_chain"] = {
                "success": pragma_outcome.success,
                "error": pragma_outcome.error,
                "result": pragma_outcome.result,
            }
            paths["result"].write_text(
                json.dumps(result_payload, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    return DataflowOutcome(bench, success, str(cell_dir), last_error, result_payload)


def _write_cell_manifest(
    cell_dir: Path,
    bench: str,
    kernel_path: Path,
    result_payload: dict[str, Any],
) -> None:
    manifest_path = cell_dir / f"{bench}_post_flash_dataflow.json"
    manifest_path.write_text(json.dumps({
        "schema": "post_flash_dataflow_manifest_v2",
        "benchmark": bench,
        "source_kernel": str(kernel_path),
        "success": result_payload.get("success"),
        "result": artifact_paths(cell_dir, bench)["result"].name,
    }, indent=2) + "\n", encoding="utf-8")


def configure_post_flash_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    skills_path = resolve_dataflow_skills_path()
    errors = validate_dataflow_skill_entries(skills_path)
    if errors:
        raise ValueError(
            f"DATAFLOW skill file invalid ({skills_path}): {'; '.join(errors)}"
        )
    os.environ.setdefault("C2HLS_DATAFLOW_SKILL_ENTRIES_JSON", str(skills_path))
    os.environ.setdefault("C2HLS_PACKAGED_SKILLS_JSON", str(skills_path))
    os.environ.setdefault("C2HLS_PACKAGED_SKILLS_ONLY", "1")
    # Sole skill source — do not double-merge overlay on itself.
    if os.getenv("C2HLS_FLASH_SKILL_ENTRIES_JSON", "").strip() == str(skills_path):
        os.environ.pop("C2HLS_FLASH_SKILL_ENTRIES_JSON", None)


def prompt_text_for_docs(prompt_policy: Optional[str] = None) -> dict[str, str]:
    """Return prompts for documentation / inspection."""
    bundle = compose_dataflow_prompts(prompt_policy)
    skills_block, _ = build_dataflow_skills_prompt_block()
    sample_user = format_dataflow_initial_user(
        bundle,
        benchmark_context="(benchmark context inserted at runtime)",
        header_name="kernel.h",
        header_code="/* header */",
        kernel_code="/* selected flash kernel */",
    )
    sample_repair = format_dataflow_repair_user(
        bundle,
        stage="csynth",
        error="(validation error inserted at runtime)",
        benchmark_context="(benchmark context inserted at runtime)",
        header_name="kernel.h",
        header_code="/* header */",
        kernel_code="/* current kernel */",
    )
    from dataflow_contract_check import format_contract_audit_user, format_contract_fix_user

    sample_contract_audit = format_contract_audit_user("/* current kernel */")
    sample_contract_fix = format_contract_fix_user(
        prompt_policy=bundle.prompt_policy,
        breaches_json='{"schema": "dataflow_contract_breach_v1", "passed": false, "breaches": []}',
        benchmark_context="(benchmark context inserted at runtime)",
        header_name="kernel.h",
        header_code="/* header */",
        kernel_code="/* current kernel */",
        skills_block=skills_block if bundle.prompt_policy == "user_skills" else "",
    )
    return {
        "prompt_policy": bundle.prompt_policy,
        "system": bundle.system_prompt,
        "initial_user": sample_user,
        "repair_user": sample_repair,
        "contract_audit_system": contract_audit_system_prompt(),
        "contract_audit_user": sample_contract_audit,
        "contract_fix_user": sample_contract_fix,
        "skills_path": bundle.skills_meta.get("skills_path", ""),
        "skill_count": str(bundle.skills_meta.get("skill_count", 0)),
        "skills_in_system": str(bundle.prompt_policy == "system_skills"),
        "skills_in_user": str(bundle.prompt_policy == "user_skills"),
        "skills_block_chars": str(len(skills_block)),
    }


def validate_recovered_dataflow_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    skip_existing: bool = True,
    prepare_kernel: bool = True,
    part: str = "",
    clock_ns: float = 0,
) -> DataflowOutcome:
    """Run csim + csynth on an existing {bench}_dataflow.cpp (no LLM)."""
    from c2hls import _load_benchmark_inputs, _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag
    from hls_eval import DEFAULT_CLOCK_NS, DEFAULT_PART

    paths = artifact_paths(cell_dir, bench)
    kernel_path = paths["kernel"]
    if not kernel_path.is_file():
        return DataflowOutcome(bench, False, str(cell_dir), f"missing {kernel_path.name}")

    if skip_existing:
        existing = load_existing_validate_result(paths["result"])
        if existing is not None:
            return DataflowOutcome(bench, True, str(cell_dir), result=existing)

    inputs = _load_benchmark_inputs(str(bench_dir))
    header_code = inputs.get("header_code", "")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs["meta"]
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or "workload"
    )
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = part or meta.get("part") or DEFAULT_PART
    clock_ns = clock_ns or meta.get("clock_ns") or DEFAULT_CLOCK_NS

    kernel_code = kernel_path.read_text(encoding="utf-8")
    prep_meta: dict[str, Any] = {}
    if prepare_kernel:
        kernel_code, prep_meta = prepare_recovered_kernel_for_validate(kernel_code, top_function)

    stage = "compile_kernel"
    last_error = ""
    success = False
    synth_report: dict[str, Any] = {}
    csim_summary: Optional[dict[str, Any]] = None

    ok, err = compile_check_cpp(
        kernel_code,
        header_code,
        header_name,
        extra_files=extra_files,
    )
    if not ok:
        last_error = err
    elif not testbench_code:
        stage = "testbench"
        last_error = "benchmark has no testbench for csim"
    else:
        tag = join_temp_tag(bench, STEP_TAG, "validate")
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
            log_prefix="[dataflow-validate]",
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
            if csim_pass:
                success = True
                synth_report = synth.get("report") or {}
            else:
                last_error = (csim_summary or {}).get("error") or "csim failed"
                stage = "csim"
        else:
            last_error = synth.get("error") or "csynth failed"

    result_payload: dict[str, Any] = {
        "schema": VALIDATE_SCHEMA,
        "benchmark": bench,
        "success": success,
        "error": last_error if not success else "",
        "source": "recovered_dataflow_cpp",
        "kernel": kernel_path.name,
        "top_function": top_function,
        "testbench": "benchmark_original",
        "stage": stage,
        "prepare": prep_meta,
        "synth_report": synth_report,
        "csim": csim_summary,
        "latency_cycles": synth_report.get("latency_cycles"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    artifacts: dict[str, str] = {}
    if kernel_code:
        if prep_meta.get("prepared"):
            kernel_path.write_text(kernel_code, encoding="utf-8")
        result_payload["kernel_sha256"] = sha256_text(kernel_code)
        artifacts["kernel"] = kernel_path.name
    if synth_report:
        paths["report"].write_text(json.dumps(synth_report, indent=2, default=str) + "\n", encoding="utf-8")
        artifacts["report"] = paths["report"].name
    if artifacts:
        result_payload["artifacts"] = artifacts

    paths["result"].write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")
    _write_cell_manifest(cell_dir, bench, kernel_path, result_payload)
    return DataflowOutcome(bench, success, str(cell_dir), last_error, result_payload)


def recover_kernels_from_history(
    matrix_root: Path,
    *,
    bench_filter: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Re-extract LLM kernels from *_dataflow_history.json and write *_dataflow.cpp."""
    recovered: list[dict[str, Any]] = []
    for cell in discover_matrix_cells(matrix_root):
        bench = cell["bench"]
        if bench_filter and bench not in bench_filter:
            continue
        cell_dir = Path(cell["cell_dir"])
        paths = artifact_paths(cell_dir, bench)
        history_path = paths["history"]
        if not history_path.is_file():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        messages = history.get("messages") or []
        kernel_code = ""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            extracted = extract_kernel_block(msg.get("content", ""))
            if extracted:
                kernel_code = extracted
                break
        if not kernel_code:
            recovered.append({"bench": bench, "recovered": False, "reason": "no kernel in history"})
            continue
        paths["kernel"].write_text(kernel_code, encoding="utf-8")
        recovered.append({
            "bench": bench,
            "recovered": True,
            "kernel": str(paths["kernel"]),
            "kernel_sha256": sha256_text(kernel_code),
        })
    return recovered
