"""Post-flash DSE 2.0: PE×SIMD grid sweep (no locked 16×4 recipe).

Each trial injects that point's PE_NUM/SIMD from ``dse_v2_grid.json``, keeps
the Crow nest + mm_pe() shape, drops the per-bench PE table, and runs LLM →
csim → csynth. Winner = lowest latency among legal (csim pass, csynth ok,
DSP >= min_dsp).

v1 DSE (``post_flash_dse.py``) is unchanged. Enable with ``C2HLS_DSE_V2=1``.
When v2 is on, v1 DSE is skipped and stream is forced off by the chain hook.

``C2HLS_DSE_V3=1`` keeps this sweep and adds v3 prompt rules plus
``post_flash_dse_v3_skill_entries.json``. For float trials it also replaces
the constant architecture floor with ``floor(5 × PE × SIMD × 0.9)``.
``C2HLS_DSE_V3_HARNESS=1`` implies v3 and adds the csynth/harness checks.
With both unset, each trial keeps the grid ``min_dsp`` (currently 300) and
prompts and winner selection stay on the v2 path. The grid membership band
(expected DSP below ``min_dsp`` or above ``max_dsp``) is unchanged either way.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from c2hls_paths import DSE_V2_GRID_JSON, SKILLS_PACKAGE_DIR
from flash_flow_artifacts import sha256_text
from post_flash_dataflow import extract_kernel_block, sanitize_kernel_source
from post_flash_dse import (
    DseOutcome,
    call_dse_llm,
    dse_max_tokens,
    format_dse_initial_user,
    load_dse_skills,
    post_flash_no_skills,
    promote_dse_as_selected,
    repair_round_limit,
    resolve_baseline_kernel,
    resolve_dse_skills_path,
)
from post_flash_dse_v3 import (
    V3_PROMPT_RULES,
    append_v3_prompt_rules,
    assess_k_coverage,
    classify_k_coverage,
    dse_v3_enabled,
    dse_v3_harness_enabled,
    float_pair_min_dsp,
    harness_csynth_kwargs,
    harness_post_csynth_error,
    resolve_dse_v3_skills_path,
    tile_jk,
)
from post_flash_pragma_opt import resolve_source_kernel, summarize_synth_report
from skill_library import Skill, render_skill_set_for_prompt_full

STEP_TAG_V2 = "dse_v2"
DEFAULT_GRID_JSON = DSE_V2_GRID_JSON
DROPPED_SKILL_IDS = frozenset({"hls-dse-pick-pe-simd-for-kernel"})
V2_SKILL_IDS = (
    "hls-dse-gemm-multi-pe-latency-hiding",
    "hls-dse-gemm-simd-k-adder-tree",
    "hls-dse-partition-match-pe-simd",
    "avoid-dse-k-recurrence-on-single-c",
)

_LOG = logging.getLogger(__name__)
_LLM_SEM: Optional[threading.Semaphore] = None
_CSYNTH_SEM: Optional[threading.Semaphore] = None

_SYSTEM_V2 = """You are an expert Xilinx Vitis HLS 2023.2 engineer running a **DSE** step after flash.

Flash already produced a **working** load-compute-store kernel (csim + csynth pass).
Your job is **not** pragma-only tuning and **not** cloning an AutoSA PE/IO netlist.
Rewrite the **compute nest** into a compact multi-PE GEMM with latency hiding.

## Architecture (mandatory)
1. Keep the exact top-level `extern "C"` signature, parameter list, array ranks/shapes, and every existing `#pragma HLS INTERFACE` line.
2. Keep bulk on-chip staging (load A/B/C, compute, store C). No m_axi RMW inside compute.
3. Tile rows by **PE** from this trial's PE_NUM (must divide I / I_P). Use I_P/J_P/K_P when kernel.h defines them.
4. **Pipeline the independent j loop** (C columns) at II=1 — this is latency hiding.
5. Fully **UNROLL** the PE loop inside that pipelined j loop. Each PE has private `Crow[p][j]`.
6. Factor k by **SIMD** from this trial. Fully UNROLL SIMD into an adder tree: `partial = a0*b0+...` then `Crow[p][j] += partial`.
7. Match ARRAY_PARTITION to PE/SIMD. Do not complete-partition the full `C[I][J]`.
8. Index B using the existing layout (`B[J][K]` → `local_B[j][k]`; `B[K][J]` → `local_B[k][j]`).

## Forbidden
- Changing top-level ports, bundles, or function name.
- Pipelining a k-loop that updates one `C[i][j]` / `crow[j]` (flash leftover; DSP stays ~3–8, II=4).
- Emitting AutoSA `kernel0`, PE/IO interconnect, or `hls::stream` of PE structs. Do not emit AutoSA netlists.
- `malloc`, system calls, or non-synthesizable constructs.
- Ignoring this trial's PE_NUM / SIMD (do not substitute another pair).

## Success check before you submit
- Pipelined compute loop is `compute_j` (independent columns), not `compute_k`.
- PE UNROLL + SIMD UNROLL are present and match this trial.
- Expected DSP uses the float formula PE×SIMD×5 (or the integer formulas in the trial block). Single-digit DSP means you still have the flash nest — revise.

## Output
Start with ```kernel immediately. Close the fence. Do not spend the token budget on analysis.
```kernel
... full kernel source ...
```
"""

_REPAIR_USER_V2 = """Repair the **DSE** kernel after a validation or architecture failure.

Keep the exact top-level signature and `#pragma HLS INTERFACE` pragmas.
Keep **this trial's** PE_NUM×SIMD; pipeline independent `j`; do not fall back to flash `crow[j] +=` over k.
Start the reply with ```kernel and close the fence. Do not spend the token budget on analysis.

{trial_block}

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


def dse_v2_enabled() -> bool:
    return _env_flag("C2HLS_DSE_V2") is True


def chain_after_flash_v2() -> bool:
    chained = _env_flag("C2HLS_DSE_V2_CHAIN_FLASH")
    if chained is not None:
        return chained
    return dse_v2_enabled()


def resolve_dse_v2_grid_path() -> Path:
    raw = os.getenv("C2HLS_DSE_V2_GRID", "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_GRID_JSON


def load_dse_v2_grid(path: Optional[Path] = None) -> dict[str, Any]:
    grid_path = path or resolve_dse_v2_grid_path()
    data = json.loads(grid_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"dse_v2 grid root must be object: {grid_path}")
    return data


def only_simd_values() -> Optional[set[int]]:
    """Optional live-run filter: C2HLS_DSE_V2_ONLY_SIMD=1,2 keeps those SIMD widths."""
    raw = os.getenv("C2HLS_DSE_V2_ONLY_SIMD", "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        out.add(int(token))
    return out or None


def _pow2_range(lo: int, hi: int) -> list[int]:
    if lo < 1 or hi < lo:
        return []
    vals: list[int] = []
    v = 1
    while v < lo:
        v *= 2
    while v <= hi:
        vals.append(v)
        v *= 2
    return vals


def parse_ijk_from_header(header_code: str) -> tuple[int, int, int]:
    """Read active #define I/J/K (or I_P/J_P/K_P). Last uncommented define wins."""
    macros: dict[str, int] = {}
    for line in (header_code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        m = re.match(r"#\s*define\s+(I_P|J_P|K_P|I|J|K)\s+(\d+)\b", stripped)
        if m:
            macros[m.group(1)] = int(m.group(2))
    i = macros.get("I_P", macros.get("I"))
    j = macros.get("J_P", macros.get("J"))
    k = macros.get("K_P", macros.get("K"))
    if i is None or j is None or k is None:
        raise ValueError("kernel.h missing I/J/K (or I_P/J_P/K_P) defines")
    return i, j, k


def detect_data_kind(header_code: str, default: str = "float") -> str:
    text = header_code or ""
    if re.search(r"typedef\s+unsigned\s+short\s+data_t", text):
        return "uint16"
    if re.search(r"typedef\s+unsigned\s+int\s+data_t", text):
        return "uint32"
    if re.search(r"typedef\s+float\s+data_t", text):
        return "float"
    return default


def expected_dsp(pe: int, simd: int, data_kind: str, grid: dict[str, Any]) -> int:
    mul = {
        "float": int(grid.get("dsp_mul_float", 5)),
        "uint16": int(grid.get("dsp_mul_uint16", 1)),
        "uint32": int(grid.get("dsp_mul_uint32", 3)),
    }.get(data_kind, int(grid.get("dsp_mul_float", 5)))
    return pe * simd * mul


@dataclass(frozen=True)
class DseV2Trial:
    pe: int
    simd: int
    i: int
    j: int
    k: int
    data_kind: str
    expected_dsp: int
    min_dsp: int
    max_dsp: int
    pe_kj: int
    i_tiles: int
    pack_bits: int
    skip_reason: str = ""

    @property
    def trial_id(self) -> str:
        return f"pe{self.pe}_simd{self.simd}"

    @property
    def legal_for_prompt(self) -> bool:
        return not self.skip_reason


def expand_dse_v2_trials(
    *,
    i: int,
    j: int,
    k: int,
    data_kind: str = "float",
    grid: Optional[dict[str, Any]] = None,
) -> list[DseV2Trial]:
    """Return promptable trials (filters applied)."""
    return [
        t
        for t in expand_dse_v2_trials_detailed(
            i=i, j=j, k=k, data_kind=data_kind, grid=grid
        )
        if t.legal_for_prompt
    ]


def expand_dse_v2_trials_detailed(
    *,
    i: int,
    j: int,
    k: int,
    data_kind: str = "float",
    grid: Optional[dict[str, Any]] = None,
) -> list[DseV2Trial]:
    cfg = grid if grid is not None else load_dse_v2_grid()
    pe_cfg = cfg.get("pe") or {}
    simd_cfg = cfg.get("simd") or {}
    pe_vals = _pow2_range(int(pe_cfg.get("min", 8)), int(pe_cfg.get("max", 128)))
    simd_vals = _pow2_range(int(simd_cfg.get("min", 1)), int(simd_cfg.get("max", 32)))
    # Membership band: drop pairs whose *expected* DSP is outside [min_dsp, max_dsp].
    # This is not the per-trial architecture floor. V3 float trials override that
    # floor below; the band stays on the grid constants (300 / 9024 by default).
    band_min_dsp = int(cfg.get("min_dsp", 300))
    max_dsp = int(cfg.get("max_dsp", 9024))
    use_pair_floor = data_kind == "float" and dse_v3_enabled()
    dsp_mul_float = int(cfg.get("dsp_mul_float", 5))
    require_pe = bool(cfg.get("require_pe_divides_I", True))
    require_simd = bool(cfg.get("require_simd_divides_K", True))
    only_simd = only_simd_values()
    elem_bits = {"float": 32, "uint16": 16, "uint32": 32}.get(data_kind, 32)

    out: list[DseV2Trial] = []
    for pe in pe_vals:
        for simd in simd_vals:
            dsp = expected_dsp(pe, simd, data_kind, cfg)
            pe_kj = (k // simd) * j if simd else 0
            i_tiles = i // pe if pe else 0
            pack_bits = simd * elem_bits
            reason = ""
            if require_pe and (pe <= 0 or i % pe != 0):
                reason = f"PE={pe} does not divide I={i}"
            elif require_simd and (simd <= 0 or k % simd != 0):
                reason = f"SIMD={simd} does not divide K={k}"
            elif dsp < band_min_dsp:
                reason = f"expected_DSP={dsp} < min_dsp={band_min_dsp}"
            elif dsp > max_dsp:
                reason = f"expected_DSP={dsp} > max_dsp={max_dsp}"
            elif only_simd is not None and simd not in only_simd:
                reason = f"simd={simd} not in C2HLS_DSE_V2_ONLY_SIMD"
            if use_pair_floor:
                trial_min_dsp = float_pair_min_dsp(pe, simd, dsp_mul=dsp_mul_float)
            else:
                trial_min_dsp = band_min_dsp
            out.append(
                DseV2Trial(
                    pe=pe,
                    simd=simd,
                    i=i,
                    j=j,
                    k=k,
                    data_kind=data_kind,
                    expected_dsp=dsp,
                    min_dsp=trial_min_dsp,
                    max_dsp=max_dsp,
                    pe_kj=pe_kj,
                    i_tiles=i_tiles,
                    pack_bits=pack_bits,
                    skip_reason=reason,
                )
            )
    return out


def format_crow_nest_template(pe: int, simd: int) -> str:
    return f"""// Copy-paste compute nest for THIS trial. PE={pe} SIMD={simd}.
// Top ABI example: autosa_mm(A[I][K], B[J][K], C[I][J]) — B is J x K.
#define PE {pe}
#define SIMD {simd}
data_t Crow[PE][J];
#pragma HLS ARRAY_PARTITION variable=Crow complete dim=1
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=PE dim=1
#pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=SIMD dim=2
#pragma HLS ARRAY_PARTITION variable=local_B cyclic factor=SIMD dim=2
compute_i0: for (int i0 = 0; i0 < I; i0 += PE) {{
  init_pe: for (int p = 0; p < PE; ++p) {{
#pragma HLS UNROLL
    init_j: for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
      Crow[p][j] = local_C[i0 + p][j];
    }}
  }}
  compute_k0: for (int k0 = 0; k0 < K; k0 += SIMD) {{
    compute_j: for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
      pe_mac: for (int p = 0; p < PE; ++p) {{
#pragma HLS UNROLL
        data_t partial = 0;
        simd_k: for (int s = 0; s < SIMD; ++s) {{
#pragma HLS UNROLL
          partial += local_A[i0 + p][k0 + s] * local_B[j][k0 + s];
        }}
        Crow[p][j] += partial;
      }}
    }}
  }}
  wb_pe: for (int p = 0; p < PE; ++p) {{
#pragma HLS UNROLL
    wb_j: for (int j = 0; j < J; ++j) {{
#pragma HLS PIPELINE II=1
      local_C[i0 + p][j] = Crow[p][j];
    }}
  }}
}}
// Success check: pipelined loop is compute_j (not compute_k). DSP ~= PE*SIMD*5 for float.
"""


def format_trial_recipe_block(trial: DseV2Trial, *, step: str = "dse") -> str:
    pack_how = {
        "float": f"{trial.simd} floats via bit_cast/union of uint32 (NOT for integer data_t)",
        "uint16": f"{trial.simd} unsigned short via raw bits (no float union)",
        "uint32": f"{trial.simd} unsigned int via raw bits (no float union)",
    }.get(trial.data_kind, f"{trial.simd} elements")
    lines = [
        f"## DSE 2.0 trial recipe for this call ({step})",
        "There is **no** locked PE×SIMD for this benchmark. The outer sweep chooses the point;",
        "you must implement **exactly** the PE_NUM and SIMD below.",
        "",
        "### Formulas (same on every trial)",
        "- PE_NUM is a power of two that divides I (or I_P).",
        "- SIMD is a power of two that divides K (or K_P).",
        "- float expected_DSP ≈ PE_NUM × SIMD × 5; uint16 ≈ PE×SIMD; uint32 ≈ PE×SIMD×3.",
        f"- reject csynth if DSP < {trial.min_dsp}.",
        "- i_tiles = I / PE_NUM; pe_kj = (K / SIMD) × J; pack_bits = SIMD × elem_bits.",
        "- instantiate PE_NUM explicit mm_pe() calls; DATAFLOW cannot call mm_pe in a C for-loop.",
        "",
        "### This trial (mandatory)",
        f"- PE_NUM={trial.pe}",
        f"- SIMD={trial.simd}",
        f"- data_t kind: {trial.data_kind}",
        f"- pack SIMD as ap_uint<{trial.pack_bits}> ({pack_how})",
        f"- pe_kj trip = (K/SIMD)*J = {trial.pe_kj}",
        f"- i_tiles = I/PE_NUM = {trial.i_tiles}",
        f"- expected DSP ≈ {trial.expected_dsp} (reject if DSP < {trial.min_dsp})",
        f"- instantiate {trial.pe} explicit mm_pe() calls; DATAFLOW cannot call mm_pe in a C for-loop",
        "- keep the exact extern C top name and parameter list",
        "- if INTERFACE uses one bundle=gmem for A/B/C, split to gmem0/gmem1/gmem2 (do not change ports)",
        "",
        "### Compute nest template (this trial's PE/SIMD)",
        "```cpp",
        format_crow_nest_template(trial.pe, trial.simd).rstrip(),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _parameterize_skill_template(skill: Skill, pe: int, simd: int) -> Skill:
    """Rewrite hardcoded PE/SIMD in skill text for this trial; leave structure intact."""
    template = skill.template or ""
    if skill.id == "hls-dse-gemm-multi-pe-latency-hiding":
        template = format_crow_nest_template(pe, simd)
    strategy = skill.strategy
    strategy = re.sub(
        r"prefer 16 if I%16==0 else 8/4",
        "use this trial's PE_NUM (must divide I)",
        strategy,
    )
    strategy = re.sub(
        r"\(prefer 4\)",
        f"(this trial SIMD={simd})",
        strategy,
    )
    steps = list(skill.required_steps or [])
    new_steps = []
    for step in steps:
        if "choose PE/SIMD from the mandatory recipe" in step:
            new_steps.append(
                f"use this trial's PE_NUM={pe} and SIMD={simd} (must divide I and K)"
            )
        else:
            new_steps.append(step)
    return Skill(
        id=skill.id,
        pattern=skill.pattern,
        strategy=strategy,
        template=template,
        confidence=skill.confidence,
        kind=skill.kind,
        bottleneck_kinds=list(skill.bottleneck_kinds or []),
        applicable_versions=list(skill.applicable_versions or []),
        applicable_fpgas=list(skill.applicable_fpgas or []),
        tags=list(skill.tags or []),
        guards=list(skill.guards or []),
        required_steps=new_steps,
        occurrences=skill.occurrences,
        sec_pass=skill.sec_pass,
        mean_advantage=skill.mean_advantage,
        last_used_at=skill.last_used_at,
        origin=skill.origin,
    )


def _active_dse_skills_path(path: Optional[Path] = None) -> Path:
    """V2 skill file unless v3 is on and the caller did not pass a path."""
    if path is not None:
        return path
    if dse_v3_enabled():
        return resolve_dse_v3_skills_path()
    return resolve_dse_skills_path()


def load_dse_v2_skills(path: Optional[Path] = None) -> list[Skill]:
    skills = load_dse_skills(_active_dse_skills_path(path))
    return [sk for sk in skills if sk.id not in DROPPED_SKILL_IDS]


def build_dse_v2_skills_prompt_block(
    trial: DseV2Trial,
    path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    if post_flash_no_skills():
        return "", {"skills_path": "", "skill_count": 0, "skill_ids": []}
    skill_path = _active_dse_skills_path(path)
    skills = [
        _parameterize_skill_template(sk, trial.pe, trial.simd)
        for sk in load_dse_v2_skills(skill_path)
    ]
    block = render_skill_set_for_prompt_full(skills)
    meta: dict[str, Any] = {
        "skills_path": str(skill_path),
        "skill_count": len(skills),
        "skill_ids": [sk.id for sk in skills],
        "dropped_skill_ids": sorted(DROPPED_SKILL_IDS),
    }
    return block, meta


def dse_v2_extra_user_text() -> str:
    """Extra user paragraph. Empty unless C2HLS_DSE_V2_EXTRA_USER is set."""
    return os.getenv("C2HLS_DSE_V2_EXTRA_USER", "").strip()


def append_dse_v2_extra_user(text: str) -> str:
    """Append C2HLS_DSE_V2_EXTRA_USER when set. Unset leaves the prompt unchanged."""
    extra = dse_v2_extra_user_text()
    if not extra:
        return text
    return text.rstrip() + "\n\n## Additional goals\n" + extra + "\n"


def format_dse_v2_initial_user(
    *,
    trial: DseV2Trial,
    skills_block: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
    synth_summary: str,
    bench: str,
    source_role: str = "flash_final",
) -> str:
    trial_block = format_trial_recipe_block(trial, step="dse")
    abi = (
        f"- Benchmark name: `{bench}`.\n"
        f"- Include `{header_name}` exactly once and reuse its declarations.\n"
        "- Keep the exact testbench-visible top signature from the header / seed.\n"
    )
    benchmark_context = trial_block + abi
    return append_dse_v2_extra_user(
        append_v3_prompt_rules(
            format_dse_initial_user(
                skills_block=skills_block,
                benchmark_context=benchmark_context,
                header_name=header_name,
                header_code=header_code,
                kernel_code=kernel_code,
                synth_summary=synth_summary,
                source_role=source_role,
            )
        )
    )


def dse_v2_system_prompt() -> str:
    """V2 system text. V3 appends load/store rules; unset returns ``_SYSTEM_V2``."""
    if not dse_v3_enabled():
        return _SYSTEM_V2
    return _SYSTEM_V2.rstrip() + "\n\n" + V3_PROMPT_RULES


def architecture_ok_v2(report: Optional[dict[str, Any]], trial: DseV2Trial) -> bool:
    if not report:
        return False
    try:
        dsp = int(report.get("dsp") or 0)
    except (TypeError, ValueError):
        return False
    return dsp >= trial.min_dsp


def trial_dir(cell_dir: Path, bench: str, trial: DseV2Trial) -> Path:
    return cell_dir / STEP_TAG_V2 / trial.trial_id


def select_winner(results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Lowest latency among legal; tie-break lower DSP, then LUT."""
    legal = [r for r in results if r.get("legal") is True]
    if not legal:
        return None

    def key(r: dict[str, Any]) -> tuple:
        lat = r.get("latency_cycles")
        dsp = r.get("dsp")
        lut = r.get("lut")
        try:
            lat_f = float(lat) if lat is not None else float("inf")
        except (TypeError, ValueError):
            lat_f = float("inf")
        try:
            dsp_f = float(dsp) if dsp is not None else float("inf")
        except (TypeError, ValueError):
            dsp_f = float("inf")
        try:
            lut_f = float(lut) if lut is not None else float("inf")
        except (TypeError, ValueError):
            lut_f = float("inf")
        return (lat_f, dsp_f, lut_f)

    return sorted(legal, key=key)[0]


def select_harness_winner(
    results: list[dict[str, Any]],
    *,
    seed_coverage: Optional[str] = None,
) -> dict[str, Any]:
    """Best latency inside each K-coverage class.

    Classes are not ranked against each other by latency. A slower full-K
    trial cannot beat a faster partial-K trial, and a partial-K winner is
    never marked ``full_gemm``. When ``seed_coverage`` matches a class, that
    class is promoted; otherwise a mixed board promotes ``partial_k``.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("legal") is not True:
            continue
        tag = str(row.get("coverage") or "unknown")
        groups.setdefault(tag, []).append(row)
    by_coverage: dict[str, dict[str, Any]] = {}
    for tag, rows in groups.items():
        best = select_winner(rows)
        if best is None:
            continue
        stamped = dict(best)
        stamped["coverage"] = tag
        stamped["full_gemm"] = tag == "full_k"
        by_coverage[tag] = stamped
    if not by_coverage:
        return {"by_coverage": {}, "winner": None}
    if seed_coverage and seed_coverage in by_coverage:
        chosen = seed_coverage
    elif len(by_coverage) == 1:
        chosen = next(iter(by_coverage))
    elif "partial_k" in by_coverage:
        chosen = "partial_k"
    else:
        chosen = next(iter(by_coverage))
    return {"by_coverage": by_coverage, "winner": dict(by_coverage[chosen])}


def _init_worker_semaphores(grid: dict[str, Any]) -> None:
    global _LLM_SEM, _CSYNTH_SEM
    workers = grid.get("workers") or {}
    llm_default = workers.get("llm", 2)
    # Hosted DeepSeek proxies accept one in-flight completion. A second
    # parallel post gets HTTP 429 and that trial never runs.
    if (os.getenv("OPENAI_BASE_URL") or "").strip() and "C2HLS_DSE_V2_LLM_WORKERS" not in os.environ:
        llm_default = 1
    llm_n = max(1, int(os.getenv("C2HLS_DSE_V2_LLM_WORKERS", llm_default)))
    csynth_n = max(
        1, int(os.getenv("C2HLS_DSE_V2_CSYNTH_WORKERS", workers.get("csynth", 4)))
    )
    _LLM_SEM = threading.Semaphore(llm_n)
    _CSYNTH_SEM = threading.Semaphore(csynth_n)


def run_dse_v2_trial(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    trial: DseV2Trial,
    source_kernel: str,
    header_code: str,
    header_name: str,
    top_function: str,
    testbench_code: str,
    extra_files: list,
    part: str,
    clock_ns: float,
    prior_report: Optional[dict[str, Any]],
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> dict[str, Any]:
    from c2hls import _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag

    tdir = trial_dir(cell_dir, bench, trial)
    tdir.mkdir(parents=True, exist_ok=True)
    result_path = tdir / f"{bench}_result.json"
    if skip_existing and result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                isinstance(existing, dict)
                and existing.get("success") is True
                and existing.get("legal") is True
            ):
                return existing
        except json.JSONDecodeError:
            pass

    skills_block, skills_meta = build_dse_v2_skills_prompt_block(trial)
    synth_summary = summarize_synth_report(prior_report)
    token_floor = dse_max_tokens()
    current = getattr(orchestrator, "max_completion_tokens", 0) or 0
    if current < token_floor:
        orchestrator.max_completion_tokens = token_floor

    system = dse_v2_system_prompt()
    user = format_dse_v2_initial_user(
        trial=trial,
        skills_block=skills_block,
        header_name=header_name,
        header_code=header_code[:12000],
        kernel_code=source_kernel[:120000],
        synth_summary=synth_summary,
        bench=bench,
        source_role=source_role,
    )
    (tdir / f"{bench}_prompt_user.txt").write_text(user, encoding="utf-8")
    (tdir / f"{bench}_prompt_system.txt").write_text(system, encoding="utf-8")
    (tdir / f"{bench}_trial.json").write_text(
        json.dumps(asdict(trial), indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, str]] = []
    attempts: list[dict[str, Any]] = []
    kernel_code = ""
    success = False
    legal = False
    last_error = ""
    synth_report: dict[str, Any] = {}
    csim_summary: Optional[dict[str, Any]] = None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    assert _LLM_SEM is not None
    with _LLM_SEM:
        reply = call_dse_llm(orchestrator, messages, purpose=f"dse_v2_{trial.trial_id}")
    history.extend(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
        ]
    )
    kernel_code = extract_kernel_block(reply)

    tag_base = f"{STEP_TAG_V2}_{trial.trial_id}"
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
                assert _CSYNTH_SEM is not None
                with _CSYNTH_SEM:
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
                        log_prefix=f"[{STEP_TAG_V2}:{trial.trial_id}]",
                        temp_tag=tag,
                        **harness_csynth_kwargs(),
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
                        if architecture_ok_v2(synth_report, trial):
                            rows, cols = tile_jk(kernel_code, j=trial.j, k=trial.k)
                            harness_err = harness_post_csynth_error(
                                kernel_code=kernel_code,
                                report=synth_report,
                                tile_rows=rows,
                                tile_cols=cols,
                            )
                            if harness_err:
                                attempt_error = harness_err
                                stage = "harness_scalar_load"
                            else:
                                success = True
                                legal = True
                        else:
                            dsp = synth_report.get("dsp")
                            attempt_error = (
                                f"architecture miss: DSP={dsp} "
                                f"(need PE×SIMD nest, DSP>={trial.min_dsp}; "
                                "do not keep flash crow[j]+= over k)"
                            )
                            stage = "architecture"
                else:
                    attempt_error = synth.get("error") or "csynth failed"

        attempts.append(
            {
                "attempt": attempt,
                "stage": stage,
                "success": success,
                "error": attempt_error[:4000],
                "dsp": (synth_report or {}).get("dsp") if synth_report else None,
                "latency_cycles": (synth_report or {}).get("latency_cycles")
                if synth_report
                else None,
            }
        )
        last_error = attempt_error
        if success:
            break
        if attempt >= repair_round_limit() - 1:
            break

        repair_kernel = kernel_code if kernel_code.strip() else source_kernel
        repair_user = append_dse_v2_extra_user(
            append_v3_prompt_rules(
                _REPAIR_USER_V2.format(
                    trial_block=format_trial_recipe_block(trial),
                    stage=stage,
                    error=attempt_error[:6000],
                    header_name=header_name,
                    header_code=header_code[:12000],
                    kernel_code=repair_kernel[:120000],
                )
            )
        )
        repair_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": repair_user},
        ]
        with _LLM_SEM:
            repair_reply = call_dse_llm(
                orchestrator, repair_messages, purpose=f"dse_v2_repair_{trial.trial_id}"
            )
        history.append({"role": "user", "content": repair_user})
        history.append({"role": "assistant", "content": repair_reply})
        extracted = extract_kernel_block(repair_reply)
        if extracted:
            kernel_code = extracted

    kernel_code = sanitize_kernel_source(kernel_code)
    dsp = (synth_report or {}).get("dsp")
    latency_cycles = (synth_report or {}).get("latency_cycles")
    lut = (synth_report or {}).get("lut")
    payload: dict[str, Any] = {
        "schema": "post_flash_dse_v2_trial_v1",
        "benchmark": bench,
        "step": STEP_TAG_V2,
        "trial_id": trial.trial_id,
        "pe": trial.pe,
        "simd": trial.simd,
        "expected_dsp": trial.expected_dsp,
        "min_dsp": trial.min_dsp,
        "success": success,
        "legal": legal,
        "error": last_error if not success else "",
        "latency_cycles": latency_cycles,
        "dsp": dsp,
        "lut": lut,
        "bram": (synth_report or {}).get("bram"),
        "ff": (synth_report or {}).get("ff"),
        "attempts": attempts,
        "synth_report": synth_report,
        "csim": csim_summary,
        "skills": skills_meta,
        "kernel_sha256": sha256_text(kernel_code) if kernel_code else "",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if dse_v3_harness_enabled():
        cov = assess_k_coverage(kernel_code, i=trial.i, j=trial.j, k=trial.k)
        placed: dict[str, Any] = {}
        for key, val in payload.items():
            placed[key] = val
            if key == "latency_cycles":
                placed["mac_count"] = cov["mac_count"]
                placed["coverage"] = cov["coverage"]
                placed["full_gemm"] = cov["full_gemm"]
        payload = placed
    if kernel_code:
        (tdir / f"{bench}.cpp").write_text(kernel_code, encoding="utf-8")
    (tdir / f"{bench}_history.json").write_text(
        json.dumps(
            {"model": getattr(orchestrator, "model", ""), "messages": history}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    (tdir / f"{bench}_report.json").write_text(
        json.dumps(synth_report or {}, indent=2, default=str) + "\n", encoding="utf-8"
    )
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def run_dse_v2_for_cell(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> DseOutcome:
    from c2hls import _load_benchmark_inputs

    if source_role not in {"flash_final", "baseline"}:
        return DseOutcome(
            bench, source_role, False, str(cell_dir), "dse_v2 is flash-final or baseline only"
        )

    prior_report: Optional[dict[str, Any]] = None
    if source_role == "baseline":
        kernel_path = resolve_baseline_kernel(cell_dir, bench, bench_dir)
        kernel_role = "baseline"
    else:
        # Prefer flash seed if present (selected may already be stream/dse).
        seed = cell_dir / f"{bench}_flash_seed.cpp"
        final = cell_dir / f"{bench}_final.cpp"
        flash_opt = cell_dir / f"{bench}_flash_opt.cpp"
        if seed.is_file() and seed.read_text(encoding="utf-8").strip():
            kernel_path = seed
            kernel_role = "flash_seed"
            seed_report = cell_dir / f"{bench}_flash_seed_report.json"
            if seed_report.is_file():
                try:
                    prior_report = json.loads(seed_report.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    prior_report = None
        elif final.is_file():
            kernel_path = final
            kernel_role = "final"
            for cand in (
                cell_dir / f"{bench}_flash_opt_report.json",
                cell_dir / f"{bench}_selected_report.json",
            ):
                if cand.is_file():
                    try:
                        prior_report = json.loads(cand.read_text(encoding="utf-8"))
                        break
                    except json.JSONDecodeError:
                        continue
        elif flash_opt.is_file():
            kernel_path = flash_opt
            kernel_role = "flash_opt"
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
            "no selected/final/flash_seed kernel cpp",
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
    testbench_code = inputs.get("testbench_code", "")
    extra_files = inputs.get("extra_files", [])
    part = meta.get("part", orchestrator.part)
    clock_ns = meta.get("clock_ns", orchestrator.clock_ns)

    grid = load_dse_v2_grid()
    data_kind = detect_data_kind(header_code, str(grid.get("data_kind_default", "float")))
    i, j, k = parse_ijk_from_header(header_code)
    detailed = expand_dse_v2_trials_detailed(
        i=i, j=j, k=k, data_kind=data_kind, grid=grid
    )
    trials = [t for t in detailed if t.legal_for_prompt]
    (cell_dir / STEP_TAG_V2).mkdir(parents=True, exist_ok=True)
    (cell_dir / STEP_TAG_V2 / "grid_expansion.json").write_text(
        json.dumps(
            {
                "i": i,
                "j": j,
                "k": k,
                "data_kind": data_kind,
                "promptable": [asdict(t) for t in trials],
                "skipped": [asdict(t) for t in detailed if not t.legal_for_prompt],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not trials:
        return DseOutcome(
            bench, source_role, False, str(cell_dir), "dse_v2 grid empty after filters"
        )

    _init_worker_semaphores(grid)
    llm_pool = int((grid.get("workers") or {}).get("llm", 2))
    if (os.getenv("OPENAI_BASE_URL") or "").strip() and "C2HLS_DSE_V2_LLM_WORKERS" not in os.environ:
        llm_pool = 1
    max_workers = max(1, min(len(trials), llm_pool * 2))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                run_dse_v2_trial,
                bench=bench,
                bench_dir=bench_dir,
                cell_dir=cell_dir,
                orchestrator=orchestrator,
                trial=trial,
                source_kernel=source_kernel,
                header_code=header_code,
                header_name=header_name,
                top_function=top_function,
                testbench_code=testbench_code,
                extra_files=extra_files,
                part=part,
                clock_ns=clock_ns,
                prior_report=prior_report if isinstance(prior_report, dict) else None,
                source_role=source_role,
                skip_existing=skip_existing,
            ): trial
            for trial in trials
        }
        for fut in as_completed(futs):
            trial = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                _LOG.exception("[dse_v2] trial %s failed: %s", trial.trial_id, exc)
                results.append(
                    {
                        "schema": "post_flash_dse_v2_trial_v1",
                        "benchmark": bench,
                        "trial_id": trial.trial_id,
                        "pe": trial.pe,
                        "simd": trial.simd,
                        "success": False,
                        "legal": False,
                        "error": str(exc)[:4000],
                        "latency_cycles": None,
                        "dsp": None,
                        "lut": None,
                    }
                )

    if dse_v3_harness_enabled():
        seed_coverage = classify_k_coverage(source_kernel)
        coverage_board = select_harness_winner(results, seed_coverage=seed_coverage)
        winner = coverage_board["winner"]
    else:
        seed_coverage = None
        coverage_board = None
        winner = select_winner(results)
    leaderboard = {
        "schema": "post_flash_dse_v2_leaderboard_v1",
        "benchmark": bench,
        "source_role": source_role,
        "source_kernel": str(kernel_path),
        "source_kernel_role": kernel_role,
        "grid_path": str(resolve_dse_v2_grid_path()),
        "n_trials": len(trials),
        "n_legal": sum(1 for r in results if r.get("legal")),
        "winner": winner,
        "trials": sorted(
            results,
            key=lambda r: (
                0 if r.get("legal") else 1,
                r.get("latency_cycles")
                if r.get("latency_cycles") is not None
                else 1e18,
            ),
        ),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if coverage_board is not None:
        leaderboard["seed_coverage"] = seed_coverage
        leaderboard["winners_by_coverage"] = coverage_board["by_coverage"]
        leaderboard["winner_coverage"] = (winner or {}).get("coverage")
        leaderboard["full_gemm"] = (winner or {}).get("full_gemm")
    board_path = cell_dir / f"{bench}_dse_v2_leaderboard.json"
    board_path.write_text(
        json.dumps(leaderboard, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if winner is None:
        return DseOutcome(
            bench,
            source_role,
            False,
            str(cell_dir),
            "dse_v2: no legal trial (csim+csynth+DSP)",
            result=leaderboard,
        )

    wdir = cell_dir / STEP_TAG_V2 / winner["trial_id"]
    w_cpp = wdir / f"{bench}.cpp"
    w_hist = wdir / f"{bench}_history.json"
    w_report = wdir / f"{bench}_report.json"
    code = w_cpp.read_text(encoding="utf-8") if w_cpp.is_file() else ""
    dse_cpp = cell_dir / f"{bench}_dse.cpp"
    dse_hist = cell_dir / f"{bench}_dse_history.json"
    dse_report = cell_dir / f"{bench}_dse_report.json"
    dse_result = cell_dir / f"{bench}_dse_result.json"
    if code:
        dse_cpp.write_text(code, encoding="utf-8")
    if w_hist.is_file():
        shutil.copy2(w_hist, dse_hist)
    if w_report.is_file():
        shutil.copy2(w_report, dse_report)

    result_payload = {
        "schema": "post_flash_dse_v2_v1",
        "benchmark": bench,
        "source_role": source_role,
        "step": STEP_TAG_V2,
        "success": True,
        "legal": True,
        "winner_trial_id": winner["trial_id"],
        "pe": winner.get("pe"),
        "simd": winner.get("simd"),
        "latency_cycles": winner.get("latency_cycles"),
        "dsp": winner.get("dsp"),
        "lut": winner.get("lut"),
        "leaderboard": board_path.name,
        "source_kernel": str(kernel_path.name),
        "source_kernel_role": kernel_role,
        "kernel_sha256": sha256_text(code) if code else "",
        "promoted": False,
    }
    if winner.get("coverage"):
        result_payload["coverage"] = winner.get("coverage")
        result_payload["mac_count"] = winner.get("mac_count")
        result_payload["full_gemm"] = winner.get("full_gemm") is True
    if code:
        promote_dse_as_selected(
            cell_dir=cell_dir,
            bench=bench,
            code=code,
            report=json.loads(w_report.read_text(encoding="utf-8"))
            if w_report.is_file()
            else {},
            result_payload={
                "success": True,
                "latency_cycles": winner.get("latency_cycles"),
                "dsp": winner.get("dsp"),
            },
        )
        result_payload["promoted"] = True
        manifest_path = cell_dir / f"{bench}_flow_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    manifest["selected_from"] = "dse_v2"
                    manifest["dse_v2_winner"] = winner["trial_id"]
                    files = dict(manifest.get("files") or {})
                    files["dse_v2_leaderboard"] = board_path.name
                    manifest["files"] = files
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, default=str) + "\n",
                        encoding="utf-8",
                    )
            except json.JSONDecodeError:
                pass

    dse_result.write_text(
        json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (cell_dir / f"{bench}_post_flash_dse_v2.json").write_text(
        json.dumps(
            {
                "schema": "post_flash_dse_v2_manifest_v1",
                "benchmark": bench,
                "source_role": source_role,
                "source_kernel": str(kernel_path),
                "success": True,
                "winner_trial_id": winner["trial_id"],
                "leaderboard": board_path.name,
                "result": dse_result.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = DseOutcome(bench, source_role, True, str(cell_dir), result=result_payload)
    _maybe_run_focus_group_after_winner(
        bench=bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        orchestrator=orchestrator,
        outcome=outcome,
    )
    return outcome


def _maybe_run_focus_group_after_winner(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    outcome: DseOutcome,
) -> None:
    """Run the focus group on the DSE v2 winner only. No-op unless enabled."""
    if not outcome or not outcome.success:
        return
    winner_id = (outcome.result or {}).get("winner_trial_id")
    if not winner_id:
        return
    try:
        from post_flash_focus_group import maybe_chain_focus_group
    except ImportError:
        return
    maybe_chain_focus_group(
        bench=bench,
        bench_dir=bench_dir,
        kernel_dir=cell_dir / STEP_TAG_V2 / str(winner_id),
        orchestrator=orchestrator,
    )


def maybe_chain_dse_v2(
    *,
    bench: str,
    bench_dir: Path,
    cell_dir: Path,
    orchestrator: Any,
    source_role: str = "flash_final",
    skip_existing: bool = True,
) -> Optional[DseOutcome]:
    if _env_flag("C2HLS_DSE_V4") is True:
        return None
    if source_role != "flash_final" or not chain_after_flash_v2() or not dse_v2_enabled():
        return None
    os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
    os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"
    try:
        outcome = run_dse_v2_for_cell(
            bench=bench,
            bench_dir=bench_dir,
            cell_dir=cell_dir,
            orchestrator=orchestrator,
            source_role=source_role,
            skip_existing=skip_existing,
        )
        if outcome.success:
            _LOG.info(
                "[dse_v2] %s passed winner=%s",
                bench,
                (outcome.result or {}).get("winner_trial_id"),
            )
        else:
            _LOG.warning("[dse_v2] %s failed: %s", bench, outcome.error[:200])
        return outcome
    except Exception as exc:
        _LOG.exception("[dse_v2] %s error: %s", bench, exc)
        try:
            (cell_dir / f"{bench}_dse_v2_chain_error.txt").write_text(
                str(exc) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        return None


def configure_post_flash_dse_v2_env() -> None:
    os.environ.setdefault("C2HLS_RUN_COSIM", "0")
    os.environ.setdefault("C2HLS_COSIM_REQUIRED", "0")
    os.environ.setdefault("C2HLS_REFERENCE_COSIM", "0")
    if dse_v2_enabled():
        os.environ["C2HLS_POST_FLASH_STREAM"] = "0"
        os.environ["C2HLS_STREAM_CHAIN_FLASH"] = "0"


__all__ = [
    "DEFAULT_GRID_JSON",
    "DROPPED_SKILL_IDS",
    "DseV2Trial",
    "STEP_TAG_V2",
    "V2_SKILL_IDS",
    "append_dse_v2_extra_user",
    "architecture_ok_v2",
    "build_dse_v2_skills_prompt_block",
    "dse_v2_extra_user_text",
    "chain_after_flash_v2",
    "configure_post_flash_dse_v2_env",
    "detect_data_kind",
    "dse_v2_enabled",
    "dse_v2_system_prompt",
    "dse_v3_enabled",
    "dse_v3_harness_enabled",
    "expand_dse_v2_trials",
    "expand_dse_v2_trials_detailed",
    "expected_dsp",
    "format_crow_nest_template",
    "format_dse_v2_initial_user",
    "format_trial_recipe_block",
    "load_dse_v2_grid",
    "load_dse_v2_skills",
    "maybe_chain_dse_v2",
    "_maybe_run_focus_group_after_winner",
    "parse_ijk_from_header",
    "resolve_dse_v2_grid_path",
    "run_dse_v2_for_cell",
    "run_dse_v2_trial",
    "select_harness_winner",
    "select_winner",
]
