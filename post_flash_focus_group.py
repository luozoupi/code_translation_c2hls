"""Post-legal focus group: load / compute / store latency, not a repair loop.

Repair still legalizes a candidate (csim + csynth). This pass is a separate
two-round analysis → specialist plans → chair kernel loop with a coverage gate
so a shorter nest that drops most of K cannot win.
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

from post_flash_dataflow import extract_kernel_block
from post_flash_dse_v2 import parse_ijk_from_header
from post_flash_latency_opt import under_device_budget

_LOG = logging.getLogger(__name__)

STEP_TAG = "focus_group"
TRANSCRIPT_SCHEMA = "post_flash_focus_group_transcript_v1"
DEFAULT_ROUNDS = 2
DEFAULT_REPAIR_ROUNDS = 3
COVERAGE_RATIO_MIN = 0.5

SPECIALIST_IDS: tuple[str, ...] = (
    "load",
    "compute",
    "store",
    "ii_pipeline",
    "unroll",
    "memory_partition",
)

_SPECIALIST_FOCUS = {
    "load": "load loops and how A/B (and C init) enter on-chip buffers",
    "compute": "compute loops, PE tiling, and the SIMD-k adder tree",
    "store": "store loops and write-back of every C[i][j] element",
    "ii_pipeline": "initiation interval and PIPELINE on the independent loops",
    "unroll": "effective UNROLL of PE and SIMD (and matching trip steps)",
    "memory_partition": "ARRAY_PARTITION, ports, and on-chip tile shapes",
}


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def focus_group_enabled() -> bool:
    return _truthy("C2HLS_FOCUS_GROUP")


def focus_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_FOCUS_GROUP_ROUNDS", str(DEFAULT_ROUNDS))))
    except ValueError:
        return DEFAULT_ROUNDS


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_FOCUS_GROUP_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def parse_pe_simd_from_kernel(kernel_code: str) -> tuple[Optional[int], Optional[int]]:
    text = kernel_code or ""
    found: dict[str, int] = {}
    patterns = (
        r"#\s*define\s+(PE_NUM|PE|SIMD)\s+(\d+)\b",
        r"const\s+int\s+(PE_NUM|PE|SIMD)\s*=\s*(\d+)\s*;",
    )
    for pat in patterns:
        for m in re.finditer(pat, text):
            found[m.group(1)] = int(m.group(2))
    pe = found.get("PE_NUM", found.get("PE"))
    simd = found.get("SIMD")
    return pe, simd


def parse_tile_sizes(kernel_code: str) -> dict[str, int]:
    text = kernel_code or ""
    out: dict[str, int] = {}
    for pat in (
        r"#\s*define\s+(TI|TJ|TK|PE|SIMD|PE_NUM)\s+(\d+)\b",
        r"const\s+int\s+(TI|TJ|TK|PE|SIMD|PE_NUM)\s*=\s*(\d+)\s*;",
    ):
        for m in re.finditer(pat, text):
            out[m.group(1)] = int(m.group(2))
    if "PE" not in out and "PE_NUM" in out:
        out["PE"] = out["PE_NUM"]
    return out


def classify_scope_region(scope: dict[str, Any]) -> str:
    raw = " ".join(
        str(scope.get(key) or "")
        for key in ("scope_id", "name", "loop_name")
    ).lower()
    tokens = re.split(r"[^a-z0-9]+", raw)
    joined = " ".join(tokens)
    if any(t.startswith("load") or t in {"read"} for t in tokens) or " load" in f" {joined}":
        return "load"
    if any(t.startswith("store") or t.startswith("wb") or t in {"write"} for t in tokens):
        return "store"
    if any(
        t.startswith("compute") or t.startswith("mac") or t in {"pe_mac", "simd", "simd_k"}
        for t in tokens
    ):
        return "compute"
    return "other"


def arithmetic_floor(
    i: Optional[int],
    j: Optional[int],
    k: Optional[int],
    pe: Optional[int],
    simd: Optional[int],
) -> Optional[int]:
    if not all(isinstance(x, int) and x > 0 for x in (i, j, k, pe, simd)):
        return None
    return (i * j * k) // (pe * simd)  # type: ignore[operator]


def _scope_trip(scope: dict[str, Any]) -> int:
    for key in ("trip", "tripcount", "trip_count"):
        val = scope.get(key)
        if val is None:
            continue
        try:
            n = int(val)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return 1


def _scope_latency(scope: dict[str, Any]) -> int:
    for key in ("latency_cycles", "latency", "interval"):
        val = scope.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return 0


def _is_loop(scope: dict[str, Any]) -> bool:
    kind = str(scope.get("kind") or "loop").lower()
    return kind in {"loop", "pipeline"}


def _is_pipelined_ii1(scope: dict[str, Any]) -> bool:
    pipelined = scope.get("pipelined")
    if isinstance(pipelined, str):
        yes = pipelined.strip().lower() in {"yes", "true", "1", "pipeline"}
    else:
        yes = bool(pipelined)
    ii = scope.get("pipeline_ii")
    if ii is None:
        return yes
    try:
        return yes and int(ii) == 1
    except (TypeError, ValueError):
        return yes


def _index_scopes(scopes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        sid = scope.get("scope_id") or scope.get("name")
        if sid:
            out[str(sid)] = scope
    return out


def ancestor_trip_product(scope: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> int:
    total = _scope_trip(scope)
    parent_id = scope.get("parent")
    seen: set[str] = set()
    while parent_id:
        key = str(parent_id)
        if key in seen:
            break
        seen.add(key)
        parent = by_id.get(key)
        if not parent:
            break
        if _is_loop(parent):
            total *= _scope_trip(parent)
        parent_id = parent.get("parent")
    return total


def infer_store_vector_width(kernel_code: str, simd: Optional[int]) -> int:
    text = kernel_code or ""
    store_windows = re.findall(
        r"(store\w*[\s\S]{0,400}idx\s*\+=\s*(SIMD|PE|TI|TJ|\d+))",
        text,
        flags=re.IGNORECASE,
    )
    for _blob, step in store_windows:
        if step.upper() == "SIMD" and simd:
            return int(simd)
        if step.isdigit():
            return max(1, int(step))
    if re.search(r"store\w*[\s\S]{0,300}idx\s*\+=\s*SIMD", text, flags=re.IGNORECASE) and simd:
        return int(simd)
    return 1


def _innermost(scopes: list[dict[str, Any]], region: str) -> list[dict[str, Any]]:
    loops = [s for s in scopes if _is_loop(s) and classify_scope_region(s) == region]
    ids = {s.get("scope_id") for s in loops}
    out = []
    for scope in loops:
        sid = scope.get("scope_id")
        has_child = any(child.get("parent") == sid and child.get("scope_id") in ids for child in loops)
        if not has_child:
            out.append(scope)
    return out


def build_analysis_pack(
    *,
    header_code: str,
    kernel_code: str,
    report: Optional[dict[str, Any]],
) -> dict[str, Any]:
    i, j, k = parse_ijk_from_header(header_code)
    pe, simd = parse_pe_simd_from_kernel(kernel_code)
    tiles = parse_tile_sizes(kernel_code)
    report = report or {}
    scopes = list(((report.get("feedback") or {}).get("scopes")) or [])
    by_id = _index_scopes(scopes)
    floor = arithmetic_floor(i, j, k, pe, simd)
    top_lat = 0
    try:
        top_lat = int(report.get("latency_cycles") or 0)
    except (TypeError, ValueError):
        top_lat = 0

    regions: dict[str, dict[str, Any]] = {}
    for region in ("load", "compute", "store", "other"):
        members = [s for s in scopes if classify_scope_region(s) == region]
        lat = max((_scope_latency(s) for s in members), default=0)
        ii1 = any(_is_pipelined_ii1(s) for s in members if _is_loop(s))
        regions[region] = {
            "scope_ids": [s.get("scope_id") for s in members],
            "latency_cycles": lat,
            "ii1": ii1,
            "fraction_of_top": (lat / top_lat) if top_lat else 0.0,
        }

    compute_total = 0
    for scope in _innermost(scopes, "compute"):
        compute_total += ancestor_trip_product(scope, by_id)

    store_width = infer_store_vector_width(kernel_code, simd)
    store_trips = 0
    for scope in _innermost(scopes, "store"):
        store_trips += ancestor_trip_product(scope, by_id)
    store_elements = store_trips * max(1, store_width)

    sum_region = sum(regions[r]["latency_cycles"] for r in ("load", "compute", "store"))
    pack = {
        "i": i,
        "j": j,
        "k": k,
        "pe": pe,
        "simd": simd,
        "tiles": tiles,
        "arithmetic_floor": floor,
        "latency_cycles": top_lat,
        "regions": regions,
        "compute_trip_total": compute_total,
        "store_trip_total": store_trips,
        "store_element_total": store_elements,
        "store_vector_width": store_width,
        "sequential_tax": {
            "sum_region_latency": sum_region,
            "top_latency": top_lat,
            "gap": max(0, sum_region - top_lat) if top_lat else sum_region,
        },
        "report": report,
        "lut": report.get("lut"),
        "dsp": report.get("dsp"),
        "ff": report.get("ff"),
        "bram": report.get("bram"),
        "uram": report.get("uram"),
    }
    pack["coverage"] = coverage_gate(pack)
    return pack


@dataclass
class CoverageResult:
    ok: bool
    reason: str = ""


def coverage_gate(pack: dict[str, Any]) -> CoverageResult:
    i, j, k = pack.get("i"), pack.get("j"), pack.get("k")
    pe, simd = pack.get("pe"), pack.get("simd")
    floor = pack.get("arithmetic_floor")
    compute_total = int(pack.get("compute_trip_total") or 0)
    store_elements = int(pack.get("store_element_total") or 0)
    reasons: list[str] = []

    if isinstance(floor, int) and floor > 0 and pe and simd:
        if compute_total < floor * COVERAGE_RATIO_MIN:
            reasons.append(
                f"compute trips {compute_total} are far below the K-complete floor {floor} "
                f"(PE={pe} SIMD={simd}); the nest does not cover full I×J×K"
            )

    if isinstance(i, int) and isinstance(j, int) and i > 0 and j > 0:
        need = i * j
        if store_elements < need * COVERAGE_RATIO_MIN:
            reasons.append(
                f"store covers about {store_elements} C elements versus I×J={need}"
            )

    tiles = pack.get("tiles") or {}
    ti = tiles.get("TI")
    if isinstance(ti, int) and isinstance(pe, int) and ti > pe:
        reasons.append(
            f"row tile TI={ti} is larger than PE={pe}; store/compute of PE rows drops rows"
        )

    if reasons:
        return CoverageResult(ok=False, reason="; ".join(reasons))
    return CoverageResult(ok=True, reason="compute trips and store cover I, J, and K")


def render_analysis_pack(pack: dict[str, Any]) -> str:
    lines = [
        f"Problem size: I={pack.get('i')} J={pack.get('j')} K={pack.get('k')}",
        f"PE={pack.get('pe')} SIMD={pack.get('simd')} tiles={pack.get('tiles')}",
        f"Arithmetic floor I×J×K/(PE×SIMD) = {pack.get('arithmetic_floor')}",
        f"Measured top latency_cycles = {pack.get('latency_cycles')}",
        f"Compute trip total (innermost × ancestors) = {pack.get('compute_trip_total')}",
        f"Store element total = {pack.get('store_element_total')} "
        f"(trips={pack.get('store_trip_total')} × width={pack.get('store_vector_width')})",
    ]
    tax = pack.get("sequential_tax") or {}
    lines.append(
        f"Sequential tax: sum(load,compute,store)={tax.get('sum_region_latency')} "
        f"vs top={tax.get('top_latency')} gap={tax.get('gap')}"
    )
    lines.append("Regions:")
    for name in ("load", "compute", "store", "other"):
        region = (pack.get("regions") or {}).get(name) or {}
        lines.append(
            f"  - {name}: latency={region.get('latency_cycles')} "
            f"fraction={region.get('fraction_of_top')} ii1={region.get('ii1')} "
            f"scopes={region.get('scope_ids')}"
        )
    cov = pack.get("coverage")
    if isinstance(cov, CoverageResult):
        lines.append(f"Coverage gate: ok={cov.ok} reason={cov.reason}")
    elif isinstance(cov, dict):
        lines.append(f"Coverage gate: ok={cov.get('ok')} reason={cov.get('reason')}")
    return "\n".join(lines)


def should_accept(
    candidate: dict[str, Any],
    best: dict[str, Any],
    *,
    part: str,
) -> bool:
    gate = candidate.get("coverage")
    if isinstance(gate, CoverageResult):
        ok = gate.ok
    elif isinstance(gate, dict):
        ok = bool(gate.get("ok"))
    else:
        ok = coverage_gate(candidate).ok
    if not ok:
        return False
    report = candidate.get("report") or candidate
    if not under_device_budget(report, part, budget_pct=100.0):
        return False
    try:
        cand_lat = float(candidate.get("latency_cycles"))
        best_lat = float(best.get("latency_cycles"))
    except (TypeError, ValueError):
        return False
    return cand_lat < best_lat


_ANALYST_SYSTEM = """You are the theoretical analyst in a Vitis HLS focus group.

Given the deterministic analysis pack, header sizes, and kernel (reference only),
write a diagnosis. Do not output kernel source. Plan only.

Cover:
- which region (load, compute, store) owns the gap versus the arithmetic floor
- whether each region is II=1 and whether load/store sit on the critical path
- what must stay: full I, full J, and full K (do not shrink the iteration space)
- sequential tax: loads issued before compute with no overlap

Output structured text with sections: schedule, gap_owner, must_keep, next_actions.
"""

_SPECIALIST_SYSTEM = """You are a latency optimizer specialist ({member}) in a Vitis HLS focus group.

Focus only on {focus}. Do not output kernel source. Plan only.

Rules:
- Do not shrink I, J, or K. Do not drop rows, columns, or the K reduction.
- Touch only your area. Cite scope_ids from the analysis pack.
- Later members will read your plan; be concrete (pragma, loop, buffer).

Output: targets, actions, avoid, expected_cycles.
"""

_CHAIR_SYSTEM = """You are the chair of a Vitis HLS focus group and the only code editor.

Read the analyst diagnosis and every specialist plan. Drop conflicts. Emit one
legal kernel that keeps full I, J, and K and applies the surviving edits.

## Rules
- Preserve the exact top-level `extern "C"` signature and every `#pragma HLS INTERFACE` line.
- Keep PE and SIMD (or the trial pair) unless a plan proves a better pair that still covers I×J×K.
- Do not legalize by deleting work. The coverage gate rejects a K-slice or a partial store.
- Stay within the device budget.

## Output
Return one fenced block only:
```kernel
... full kernel source ...
```
"""

_REPAIR_USER = """Repair this kernel after a validation failure. This is a legality repair,
not a focus-group performance round.

Restore compile / csim / csynth. Do not change the numerical coverage of C:
every C[i][j] must still equal the full I×J×K product (or C + A×B if the seed
loaded C). Do not shrink I, J, or K to make latency look smaller.

## Failure
Stage: {stage}
```
{error}
```

## Chair plan snapshot
{plan_text}

## Current kernel
```cpp
{kernel_code}
```

Return a single ```kernel``` block.
"""


def _specialist_system(member: str) -> str:
    return _SPECIALIST_SYSTEM.format(
        member=member,
        focus=_SPECIALIST_FOCUS.get(member, member),
    )


def prompt_text_for_docs() -> dict[str, Any]:
    return {
        "analyst_system": _ANALYST_SYSTEM,
        "specialist_systems": {name: _specialist_system(name) for name in SPECIALIST_IDS},
        "chair_system": _CHAIR_SYSTEM,
        "repair_user": _REPAIR_USER,
        "analyst_user": (
            "## Analysis pack\n{analysis_pack}\n\n## Header\n```cpp\n{header_code}\n```\n"
        ),
        "chair_user": (
            "## Analysis pack\n{analysis_pack}\n\n## Analyst\n{analyst_text}\n\n"
            "## Specialist plans\n{plans_text}\n\n## Current kernel\n```cpp\n{kernel_code}\n```\n"
        ),
    }


def _transcript_brief(transcript: dict[str, Any]) -> str:
    rounds = transcript.get("rounds") or []
    if not rounds:
        return "(no prior focus round)"
    parts = []
    for row in rounds:
        parts.append(
            f"round {row.get('round')}: accepted={row.get('accepted')} "
            f"reason={row.get('reason')} latency={row.get('latency_cycles')} "
            f"coverage={((row.get('analysis') or {}).get('coverage') or {})}"
        )
        plans = row.get("plans") or {}
        for name in SPECIALIST_IDS:
            if name in plans:
                parts.append(f"  prior {name} plan: {str(plans[name])[:400]}")
    return "\n".join(parts)


def _call_llm(orchestrator: Any, messages: list[dict[str, str]]) -> str:
    return orchestrator._call_llm(messages) or ""


@dataclass
class FocusGroupOutcome:
    bench: str
    success: bool
    out_dir: str
    error: str = ""
    result: Optional[dict[str, Any]] = None


def run_focus_group(
    *,
    bench: str,
    kernel_code: str,
    header_code: str,
    header_name: str,
    report: dict[str, Any],
    orchestrator: Any,
    out_dir: Path,
    testbench_code: str = "",
    extra_files: Optional[list] = None,
    top_function: str = "autosa_mm",
    part: str = "xcu280-fsvh2892-2L-e",
    clock_ns: float = 3.33,
) -> FocusGroupOutcome:
    from c2hls import _run_synth_csim_cosim, compile_check_cpp
    from c2hls_temp import join_temp_tag
    from post_flash_dse import dse_max_tokens

    token_floor = dse_max_tokens()
    current = getattr(orchestrator, "max_completion_tokens", 0) or 0
    if current < token_floor:
        orchestrator.max_completion_tokens = token_floor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_files = extra_files or []
    N = focus_round_limit()
    R = repair_round_limit()

    best_kernel = kernel_code
    best_report = report or {}
    best_pack = build_analysis_pack(
        header_code=header_code, kernel_code=best_kernel, report=best_report
    )
    transcript: dict[str, Any] = {
        "schema": TRANSCRIPT_SCHEMA,
        "benchmark": bench,
        "rounds": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def _validate(code: str, tag: str) -> tuple[bool, str, dict[str, Any], str]:
        ok, err = compile_check_cpp(code, header_code, header_name, extra_files=extra_files)
        if not ok:
            return False, err, {}, "compile"
        if not testbench_code:
            return False, "benchmark has no testbench for csim", {}, "testbench"
        outcome = _run_synth_csim_cosim(
            code,
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
        if not synth.get("success"):
            return False, synth.get("error") or "csynth failed", {}, "csynth"
        csim_pass = (
            csim_summary is None
            or csim_summary.get("passed")
            or csim_summary.get("status") == "passed"
        )
        if not csim_pass:
            return False, (csim_summary or {}).get("error") or "csim failed", synth.get("report") or {}, "csim"
        return True, "", synth.get("report") or {}, "csynth"

    for round_idx in range(1, N + 1):
        pack = build_analysis_pack(
            header_code=header_code, kernel_code=best_kernel, report=best_report
        )
        analysis_text = render_analysis_pack(pack)
        prior = _transcript_brief(transcript)
        header_snip = header_code[:4000]
        kernel_snip = best_kernel[:120000]

        analyst_user = (
            f"## Prior focus-group transcript\n{prior}\n\n"
            f"## Analysis pack\n{analysis_text}\n\n"
            f"## Header ({header_name})\n```cpp\n{header_snip}\n```\n\n"
            f"## Kernel (reference only)\n```cpp\n{kernel_snip}\n```\n"
        )
        analyst_text = _call_llm(
            orchestrator,
            [
                {"role": "system", "content": _ANALYST_SYSTEM},
                {"role": "user", "content": analyst_user},
            ],
        )

        plans: dict[str, str] = {}
        filed = ""
        for member in SPECIALIST_IDS:
            spec_user = (
                f"## Analysis pack\n{analysis_text}\n\n"
                f"## Theoretical analyst\n{analyst_text}\n\n"
                f"## Plans already filed this round\n{filed or '(none yet)'}\n\n"
                f"## Prior rounds\n{prior}\n\n"
                f"## Kernel (reference only — plan only)\n```cpp\n{kernel_snip}\n```\n"
            )
            plan = _call_llm(
                orchestrator,
                [
                    {"role": "system", "content": _specialist_system(member)},
                    {"role": "user", "content": spec_user},
                ],
            )
            plans[member] = plan
            filed += f"\n### {member}\n{plan}\n"

        plans_text = filed
        chair_user = (
            f"## Analysis pack\n{analysis_text}\n\n"
            f"## Theoretical analyst\n{analyst_text}\n\n"
            f"## Specialist plans\n{plans_text}\n\n"
            f"## Prior rounds\n{prior}\n\n"
            f"## Current kernel\n```cpp\n{kernel_snip}\n```\n"
        )
        chair_reply = _call_llm(
            orchestrator,
            [
                {"role": "system", "content": _CHAIR_SYSTEM},
                {"role": "user", "content": chair_user},
            ],
        )
        cand_kernel = extract_kernel_block(chair_reply) or best_kernel
        repairs: list[dict[str, Any]] = []
        legal, err, cand_report, stage = _validate(
            cand_kernel, join_temp_tag(bench, STEP_TAG, f"r{round_idx}")
        )
        for repair_index in range(1, R + 1):
            if legal:
                break
            repair_user = _REPAIR_USER.format(
                stage=stage,
                error=(err or "")[:8000],
                plan_text=plans_text[:8000],
                kernel_code=cand_kernel[:120000],
            )
            repair_reply = _call_llm(
                orchestrator,
                [
                    {"role": "system", "content": _CHAIR_SYSTEM},
                    {"role": "user", "content": repair_user},
                ],
            )
            extracted = extract_kernel_block(repair_reply)
            if extracted:
                cand_kernel = extracted
            legal, err, cand_report, stage = _validate(
                cand_kernel, join_temp_tag(bench, STEP_TAG, f"r{round_idx}a{repair_index}")
            )
            repairs.append(
                {
                    "repair_index": repair_index,
                    "stage": stage,
                    "ok": legal,
                    "error": (err or "")[:1000],
                }
            )

        reason = ""
        accepted = False
        cand_pack: Optional[dict[str, Any]] = None
        if not legal:
            reason = f"illegal after repairs: {err}"
        else:
            cand_pack = build_analysis_pack(
                header_code=header_code, kernel_code=cand_kernel, report=cand_report
            )
            if should_accept(cand_pack, best_pack, part=part):
                accepted = True
                reason = "legal, coverage ok, under budget, strictly lower latency"
                best_kernel = cand_kernel
                best_report = cand_report
                best_pack = cand_pack
            else:
                gate = cand_pack.get("coverage")
                gate_ok = gate.ok if isinstance(gate, CoverageResult) else False
                reason = (
                    "rejected: coverage failed"
                    if not gate_ok
                    else "rejected: not faster or over budget"
                )

        (out_dir / f"r{round_idx}_candidate.cpp").write_text(cand_kernel, encoding="utf-8")
        (out_dir / f"r{round_idx}_report.json").write_text(
            json.dumps(cand_report or {}, indent=2, default=str) + "\n", encoding="utf-8"
        )
        cov = (cand_pack or {}).get("coverage")
        if isinstance(cov, CoverageResult):
            cov_dump = {"ok": cov.ok, "reason": cov.reason}
        else:
            cov_dump = cov
        analysis_dump = {
            k: v
            for k, v in (cand_pack or pack).items()
            if k != "report"
        }
        analysis_dump["coverage"] = cov_dump
        if isinstance(pack.get("coverage"), CoverageResult):
            pack_cov = {"ok": pack["coverage"].ok, "reason": pack["coverage"].reason}
        else:
            pack_cov = pack.get("coverage")
        seed_analysis = {k: v for k, v in pack.items() if k != "report"}
        seed_analysis["coverage"] = pack_cov

        transcript["rounds"].append(
            {
                "round": round_idx,
                "analysis": seed_analysis,
                "analyst": analyst_text,
                "plans": plans,
                "chair_decision": "emit one kernel from surviving plans",
                "repairs": repairs,
                "accepted": accepted,
                "reason": reason,
                "latency_cycles": (cand_report or {}).get("latency_cycles") if legal else None,
                "candidate_regions": (cand_pack or {}).get("regions") if cand_pack else None,
            }
        )

    selected_path = out_dir / "selected.cpp"
    selected_path.write_text(best_kernel, encoding="utf-8")
    (out_dir / "selected_report.json").write_text(
        json.dumps(best_report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    transcript["final"] = {
        "success": True,
        "latency_cycles": best_pack.get("latency_cycles"),
        "coverage": {
            "ok": best_pack["coverage"].ok,
            "reason": best_pack["coverage"].reason,
        }
        if isinstance(best_pack.get("coverage"), CoverageResult)
        else best_pack.get("coverage"),
        "selected": selected_path.name,
    }
    transcript["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "focus_group_transcript.json").write_text(
        json.dumps(transcript, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / f"{bench}_focus_group_prompts.json").write_text(
        json.dumps(prompt_text_for_docs(), indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "schema": "post_flash_focus_group_result_v1",
        "benchmark": bench,
        "success": True,
        "latency_cycles": best_pack.get("latency_cycles"),
        "transcript": "focus_group_transcript.json",
        "selected": selected_path.name,
    }
    (out_dir / f"{bench}_focus_group_result.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return FocusGroupOutcome(bench=bench, success=True, out_dir=str(out_dir), result=result)


def resolve_kernel_from_dir(kernel_dir: Path, bench: str) -> tuple[Optional[Path], Optional[Path]]:
    kernel_dir = Path(kernel_dir)
    for cpp in (
        kernel_dir / f"{bench}.cpp",
        kernel_dir / f"{bench}_dse.cpp",
        kernel_dir / f"{bench}_selected.cpp",
    ):
        if cpp.is_file():
            report = kernel_dir / cpp.name.replace(".cpp", "_report.json")
            if not report.is_file():
                alt = kernel_dir / f"{bench}_report.json"
                report = alt if alt.is_file() else None
            return cpp, report
    return None, None


def run_focus_group_for_kernel_dir(
    *,
    bench: str,
    bench_dir: Path,
    kernel_dir: Path,
    orchestrator: Any,
    skip_existing: bool = True,
) -> FocusGroupOutcome:
    from c2hls import _load_benchmark_inputs

    kernel_dir = Path(kernel_dir)
    out_dir = kernel_dir / STEP_TAG
    result_path = out_dir / f"{bench}_focus_group_result.json"
    if skip_existing and result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("success"):
                return FocusGroupOutcome(bench, True, str(out_dir), result=existing)
        except json.JSONDecodeError:
            pass

    cpp, report_path = resolve_kernel_from_dir(kernel_dir, bench)
    if cpp is None:
        return FocusGroupOutcome(bench, False, str(out_dir), error="no kernel cpp in kernel_dir")
    kernel_code = cpp.read_text(encoding="utf-8")
    report: dict[str, Any] = {}
    if report_path and report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report = loaded
        except json.JSONDecodeError:
            report = {}

    inputs = _load_benchmark_inputs(str(bench_dir)) if bench_dir and Path(bench_dir).is_dir() else {}
    header_code = inputs.get("header_code") or ""
    if not header_code:
        header_file = kernel_dir / "kernel.h"
        if header_file.is_file():
            header_code = header_file.read_text(encoding="utf-8")
    header_name = inputs.get("header_name") or "kernel.h"
    meta = inputs.get("meta") or {}
    top_function = (
        meta.get("translated_hls_top")
        or meta.get("hls_top")
        or meta.get("kernel_top")
        or bench
    )
    return run_focus_group(
        bench=bench,
        kernel_code=kernel_code,
        header_code=header_code,
        header_name=header_name,
        report=report,
        orchestrator=orchestrator,
        out_dir=out_dir,
        testbench_code=inputs.get("testbench_code") or "",
        extra_files=inputs.get("extra_files") or [],
        top_function=top_function,
        part=meta.get("part") or getattr(orchestrator, "part", "xcu280-fsvh2892-2L-e"),
        clock_ns=float(meta.get("clock_ns") or getattr(orchestrator, "clock_ns", 3.33)),
    )


def maybe_chain_focus_group(
    *,
    bench: str,
    bench_dir: Path,
    kernel_dir: Path,
    orchestrator: Any,
    skip_existing: bool = True,
) -> Optional[FocusGroupOutcome]:
    if not focus_group_enabled():
        return None
    try:
        return run_focus_group_for_kernel_dir(
            bench=bench,
            bench_dir=bench_dir,
            kernel_dir=kernel_dir,
            orchestrator=orchestrator,
            skip_existing=skip_existing,
        )
    except Exception as exc:
        _LOG.exception("[focus_group] %s error: %s", bench, exc)
        return None


__all__ = [
    "COVERAGE_RATIO_MIN",
    "CoverageResult",
    "FocusGroupOutcome",
    "SPECIALIST_IDS",
    "STEP_TAG",
    "TRANSCRIPT_SCHEMA",
    "ancestor_trip_product",
    "arithmetic_floor",
    "build_analysis_pack",
    "classify_scope_region",
    "coverage_gate",
    "focus_group_enabled",
    "focus_round_limit",
    "maybe_chain_focus_group",
    "parse_pe_simd_from_kernel",
    "prompt_text_for_docs",
    "render_analysis_pack",
    "repair_round_limit",
    "resolve_kernel_from_dir",
    "run_focus_group",
    "run_focus_group_for_kernel_dir",
    "should_accept",
]
