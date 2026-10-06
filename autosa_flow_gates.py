"""Behavioral gates for AutoSA-ordered flash → compute → I/O.

No Vitis. Reports are csynth summaries: latency_cycles, interval, dsp.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GateVerdict:
    ok: bool
    reason: str


def _num(report: Optional[dict[str, Any]], key: str) -> Optional[float]:
    if not report:
        return None
    try:
        return float(report.get(key))
    except (TypeError, ValueError):
        return None


def compute_architecture_ok(
    report: Optional[dict[str, Any]],
    *,
    min_dsp: int,
) -> GateVerdict:
    dsp = _num(report, "dsp")
    if dsp is None or dsp < min_dsp:
        return GateVerdict(
            False,
            f"compute: DSP {dsp} below {min_dsp} (still one weak datapath, not many workers)",
        )
    return GateVerdict(
        True,
        f"compute: DSP {int(dsp)} ≥ {min_dsp}; kernel may still be load+compute+store",
    )


def io_overlap_ok(
    report: Optional[dict[str, Any]],
    *,
    max_latency_over_interval: float = 1.15,
) -> GateVerdict:
    lat = _num(report, "latency_cycles")
    ii = _num(report, "interval")
    if lat is None or ii is None or ii <= 0:
        return GateVerdict(False, "I/O: missing latency or interval")
    ratio = lat / ii
    if ratio > max_latency_over_interval:
        return GateVerdict(
            False,
            f"I/O: one run is {lat:.0f} vs room-time {ii:.0f} ({ratio:.2f}×); "
            "memory is not filling the next chapter during compute",
        )
    return GateVerdict(
        True,
        f"I/O: one run {lat:.0f} ≈ slowest room {ii:.0f} ({ratio:.2f}×)",
    )


def flash_min_dsp() -> Optional[int]:
    """Hard DSP floor for the flash step. Unset/invalid → gate off."""
    raw = os.getenv("C2HLS_FLASH_MIN_DSP", "").strip()
    if not raw.isdigit():
        return None
    return max(1, int(raw))


def flash_dsp_floor_ok(report: Optional[dict[str, Any]], *, min_dsp: int) -> bool:
    dsp = _num(report, "dsp")
    return dsp is not None and dsp >= min_dsp


def flash_dsp_floor_error(
    report: Optional[dict[str, Any]],
    *,
    min_dsp: int,
) -> Optional[str]:
    """Reject text for a flash kernel whose csynth DSP is below the cutoff."""
    if flash_dsp_floor_ok(report, min_dsp=min_dsp):
        return None
    dsp = _num(report, "dsp")
    dsp_s = "missing" if dsp is None else str(int(dsp))
    lat = _num(report, "latency_cycles")
    lat_s = "unknown" if lat is None else str(int(lat))
    return (
        f"REJECTED: csynth DSP={dsp_s} is below the hard cutoff DSP>={min_dsp}. "
        f"This kernel was not accepted. Latency was {lat_s} cycles. "
        f"DSP={dsp_s} means a weak datapath (typically one MAC / a k-loop that "
        f"updates a single C[i][j] or crow[j], flash leftover ~3–10 DSP). "
        f"Rewrite so many MACs run in parallel until csynth reports DSP>={min_dsp}. "
        f"Keep the exact extern C top, parameter list, and legal INTERFACE lines."
    )


def flash_dsp_floor_initial_guidance(min_dsp: int) -> str:
    return (
        f"HARD DSP CUTOFF for this flash rewrite: csynth DSP must be >={min_dsp}. "
        f"A kernel with DSP below {min_dsp} will be rejected even if it compiles "
        f"and csim-passes. Single-digit DSP (one MAC, k-recurrence on one C[i][j]) "
        f"is not accepted. Emit enough parallel multipliers."
    )


def flash_max_dsp() -> Optional[int]:
    """Hard DSP ceiling (U280 = 9024). Unset/invalid → gate off."""
    raw = os.getenv("C2HLS_FLASH_MAX_DSP", "").strip()
    if not raw.isdigit():
        return None
    return max(1, int(raw))


def flash_dsp_ceiling_ok(report: Optional[dict[str, Any]], *, max_dsp: int) -> bool:
    dsp = _num(report, "dsp")
    return dsp is not None and dsp <= max_dsp


def flash_dsp_ceiling_error(
    report: Optional[dict[str, Any]],
    *,
    max_dsp: int,
) -> Optional[str]:
    """Reject text when csynth DSP is above the device/legal cap."""
    if flash_dsp_ceiling_ok(report, max_dsp=max_dsp):
        return None
    dsp = _num(report, "dsp")
    dsp_s = "missing" if dsp is None else str(int(dsp))
    lat = _num(report, "latency_cycles")
    lat_s = "unknown" if lat is None else str(int(lat))
    return (
        f"REJECTED: csynth DSP={dsp_s} is above the hard cap DSP<={max_dsp}. "
        f"This kernel was not accepted. Latency was {lat_s} cycles. "
        f"The Alveo U280 has 9024 DSP; a point above that cannot place. "
        f"Cut parallel MACs: smaller PE_BLK, smaller ROW_UF, a K_TILE so "
        f"PE_BLK*K_TILE*dsp_per_mac stays under {max_dsp}, or fewer unrolled "
        f"output channels for conv. Keep the exact extern C top and ABI."
    )


def flash_dsp_ceiling_initial_guidance(max_dsp: int) -> str:
    return (
        f"HARD DSP CAP for this flash rewrite: csynth DSP must be <={max_dsp} "
        f"(U280 device is 9024). A kernel with DSP above {max_dsp} will be "
        f"rejected even if it compiles. If PE_BLK*K (or O*I*K*K for conv) "
        f"would overshoot, tile K or unroll a subset of output channels."
    )


def _env_on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def flash_dsp_redo() -> bool:
    """Fill-DSP flash variant (not a random floor). Off unless env is set."""
    return _env_on("C2HLS_FLASH_DSP_REDO")


def _pct_env(name: str, default: int = 90) -> float:
    raw = os.getenv(name, str(default)).strip()
    if not raw.isdigit():
        return max(1, min(100, default)) / 100.0
    return max(1, min(100, int(raw))) / 100.0


def flash_dsp_fill_pct() -> float:
    """Minimum DSP utilization of the device cap to count as filled (default 90%)."""
    return _pct_env("C2HLS_FLASH_DSP_FILL_PCT", 90)


def flash_block_pct() -> float:
    """Other-resource utilization that blocks adding more DSP (default 90%)."""
    return _pct_env("C2HLS_FLASH_BLOCK_PCT", 90)


def _device_caps(part: Optional[str] = None) -> dict[str, float]:
    try:
        from rubric import _device_limits_for_part
    except Exception:
        return {
            "bram": 4032,
            "dsp": 9024,
            "ff": 2607360,
            "lut": 1303680,
            "uram": 960,
        }
    active = part or os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    caps = dict(_device_limits_for_part(active) or {})
    return {
        key: float(caps[key])
        for key in ("bram", "dsp", "ff", "lut", "uram")
        if key in caps
    }


def _resource_utils(
    report: Optional[dict[str, Any]],
    *,
    part: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    caps = _device_caps(part)
    out: dict[str, dict[str, float]] = {}
    for key, cap in caps.items():
        if cap <= 0:
            continue
        used = _num(report, key)
        if used is None:
            continue
        out[key] = {
            "used": float(used),
            "capacity": float(cap),
            "utilization": float(used) / float(cap),
        }
    return out


def flash_resource_cap_ok(
    report: Optional[dict[str, Any]],
    *,
    part: Optional[str] = None,
) -> bool:
    utils = _resource_utils(report, part=part)
    if not utils:
        return False
    return all(row["utilization"] < 1.0 for row in utils.values())


def flash_resource_cap_error(
    report: Optional[dict[str, Any]],
    *,
    part: Optional[str] = None,
) -> Optional[str]:
    """Reject when any of DSP/BRAM/FF/LUT/URAM is at or above 100% of the device."""
    if flash_resource_cap_ok(report, part=part):
        return None
    utils = _resource_utils(report, part=part)
    if not utils:
        return (
            "REJECTED: csynth resource report is missing BRAM/DSP/FF/LUT/URAM "
            "counts; cannot prove the kernel stays below 100% of U280."
        )
    over = [
        (
            f"{key} {int(row['used'])} is {100.0 * row['utilization']:.1f}% of "
            f"device {int(row['capacity'])}"
        )
        for key, row in utils.items()
        if row["utilization"] >= 1.0
    ]
    lat = _num(report, "latency_cycles")
    lat_s = "unknown" if lat is None else str(int(lat))
    return (
        "REJECTED: at least one resource is at or above 100% of Alveo U280. "
        + "; ".join(over)
        + f". Latency was {lat_s} cycles. Shrink parallel MACs or tiling until "
        "DSP, BRAM, FF, LUT, and URAM are all strictly below 100%."
    )


def flash_dsp_fill_ok(
    report: Optional[dict[str, Any]],
    *,
    part: Optional[str] = None,
    fill_pct: Optional[float] = None,
    block_pct: Optional[float] = None,
) -> bool:
    """True when DSP is filled, or another resource is blocking more DSP."""
    if not flash_resource_cap_ok(report, part=part):
        return False
    utils = _resource_utils(report, part=part)
    dsp_row = utils.get("dsp")
    if not dsp_row:
        return False
    fill = flash_dsp_fill_pct() if fill_pct is None else fill_pct
    block = flash_block_pct() if block_pct is None else block_pct
    if dsp_row["utilization"] >= fill:
        return True
    other = [
        row["utilization"]
        for key, row in utils.items()
        if key != "dsp"
    ]
    return bool(other) and max(other) >= block


def flash_dsp_fill_error(
    report: Optional[dict[str, Any]],
    *,
    part: Optional[str] = None,
) -> Optional[str]:
    """Reject a legal-but-sparse datapath that still has device headroom."""
    if flash_dsp_fill_ok(report, part=part):
        return None
    utils = _resource_utils(report, part=part)
    dsp_row = utils.get("dsp")
    dsp_s = "missing" if not dsp_row else str(int(dsp_row["used"]))
    dsp_pct = "missing" if not dsp_row else f"{100.0 * dsp_row['utilization']:.1f}%"
    lat = _num(report, "latency_cycles")
    lat_s = "unknown" if lat is None else str(int(lat))
    fill = int(round(flash_dsp_fill_pct() * 100))
    others = ", ".join(
        f"{key} {100.0 * row['utilization']:.1f}%"
        for key, row in utils.items()
        if key != "dsp"
    ) or "BRAM/FF/LUT/URAM unreported"
    max_dsp = int(_device_caps(part).get("dsp") or 9024)
    return (
        f"REJECTED: csynth DSP={dsp_s} ({dsp_pct} of {max_dsp}) is not a filled "
        f"datapath. Headroom remains ({others}). Latency was {lat_s} cycles. "
        f"DSP=2000 or DSP=5000 is not done while LUT/FF/BRAM still have room. "
        f"Emit more parallel MACs until DSP is at least {fill}% of {max_dsp} "
        f"or another resource is near 100% and would overflow. Stay strictly "
        f"below 100% of every resource. Keep the exact extern C top and ABI."
    )


def flash_dsp_redo_initial_guidance(min_dsp: int, max_dsp: int) -> str:
    fill = int(round(flash_dsp_fill_pct() * 100))
    return (
        f"FLASH DSP REDO: fill Alveo U280 DSP, then keep the lowest-latency "
        f"legal kernel. Csynth DSP must be >={min_dsp} (single-digit leftover "
        f"is rejected) and strictly <={max_dsp} (device DSP=9024 is 100%). "
        f"A 64-wide k unroll is only ~320 DSP — that is not a stop. Keep adding "
        f"parallel MACs (unroll k AND spatial PE / i,j) until DSP is at least "
        f"{fill}% of {max_dsp} OR BRAM/FF/LUT/URAM is also near 100% and would "
        f"overflow if you added more. DSP=2000 or DSP=5000 is NOT done if other "
        f"resources still have headroom. Never at or above 100% of DSP/BRAM/FF/"
        f"LUT/URAM. Several flash candidates are generated; the pipeline selects "
        f"the filled legal kernel with the lowest latency_cycles (not interval). "
        f"Keep the exact extern C top and kernel.h ABI. Do not clone AutoSA kernel0."
    )


def flash_dsp_redo_latency_key(report: Optional[dict[str, Any]]) -> tuple[float, float]:
    lat = _num(report, "latency_cycles")
    worst = _num(report, "latency_cycles_worst")
    if lat is None:
        lat = float("inf")
    if worst is None:
        worst = lat
    return (float(lat), float(worst))


def select_flash_dsp_redo_winner(
    attempts: Optional[list[dict[str, Any]]],
    *,
    part: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Lowest latency_cycles among filled, under-100% successful attempts."""
    ok: list[dict[str, Any]] = []
    for attempt in attempts or []:
        if not attempt.get("success"):
            continue
        report = attempt.get("report")
        if not isinstance(report, dict):
            continue
        if not flash_resource_cap_ok(report, part=part):
            continue
        if not flash_dsp_fill_ok(report, part=part):
            continue
        ok.append(attempt)
    if not ok:
        return None
    return min(ok, key=lambda a: flash_dsp_redo_latency_key(a.get("report") or {}))


def flash_k_tile() -> Optional[int]:
    raw = os.getenv("C2HLS_FLASH_K_TILE", "").strip()
    if not raw.isdigit():
        return None
    return max(8, int(raw))


def flash_onchip_tile() -> bool:
    raw = os.getenv("C2HLS_FLASH_ONCHIP_TILE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def flash_onchip_tile_initial_guidance(k_tile: Optional[int] = None) -> str:
    kt = k_tile if k_tile is not None else flash_k_tile()
    kt_s = str(kt) if kt else "64 or 128"
    return (
        "MANDATORY tile for this flash rewrite: the 64³ complete-partition "
        "recipe does not apply. Do not allocate or ARRAY_PARTITION complete "
        f"full A[I][K] / B[J][K] / C[I][J]. Use K_TILE={kt_s} (and a small "
        "I_TILE / J_TILE=PE_BLK). Local buffers only cover the tile: "
        "A_tile[I_TILE][K_TILE], B_tile[PE_BLK][K_TILE], C_acc[I_TILE][PE_BLK]. "
        "Walk k0 += K_TILE; accumulate write-once into C_acc; store the C tile. "
        "PE_BLK * K_TILE * dsp_per_mac must land in [5000, 9024] when that is "
        "legal. Separate load_A_tile and load_B_tile (never fused). Keep kernel.h ABI."
    )


def flash_row_uf() -> Optional[int]:
    """Mandatory output-row unroll for the flash step. Unset/invalid → off."""
    raw = os.getenv("C2HLS_FLASH_ROW_UF", "").strip()
    if not raw.isdigit():
        return None
    return max(1, int(raw))


def flash_pe_blk() -> Optional[int]:
    """Mandatory compute PE_BLK for the flash step. Unset/invalid → off."""
    raw = os.getenv("C2HLS_FLASH_PE_BLK", "").strip()
    if not raw.isdigit():
        return None
    return max(1, int(raw))


def flash_pe_blk_initial_guidance(pe_blk: int) -> str:
    return (
        f"MANDATORY compute width for this flash rewrite: PE_BLK={pe_blk} "
        f"(or PE={pe_blk}). Nested `for (i) for (j += {pe_blk})` with "
        f"`#pragma HLS UNROLL` of {pe_blk} affine write-once C_local stores "
        f"per PIPELINE II=1 iteration. Do not use PE_BLK=8. Keep II=1; do not "
        f"linearize with g>>k. The Alveo U280 budget is 9024 DSP — emit the "
        f"multipliers PE_BLK={pe_blk} needs."
    )



def flash_skill_bin() -> Optional[str]:
    """Curated flash skill bin. Unset -> None."""
    raw = os.getenv("C2HLS_FLASH_SKILL_BIN", "").strip().lower().replace("-", "_")
    if raw in {"generic", "gemm_family", "systolic_io", "onchip", "zero_shot"}:
        return raw
    return None


def flash_generic_hls_initial_guidance() -> str:
    return (
        "MANDATORY for this flash rewrite: generic HLS only. PIPELINE II=1 on "
        "the hot loop after independent inners are UNROLL. Stage the working "
        "set when it fits. Distinct m_axi bundles, max_widen_bitwidth=512, "
        "LANES = 512/element_bits. Affine write-once locals, no m_axi RMW, "
        "no linearized g. Do not emit PE_BLK. Do not emit DATAFLOW unless "
        "there are INLINE-off tasks. Keep the kernel.h ABI."
    )


def flash_gemm_family_initial_guidance() -> str:
    return (
        "MANDATORY for this flash rewrite: GEMM-family rules. Keep kernel.h "
        "ranks: A is IxK, B is JxK, C is IxJ (catapult uses I_P/J_P/K_P). "
        "Gold C from zero — skip load_C. Stage A/B/C when they fit. Affine "
        "write-once C_local. UNROLL k as an adder tree. Three AXI bundles, "
        "512-bit. Separate load_A and load_B (do not fuse A+B into one "
        "lockstep loop). Do not emit an AutoSA interconnect netlist."
    )


def flash_systolic_io_initial_guidance() -> str:
    return (
        "MANDATORY for this flash rewrite: hide load-store with a PE stream "
        "array, not spend-chip DSP. INLINE-off load_A / load_B / mm_pe / "
        "drain_B / store_C under DATAFLOW. Packed ap_uint SIMD streams. "
        "Crow ram_2p, one pe_kj pipeline. Forward B along the PE chain. "
        "When I/PE_NUM>1, wrap DATAFLOW in the top i0 tile loop. Keep the "
        "kernel.h ABI. This is not the 940-class on-chip pack."
    )

def flash_onchip() -> bool:
    """On-chip 940-class GEMM pack. Unset → off."""
    raw = os.getenv("C2HLS_FLASH_ONCHIP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def flash_onchip_initial_guidance() -> str:
    return (
        "MANDATORY for this flash rewrite: on-chip GEMM (940-class), not a "
        "systolic array and not tile ping-pong. Ignore FLASH prompt item 5 "
        "(DATAFLOW / buf[2] tiles). Do not emit #pragma HLS DATAFLOW or "
        "kernel0. Stage all of A and B locally when they fit and PE_BLK*K "
        "stays under 9024 DSP; otherwise tile K (do not complete-partition "
        "full 256+ K or 1024³ arrays). LANES=16 512-bit "
        "load/store. TWO loops: load_A then load_B (nested i / j += LANES). "
        "Never emit load_A_B / load_AB. Never zip A and B in one pipeline "
        "(A_loc=A[...] and B_loc=B[...] in the same loop body). HLS can "
        "overlap two independent load modules; a fused lockstep load is a "
        "hard reject (csynth can look cheap; RTL stalls on either AXI). "
        "Affine write-once C_local[row][j0+p] = dot — no g>>k, "
        "no C RMW. One II=1 pipeline over I*(J/PE_BLK) groups with K fully "
        "unrolled as an adder tree of independent acc0..acc7 (do NOT fold "
        "k-slices into one acc[p] += eight products — that inflates pipeline "
        "depth, e.g. 1032 vs 938). PE_BLK is 16/32/64 after II=1. Csynth DSP "
        "must be <= 9024. If PE_BLK*K would overshoot, tile K instead of "
        "unrolling all of K. LCST is correct: one-call latency ≈ "
        "max(load_A,load_B)+compute+store."
    )


_FUSED_AB_LABEL_RE = re.compile(
    r"\b(?:load_A_B|load_AB|load_ab|load_AandB|load_A_and_B|"
    r"load_both(?:_AB)?|fused_load(?:_A_B|_AB)?)\b",
    re.IGNORECASE,
)
_A_FROM_AXI_RE = re.compile(
    r"\b(?:A_loc|A_local|A_buf|local_A)\s*\[[^\]]+\](?:\s*\[[^\]]+\])?\s*=\s*A\s*\[",
)
_B_FROM_AXI_RE = re.compile(
    r"\b(?:B_loc|B_local|B_buf|local_B)\s*\[[^\]]+\](?:\s*\[[^\]]+\])?\s*=\s*B\s*\[",
)
_A_FROM_AXI_FLAT_RE = re.compile(
    r"\b(?:A_loc|A_local|A_buf|local_A)\s*(?:\[[^\]]+\])?\s*=\s*A\s*\[",
)
_B_FROM_AXI_FLAT_RE = re.compile(
    r"\b(?:B_loc|B_local|B_buf|local_B)\s*(?:\[[^\]]+\])?\s*=\s*B\s*\[",
)


def _strip_c_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    return re.sub(r"//.*?$", " ", code, flags=re.M)


def _for_loop_bodies(code: str) -> list[str]:
    """Bodies of every `for (...)` with braces, including nested loops."""
    bodies: list[str] = []
    i = 0
    n = len(code)
    while True:
        match = re.search(r"\bfor\s*\(", code[i:])
        if not match:
            break
        p = i + match.end()
        depth = 1
        while p < n and depth:
            ch = code[p]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            p += 1
        while p < n and code[p].isspace():
            p += 1
        if p < n and code[p] == "{":
            start = p + 1
            depth = 1
            p += 1
            while p < n and depth:
                ch = code[p]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                p += 1
            bodies.append(code[start : p - 1])
        i = max(p, i + match.end())
    return bodies


def _loop_zips_ab(body: str) -> bool:
    a_hit = _A_FROM_AXI_RE.search(body) or _A_FROM_AXI_FLAT_RE.search(body)
    b_hit = _B_FROM_AXI_RE.search(body) or _B_FROM_AXI_FLAT_RE.search(body)
    if not (a_hit and b_hit):
        return False
    return re.search(r"\bfor\s*\(", body) is None


def flash_fused_ab_in_code(code: Optional[str]) -> bool:
    """True when source fuses A and B DRAM loads into one loop/label."""
    if not code:
        return False
    src = _strip_c_comments(code)
    if _FUSED_AB_LABEL_RE.search(src):
        return True
    return any(_loop_zips_ab(body) for body in _for_loop_bodies(src))


def flash_fused_ab_in_report(report: Optional[dict[str, Any]]) -> bool:
    if not report:
        return False
    blob = json.dumps(report)
    return bool(
        re.search(r"load_A_B|Pipeline_load_A_B|load_AB\b", blob)
    )


def flash_fused_ab_error(
    code: Optional[str] = None,
    report: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Reject text when A and B are loaded in one lockstep pipeline."""
    if not flash_fused_ab_in_code(code) and not flash_fused_ab_in_report(report):
        return None
    return (
        "REJECTED: fused A+B load. This kernel was not accepted. "
        "Do not emit load_A_B / load_AB and do not zip A_loc=A[...] and "
        "B_loc=B[...] in the same loop body. Csynth can price that lockstep "
        "II=1 as cheap (the 871/906-class failure); RTL stalls on either AXI. "
        "Emit two loops: load_A then load_B (nested i / j += LANES). HLS can "
        "overlap those two independent modules. Then compute, then store_C. "
        "Keep the exact extern C top, parameter list, and legal INTERFACE lines."
    )


def flash_tile_pp() -> bool:
    """Mandatory in-GEMM tile ping-pong for flash. Unset → off."""
    raw = os.getenv("C2HLS_FLASH_TILE_PP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def flash_tile_pp_initial_guidance() -> str:
    return (
        "MANDATORY overlap for this flash rewrite: ping-pong TILES inside this "
        "one GEMM, not DATAFLOW around a full-matrix load/compute/store. "
        "Today max(load_A, load_B)+compute+store is the one-call latency; bulk "
        "DATAFLOW does not hide it. Use 2 or 4 large tiles (IT rows x PE_BLK "
        "cols). Double-buffer `A_buf[2][IT][K]`, `B_buf[2][PE_BLK][K]`, "
        "`C_buf[2][IT][PE_BLK]` (or equivalent) with `t & 1`. INLINE-off "
        "load_tile / compute_tile / store_tile; `#pragma HLS DATAFLOW` **inside** "
        "the tile loop so load(t+1), compute(t), store(t-1) overlap. "
        "Arrays declared inside that loop without `buf[2]` / `t & 1` are "
        "**not** ping-pong (explicit double-buffer required). Load B once "
        "outside the tile loop; do not reload full B every tile. Keep PE_BLK=16, LANES=16, "
        "affine write-once C, PIPELINE II=1. Do not use 64 one-row tiles "
        "(each tile re-pays compute depth). Skip load_C. Target one-call "
        "latency ≈ max(load, compute, store), not the LCST sum. "
        "Interval < latency is not overlap."
    )


def flash_row_uf_initial_guidance(row_uf: int) -> str:
    return (
        f"MANDATORY output-row unroll for this flash rewrite: ROW_UF={row_uf}. "
        f"In the same PIPELINE II=1 compute iteration, UNROLL r=0..{row_uf}-1 "
        f"independent output rows and PE_BLK columns. Trip count is "
        f"(I/{row_uf})*(J/PE_BLK), not I*(J/PE_BLK). "
        f"uint16 is ~1 DSP/MAC so PE_BLK=16 times K=64 is only ~1024 DSP — "
        f"ROW_UF={row_uf} is how this kernel spends 5000+ DSP. "
        f"Stay at or under the U280 cap 9024 DSP. Affine write-once C, "
        f"separate load_A / load_B, kernel.h ABI."
    )


def within_rank1(agent_cycles: float, *, rank1: int, tol: float = 1.02) -> bool:
    return agent_cycles <= rank1 * tol


# ---------------------------------------------------------------------------
# Ping-pong DATAFLOW: explicit buf[2] + t&1 + tile DATAFLOW + csynth max
# ---------------------------------------------------------------------------
# Arrays declared inside a DATAFLOW tile loop are **not** ping-pong. Require
# explicit double buffers (`buf[2]` / ping-pong names) and `t & 1` / `t%2`,
# INLINE-off load / compute / store, and B loaded once outside the tile loop.
# Pattern: load(t+1), compute(t), store(t-1) (or NT>=2 + buf[2] equivalent).
# One-call latency tracks max(load, compute, store), not the LCST sum.
# Canonical syntax fail: 101836 wrap (DATAFLOW + A_loc inside, load_B every tile).
# Interval < latency is the next kernel launch, not in-GEMM tile overlap.

_SERIAL_SUM_REL = 0.10
_MAX_OVERLAP_SLACK = 1.25
_MAX_OVERLAP_FILL = 256
_DATAFLOW_PRAGMA_RE = re.compile(r"#\s*pragma\s+HLS\s+DATAFLOW\b", re.IGNORECASE)
_INLINE_OFF_RE = re.compile(r"#\s*pragma\s+HLS\s+INLINE\s+off\b", re.IGNORECASE)
_PP_INDEX_RE = re.compile(
    r"(?:t|tile|i0)(?:\s*[+-]\s*1)?\s*\)?\s*&\s*1"
    r"|(?:t|tile)\s*%\s*2",
    re.IGNORECASE,
)
_LOAD_B_CALL_RE = re.compile(r"\bload_B\s*\(", re.IGNORECASE)
_ARRAY_DECL_RE = re.compile(
    r"\b(?:static\s+)?(?:const\s+)?(?:volatile\s+)?"
    r"(?:data_t|float|double|half|int|char|short|long|"
    r"ap_uint|ap_int|uint\d+_t|int\d+_t)\s+"
    r"([A-Za-z_]\w*)\s*((?:\[[^\]]+\])+)\s*;",
)
_SKIP_MODULE_RE = re.compile(
    r"entry_proc|control_s_axi|gmem\d+_m_axi|\bglbl\b",
    re.IGNORECASE,
)
_LOAD_MODULE_RE = re.compile(r"\bload", re.IGNORECASE)
_STORE_MODULE_RE = re.compile(r"store|drain", re.IGNORECASE)
_PARENT_MODULE_RE = re.compile(
    r"dataflow_parent|dataflow_in_loop",
    re.IGNORECASE,
)
_SUMMARY_LAT_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\w+)\s*\|"
)


def _near(actual: float, target: float, rel: float) -> bool:
    if target <= 0:
        return False
    return abs(actual - target) <= rel * target


def _for_loop_spans(code: str) -> list[tuple[int, int]]:
    """Inclusive body start / exclusive body end for every braced `for`."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(code)
    while True:
        match = re.search(r"\bfor\s*\(", code[i:])
        if not match:
            break
        p = i + match.end()
        depth = 1
        while p < n and depth:
            ch = code[p]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            p += 1
        while p < n and code[p].isspace():
            p += 1
        if p < n and code[p] == "{":
            start = p + 1
            depth = 1
            p += 1
            while p < n and depth:
                ch = code[p]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                p += 1
            spans.append((start, p - 1))
        i = max(p, i + match.end())
    return spans


def _innermost_loop_body(code: str, pos: int) -> Optional[str]:
    containing = [(s, e) for s, e in _for_loop_spans(code) if s <= pos < e]
    if not containing:
        return None
    start, end = min(containing, key=lambda se: se[1] - se[0])
    return code[start:end]


def tile_loop_around_dataflow(code: Optional[str]) -> bool:
    """True when `#pragma HLS DATAFLOW` sits inside a for-loop body."""
    if not code:
        return False
    src = _strip_c_comments(code)
    return any(
        _innermost_loop_body(src, m.start()) is not None
        for m in _DATAFLOW_PRAGMA_RE.finditer(src)
    )


def pingpong_arrays_inside_dataflow_loop(code: Optional[str]) -> bool:
    """True when an array is declared in the DATAFLOW-region loop.

    This is **not** sufficient for ping-pong. HLS may channel-duplicate those
    arrays; the judge requires explicit `buf[2]` and `t & 1`.
    """
    if not code:
        return False
    src = _strip_c_comments(code)
    for match in _DATAFLOW_PRAGMA_RE.finditer(src):
        body = _innermost_loop_body(src, match.start())
        if body and _ARRAY_DECL_RE.search(body):
            return True
    return False


def _c_function_spans(code: str) -> list[tuple[str, int, int]]:
    """(name, body_start, body_end_exclusive) for each `void name(...) { ... }`."""
    out: list[tuple[str, int, int]] = []
    i = 0
    n = len(code)
    while True:
        match = re.search(r"(?:static\s+)?void\s+([A-Za-z_]\w*)\s*\(", code[i:])
        if not match:
            break
        name = match.group(1)
        p = i + match.end()
        depth = 1
        while p < n and depth:
            ch = code[p]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            p += 1
        while p < n and code[p].isspace():
            p += 1
        if p < n and code[p] == "{":
            start = p + 1
            depth = 1
            p += 1
            while p < n and depth:
                ch = code[p]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                p += 1
            out.append((name, start, p - 1))
            i = p
        else:
            i = i + match.end()
    return out


def explicit_double_buffer(code: Optional[str]) -> bool:
    """True when a `[2]` array or a ping/pong named pair is declared."""
    if not code:
        return False
    src = _strip_c_comments(code)
    names: list[str] = []
    for decl in _ARRAY_DECL_RE.finditer(src):
        name = decl.group(1) or ""
        dims = decl.group(2) or ""
        names.append(name)
        if re.search(r"\[\s*2\s*\]", dims):
            return True
    has_ping = any(re.search(r"ping", n, re.IGNORECASE) for n in names)
    has_pong = any(re.search(r"pong", n, re.IGNORECASE) for n in names)
    return bool(has_ping and has_pong)


def pingpong_index(code: Optional[str]) -> bool:
    """True when the C indexes a bank with `t & 1`, `t%2`, or ping/pong names."""
    if not code:
        return False
    src = _strip_c_comments(code)
    if _PP_INDEX_RE.search(src):
        return True
    return bool(
        re.search(r"\bping\b", src, re.IGNORECASE)
        and re.search(r"\bpong\b", src, re.IGNORECASE)
    )


def inline_off_lcs_tasks(code: Optional[str]) -> bool:
    """True when INLINE-off load, compute, and store tasks exist."""
    if not code:
        return False
    src = _strip_c_comments(code)
    has_load = has_compute = has_store = False
    for name, start, end in _c_function_spans(src):
        blob = src[max(0, start - 120) : end]
        if not _INLINE_OFF_RE.search(blob):
            continue
        lower = name.lower()
        if "load" in lower:
            has_load = True
        if "compute" in lower:
            has_compute = True
        if "store" in lower or "drain" in lower:
            has_store = True
    return has_load and has_compute and has_store


def _dataflow_loop_bodies(code: str) -> list[str]:
    bodies: list[str] = []
    for match in _DATAFLOW_PRAGMA_RE.finditer(code):
        body = _innermost_loop_body(code, match.start())
        if body:
            bodies.append(body)
    return bodies


def load_b_in_dataflow_allowed() -> bool:
    """Opt-in: C2HLS_PP_LOAD_B_IN_DF lets load_B be a tile DATAFLOW task."""
    return os.environ.get("C2HLS_PP_LOAD_B_IN_DF", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_b_inside_dataflow_loop(code: Optional[str]) -> bool:
    """True when full-matrix `load_B(` is called inside the tile DATAFLOW loop."""
    if not code:
        return False
    src = _strip_c_comments(code)
    return any(_LOAD_B_CALL_RE.search(body) for body in _dataflow_loop_bodies(src))


def tile_trip_at_least_two(code: Optional[str]) -> bool:
    """True when the tile loop is NT>=2 (or an equivalent trip of 2+)."""
    if not code:
        return False
    src = _strip_c_comments(code)
    defined = re.search(r"#\s*define\s+NT\s+(\d+)", src)
    if defined:
        return int(defined.group(1)) >= 2
    bound = re.search(r"for\s*\(\s*int\s+t\s*=\s*0\s*;\s*t\s*<\s*(\d+)", src)
    if bound:
        return int(bound.group(1)) >= 2
    if re.search(r"t\s*<\s*NT\b", src):
        return True
    if re.search(r"t\s*<\s*\(\s*I\s*/", src):
        return True
    return False


def pingpong_arrays_outside_dataflow_loop(code: Optional[str]) -> bool:
    """True when a `[2]` (or named ping/pong) array sits outside the DF loop."""
    if not code:
        return False
    src = _strip_c_comments(code)
    inside_spans = []
    for match in _DATAFLOW_PRAGMA_RE.finditer(src):
        containing = [(s, e) for s, e in _for_loop_spans(src) if s <= match.start() < e]
        if containing:
            inside_spans.append(min(containing, key=lambda se: se[1] - se[0]))
    for decl in _ARRAY_DECL_RE.finditer(src):
        dims = decl.group(2) or ""
        name = decl.group(1) or ""
        is_pp = bool(re.search(r"\[\s*2\s*\]", dims)) or bool(
            re.search(r"ping|pong|buf0|buf1", name, re.IGNORECASE)
        )
        if not is_pp:
            continue
        pos = decl.start()
        if any(s <= pos < e for s, e in inside_spans):
            continue
        return True
    return False


def pingpong_dataflow_code_ok(code: Optional[str]) -> GateVerdict:
    """Source must be explicit ping-pong: buf[2] + t&1 + tile DATAFLOW + LCS tasks."""
    src = code or ""
    stripped = _strip_c_comments(src)
    has_df = bool(_DATAFLOW_PRAGMA_RE.search(stripped))
    tile_loop = tile_loop_around_dataflow(src)
    arrays_inside = pingpong_arrays_inside_dataflow_loop(src)
    has_buf2 = explicit_double_buffer(src)
    has_index = pingpong_index(src)
    has_lcs = inline_off_lcs_tasks(src)
    b_inside = load_b_inside_dataflow_loop(src)
    trip_ok = tile_trip_at_least_two(src)
    if not has_df:
        return GateVerdict(False, "no #pragma HLS DATAFLOW")
    if not tile_loop:
        return GateVerdict(
            False,
            "DATAFLOW is not inside a tile/iteration loop (one-shot full-matrix "
            "load/compute/store is not ping-pong)",
        )
    if not has_buf2:
        extra = (
            " (arrays inside the DATAFLOW loop without buf[2] is not ping-pong)"
            if arrays_inside
            else ""
        )
        return GateVerdict(
            False,
            "no explicit double buffer buf[2] / [2][...] or ping/pong pair" + extra,
        )
    if not has_index:
        return GateVerdict(
            False,
            "double buffer is not indexed with t & 1 / t%2 / ping-pong names",
        )
    if not has_lcs:
        return GateVerdict(
            False,
            "load / compute / store are not INLINE-off tasks",
        )
    if b_inside and not load_b_in_dataflow_allowed():
        return GateVerdict(
            False,
            "load_B is inside the tile DATAFLOW loop; load B once outside and "
            "ping-pong A/C (reloading full B every tile is not ping-pong)",
        )
    if not trip_ok:
        return GateVerdict(False, "tile trip NT<2; need NT>=2 with buf[2]")
    b_note = (
        ", load_B in tile DATAFLOW (C2HLS_PP_LOAD_B_IN_DF)"
        if b_inside
        else ", B loaded outside"
    )
    return GateVerdict(
        True,
        "explicit ping-pong: buf[2] + t&1, tile DATAFLOW, INLINE-off "
        "load/compute/store" + b_note,
    )


def pingpong_dataflow_code_intent(code: Optional[str]) -> dict[str, Any]:
    verdict = pingpong_dataflow_code_ok(code)
    src = code or ""
    has_buf2 = explicit_double_buffer(src)
    has_index = pingpong_index(src)
    b_inside = load_b_inside_dataflow_loop(src)
    ping_pong = bool(
        has_buf2 and has_index and (not b_inside or load_b_in_dataflow_allowed())
    )
    return {
        "dataflow": bool(_DATAFLOW_PRAGMA_RE.search(_strip_c_comments(src))),
        "ping_pong": ping_pong,
        "tile_loop": tile_loop_around_dataflow(src),
        "arrays_inside": pingpong_arrays_inside_dataflow_loop(src),
        "arrays_outside": pingpong_arrays_outside_dataflow_loop(src),
        "explicit_buf2": has_buf2,
        "pingpong_index": has_index,
        "load_b_inside": b_inside,
        "inline_off_tasks": inline_off_lcs_tasks(src),
        "ok": verdict.ok,
        "reason": verdict.reason,
    }


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_csynth_latency_summary(rpt: Optional[str]) -> dict[str, Any]:
    """Kernel latency min/max + interval + pipeline type from a csynth.rpt."""
    empty = {
        "latency_cycles": None,
        "latency_cycles_max": None,
        "interval": None,
        "pipeline_type": None,
    }
    if not rpt:
        return empty
    perf = rpt.split("== Utilization")[0]
    if "+ Latency:" in perf:
        lat_sec = perf.split("+ Latency:", 1)[1]
    else:
        lat_sec = perf
    summary = lat_sec.split("* Instance:")[0]
    match = _SUMMARY_LAT_RE.search(summary)
    if not match:
        return empty
    return {
        "latency_cycles": int(match.group(1)),
        "latency_cycles_max": int(match.group(2)),
        "interval": int(match.group(3)),
        "pipeline_type": match.group(5),
    }


def parse_csynth_instance_latencies(rpt: Optional[str]) -> list[dict[str, Any]]:
    """Process/module latencies from the Performance Estimates instance table."""
    if not rpt:
        return []
    perf = rpt.split("== Utilization")[0]
    if "* Instance:" not in perf:
        return []
    inst_sec = perf.split("* Instance:", 1)[1]
    inst_sec = inst_sec.split("* Loop:", 1)[0]
    rows: list[dict[str, Any]] = []
    for line in inst_sec.splitlines():
        if not line.strip().startswith("|") or line.strip().startswith("|+"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 8:
            continue
        if parts[0].lower() in {"instance", "module"} or parts[1].lower() in {
            "module",
            "instance",
        }:
            continue
        if re.match(r"^[-+]+$", parts[0].replace(" ", "")):
            continue
        lat = _as_int(parts[2])
        if lat is None:
            continue
        name = parts[1]
        if _SKIP_MODULE_RE.search(name):
            continue
        rows.append(
            {
                "instance": parts[0],
                "name": name,
                "latency_cycles": lat,
                "interval": _as_int(parts[6]) if len(parts) > 6 else None,
                "pipeline_type": parts[8] if len(parts) > 8 else None,
            }
        )
    return rows


def _modules_from_feedback(report: dict[str, Any]) -> list[dict[str, Any]]:
    scopes = ((report.get("feedback") or {}).get("scopes") or [])
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        if (scope.get("kind") or "") != "module":
            continue
        if scope.get("parent") is None and int(scope.get("depth") or 0) == 0:
            continue
        name = str(scope.get("name") or "")
        if _SKIP_MODULE_RE.search(name):
            continue
        lat = _as_int(scope.get("latency_cycles"))
        if lat is None:
            continue
        rows.append(
            {
                "name": name,
                "latency_cycles": lat,
                "interval": _as_int(scope.get("interval")),
                "pipeline_type": scope.get("pipelined"),
            }
        )
    return rows


def _normalize_module_row(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("module") or raw.get("instance") or "")
    if not name or _SKIP_MODULE_RE.search(name):
        return None
    lat = _as_int(raw.get("latency_cycles") or raw.get("latency"))
    if lat is None:
        return None
    return {
        "name": name,
        "latency_cycles": lat,
        "interval": _as_int(raw.get("interval")),
        "pipeline_type": raw.get("pipeline_type") or raw.get("pipelined"),
    }


def _find_hls_report_dir(work_dir: Optional[str]) -> Optional[str]:
    if not work_dir:
        return None
    candidates = [
        os.path.join(work_dir, "hls_proj", "sol1", "syn", "report"),
        os.path.join(work_dir, "sol1", "syn", "report"),
        work_dir,
    ]
    for path in candidates:
        if os.path.isdir(path) and any(
            name.endswith("_csynth.rpt") or name == "csynth.rpt"
            for name in os.listdir(path)
        ):
            return path
    return None


def enrich_report_with_csynth_modules(
    report: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Attach process latencies from rpt text, work_dir, or feedback scopes."""
    out: dict[str, Any] = dict(report or {})
    modules = [
        row
        for row in (_normalize_module_row(m) for m in (out.get("modules") or []))
        if row
    ]
    regions = list(out.get("dataflow_regions") or [])
    rpt = out.get("csynth_rpt") or out.get("report_raw") or ""
    if isinstance(rpt, str) and rpt and not modules:
        modules = parse_csynth_instance_latencies(rpt)
        summary = parse_csynth_latency_summary(rpt)
        if out.get("latency_cycles") is None and summary.get("latency_cycles") is not None:
            out["latency_cycles"] = summary["latency_cycles"]
        if out.get("interval") is None and summary.get("interval") is not None:
            out["interval"] = summary["interval"]
        if summary.get("pipeline_type") and not out.get("pipeline_type"):
            out["pipeline_type"] = summary["pipeline_type"]

    work = out.get("work_dir")
    report_dir = _find_hls_report_dir(str(work) if work else None)
    if report_dir and not regions:
        try:
            names = os.listdir(report_dir)
        except OSError:
            names = []
        for fname in sorted(names):
            if not fname.endswith(".rpt"):
                continue
            if fname not in {"csynth.rpt"} and not fname.endswith("_csynth.rpt"):
                continue
            path = os.path.join(report_dir, fname)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            inst = parse_csynth_instance_latencies(text)
            summary = parse_csynth_latency_summary(text)
            stem = fname.replace("_csynth.rpt", "").replace(".rpt", "")
            blob = {
                "name": stem,
                "path": path,
                "latency_cycles": summary.get("latency_cycles"),
                "interval": summary.get("interval"),
                "pipeline_type": summary.get("pipeline_type"),
                "modules": inst,
            }
            if stem in {"csynth"}:
                continue
            if _PARENT_MODULE_RE.search(stem) or (
                summary.get("pipeline_type") or ""
            ).lower() == "dataflow":
                regions.append(blob)
            if not modules and inst and not _PARENT_MODULE_RE.search(stem):
                if "dataflow_in_loop" not in stem:
                    modules = inst
                    if out.get("latency_cycles") is None:
                        out["latency_cycles"] = summary.get("latency_cycles")
                    if out.get("interval") is None:
                        out["interval"] = summary.get("interval")

    if not modules:
        modules = _modules_from_feedback(out)
    if modules:
        out["modules"] = modules
    if regions:
        out["dataflow_regions"] = regions
    return out


def lcst_serial_sum(modules: list[dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    """Return (serial LCST sum, max process). Independent loads overlap."""
    rows = [m for m in modules if (_as_int(m.get("latency_cycles")) or 0) > 0]
    rows = [m for m in rows if not _PARENT_MODULE_RE.search(str(m.get("name") or ""))]
    if not rows:
        return None, None
    lats = [int(m["latency_cycles"]) for m in rows]
    max_p = max(lats)
    loads = [
        int(m["latency_cycles"])
        for m in rows
        if _LOAD_MODULE_RE.search(str(m.get("name") or ""))
    ]
    stores = [
        int(m["latency_cycles"])
        for m in rows
        if _STORE_MODULE_RE.search(str(m.get("name") or ""))
    ]
    computes = [
        int(m["latency_cycles"])
        for m in rows
        if not _LOAD_MODULE_RE.search(str(m.get("name") or ""))
        and not _STORE_MODULE_RE.search(str(m.get("name") or ""))
    ]
    if loads and (computes or stores):
        serial = max(loads) + sum(computes) + sum(stores)
        return serial, max_p
    return sum(lats), max_p


def _infer_dataflow_trip(
    code: Optional[str],
    region: Optional[dict[str, Any]],
) -> Optional[int]:
    if region:
        trip = _as_int(region.get("trip_count"))
        if trip and trip > 0:
            return trip
    src = code or ""
    nt = re.search(r"#\s*define\s+NT\s+(\d+)", src)
    if nt and re.search(r"\bt\s*<\s*NT\b", src):
        return max(1, int(nt.group(1)))
    lit = re.search(
        r"for\s*\(\s*(?:int\s+)?(?:t|i0|tile)\s*=\s*0\s*;\s*(?:t|i0|tile)\s*<\s*(\d+)",
        src,
    )
    if lit:
        return max(1, int(lit.group(1)))
    pe = re.search(r"#\s*define\s+PE(?:_NUM|_BLK)?\s+(\d+)", src)
    if pe and re.search(r"I\s*/\s*PE", src):
        return max(1, 64 // int(pe.group(1)))
    return None


def _dataflow_regions(report: dict[str, Any]) -> list[dict[str, Any]]:
    regions = list(report.get("dataflow_regions") or [])
    out: list[dict[str, Any]] = []
    for raw in regions:
        if not isinstance(raw, dict):
            continue
        mods = [
            row
            for row in (_normalize_module_row(m) for m in (raw.get("modules") or []))
            if row
        ]
        blob = dict(raw)
        blob["modules"] = mods
        out.append(blob)
    return out


def _parent_latency(report: dict[str, Any]) -> Optional[int]:
    for row in report.get("modules") or []:
        name = str(row.get("name") or "")
        if _PARENT_MODULE_RE.search(name):
            lat = _as_int(row.get("latency_cycles"))
            if lat:
                return lat
    for region in _dataflow_regions(report):
        name = str(region.get("name") or "")
        if "parent" in name.lower():
            lat = _as_int(region.get("latency_cycles"))
            if lat:
                return lat
    return None


def pingpong_dataflow_csynth_ok(
    report: Optional[dict[str, Any]],
    *,
    tile_loop: bool,
    code: Optional[str] = None,
) -> GateVerdict:
    """Csynth overlap: one-call / tile-loop latency ≈ max(rooms), not the sum.

    Interval < latency is **not** evidence (that is the next kernel launch).
    Reject kernel latency within ~10% of the serial LCST sum of DATAFLOW
    processes. With a tile loop, the parent should track N * max(rooms),
    not N * sum.
    """
    enriched = enrich_report_with_csynth_modules(report)
    kernel_lat = _as_int(enriched.get("latency_cycles"))
    if kernel_lat is None or kernel_lat <= 0:
        return GateVerdict(False, "I/O: missing latency_cycles for ping-pong overlap")

    regions = _dataflow_regions(enriched)
    inner = None
    for region in regions:
        mods = region.get("modules") or []
        serial, max_p = lcst_serial_sum(mods)
        if serial and max_p and len(mods) >= 2:
            inner = region
            break
    if inner is None:
        for region in regions:
            if region.get("modules"):
                inner = region
                break

    top_mods = [
        m
        for m in (enriched.get("modules") or [])
        if _normalize_module_row(m)
    ]
    top_rooms = [
        m
        for m in top_mods
        if not _PARENT_MODULE_RE.search(str(m.get("name") or ""))
    ]

    if not tile_loop:
        serial, max_p = lcst_serial_sum(top_rooms)
        if serial is None:
            serial, max_p = lcst_serial_sum(top_mods)
        if serial and _near(kernel_lat, serial, _SERIAL_SUM_REL):
            return GateVerdict(
                False,
                f"csynth kernel {kernel_lat} is within 10% of serial LCST sum "
                f"{serial} (max process {max_p}); one-shot DATAFLOW, no tile overlap",
            )
        if max_p and kernel_lat > max_p * _MAX_OVERLAP_SLACK + _MAX_OVERLAP_FILL:
            return GateVerdict(
                False,
                f"csynth kernel {kernel_lat} is far above max(process)={max_p}; "
                "interval<latency is not in-GEMM ping-pong overlap",
            )
        return GateVerdict(
            False,
            "no tile-loop DATAFLOW in source; csynth cannot show load(t+1)/"
            "compute(t)/store(t-1) overlap",
        )

    if inner is not None:
        mods = inner.get("modules") or []
        serial, max_p = lcst_serial_sum(mods)
        inner_lat = _as_int(inner.get("latency_cycles"))
        inner_ii = _as_int(inner.get("interval"))
        trip = _infer_dataflow_trip(code, inner)
        parent = _parent_latency(enriched)
        if serial and max_p:
            n = trip if trip and trip >= 1 else None
            if n == 1:
                if inner_lat and _near(inner_lat, serial, _SERIAL_SUM_REL):
                    return GateVerdict(
                        False,
                        f"DATAFLOW region latency {inner_lat} ≈ sum {serial} "
                        f"(one tile, no overlap); max={max_p}",
                    )
            if n and n >= 2 and parent:
                n_max = n * max_p
                n_sum = n * serial
                if _near(parent, n_sum, _SERIAL_SUM_REL):
                    return GateVerdict(
                        False,
                        f"tile DATAFLOW parent {parent} ≈ N*sum ({n}*{serial}="
                        f"{n_sum}); rooms still serial",
                    )
                if parent <= n_max * _MAX_OVERLAP_SLACK + _MAX_OVERLAP_FILL:
                    return GateVerdict(
                        True,
                        f"tile DATAFLOW parent {parent} ≈ {n}*max({max_p})="
                        f"{n_max} (not {n}*sum={n_sum}); load/compute/store overlap",
                    )
                if parent < 0.85 * n_sum:
                    return GateVerdict(
                        True,
                        f"tile DATAFLOW parent {parent} is below 0.85 of N*sum "
                        f"{n_sum} (max={max_p}, N={n}); rooms overlap",
                    )
                return GateVerdict(
                    False,
                    f"tile DATAFLOW parent {parent} is not near N*max {n_max} "
                    f"and not far from N*sum {n_sum}",
                )
            if inner_ii and max_p and _near(inner_ii, max_p, 0.15):
                if inner_lat and inner_lat < 0.92 * serial:
                    return GateVerdict(
                        True,
                        f"DATAFLOW in-loop II {inner_ii} ≈ max {max_p}; "
                        f"region {inner_lat} < sum {serial}",
                    )
                if inner_lat and _near(inner_lat, serial, _SERIAL_SUM_REL) and (
                    not inner_ii or _near(inner_ii, inner_lat, 0.10)
                ):
                    return GateVerdict(
                        False,
                        f"DATAFLOW in-loop latency {inner_lat} ≈ sum {serial} "
                        "with II≈latency; one big chunk, no overlapping tiles",
                    )
                return GateVerdict(
                    True,
                    f"DATAFLOW in-loop II {inner_ii} ≈ max(process) {max_p}",
                )
            if inner_lat and serial and _near(inner_lat, serial, _SERIAL_SUM_REL):
                return GateVerdict(
                    False,
                    f"DATAFLOW region {inner_lat} ≈ serial sum {serial} "
                    f"(max={max_p}); no overlap",
                )

    serial, max_p = lcst_serial_sum(top_rooms)
    if serial and _near(kernel_lat, serial, _SERIAL_SUM_REL):
        return GateVerdict(
            False,
            f"csynth kernel {kernel_lat} is within 10% of serial LCST sum "
            f"{serial} (max {max_p})",
        )
    parent = _parent_latency(enriched)
    if parent and max_p and parent <= max_p * _MAX_OVERLAP_SLACK + _MAX_OVERLAP_FILL:
        return GateVerdict(
            True,
            f"csynth parent/kernel {parent} tracks max(process) {max_p}",
        )
    if max_p and kernel_lat <= max_p * _MAX_OVERLAP_SLACK + _MAX_OVERLAP_FILL:
        return GateVerdict(
            True,
            f"csynth kernel {kernel_lat} ≈ max(process) {max_p}",
        )
    if not top_mods and not regions:
        return GateVerdict(
            False,
            "csynth process latencies missing; interval<latency is not "
            "ping-pong DATAFLOW overlap",
        )
    return GateVerdict(
        False,
        f"csynth kernel {kernel_lat} does not track max(load,compute,store) "
        f"(max={max_p}, serial_sum={serial})",
    )


def pingpong_dataflow_ok(
    code: Optional[str],
    report: Optional[dict[str, Any]],
) -> GateVerdict:
    """Pass only with legal source **and** csynth overlap (max, not sum)."""
    src = pingpong_dataflow_code_ok(code)
    if not src.ok:
        return src
    return pingpong_dataflow_csynth_ok(
        report,
        tile_loop=True,
        code=code,
    )


# ---------------------------------------------------------------------------
# Enforcement must wrap the flash kernel: keep LANES=16 loads and compute.
# ---------------------------------------------------------------------------
# The 20260909 run accepted a legal tile DATAFLOW at 12642 after flash 4808
# because the repair prompt pasted a scalar II=1 load_B and the overlap
# judge did not compare against flash. Wide 64x64 float is ~256 cycles;
# a scalar II=1 walk is ~4096.

_FLASH_REGRESS_REL = 0.10
_SCALAR_LOAD_CYCLES = 2000
_WIDE_LOAD_CYCLES = 512
_LANES_DEF_RE = re.compile(r"\bLANES\s*=\s*16\b")
_LANES_STEP_RE = re.compile(r"(?:k0|j0|i0)\s*\+=\s*(?:LANES|16)\b")
_LANES_LOOP_RE = re.compile(
    r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(?:LANES|16)\s*;"
)
_PE_DEF_RE = re.compile(r"#\s*define\s+PE(?:_NUM)?\s+(\d+)")
_SIMD_DEF_RE = re.compile(r"#\s*define\s+SIMD\s+(\d+)")
_PE_NEST_RE = re.compile(
    r"(?:i0\s*\+=\s*PE(?:_NUM)?|\bCrow\s*\[|pe_mac\s*:|k0\s*\+=\s*SIMD)",
    re.IGNORECASE,
)


def wide_axi_copy_in_code(code: Optional[str]) -> bool:
    """True when load/store walks LANES (16) elements per PIPELINE iteration."""
    if not code:
        return False
    src = _strip_c_comments(code)
    if _LANES_DEF_RE.search(src) and _LANES_STEP_RE.search(src):
        return True
    if _LANES_STEP_RE.search(src) and _LANES_LOOP_RE.search(src):
        return True
    return False


def pe_simd_macros(code: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """PE × SIMD from compute-rewrite macros. None if the kernel is not a PE nest."""
    src = _strip_c_comments(code or "")
    pe_m = _PE_DEF_RE.search(src)
    simd_m = _SIMD_DEF_RE.search(src)
    pe = int(pe_m.group(1)) if pe_m else None
    simd = int(simd_m.group(1)) if simd_m else None
    return pe, simd


def pe_compute_nest_in_code(code: Optional[str]) -> bool:
    """True when the PE×SIMD MAC nest is still present (not unroll-K GEMM)."""
    return bool(_PE_NEST_RE.search(_strip_c_comments(code or "")))


def _load_module_latencies(report: Optional[dict[str, Any]]) -> list[int]:
    enriched = enrich_report_with_csynth_modules(report)
    lats: list[int] = []

    def take(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            name = str(row.get("name") or "")
            if _PARENT_MODULE_RE.search(name):
                continue
            if not _LOAD_MODULE_RE.search(name):
                continue
            lat = _as_int(row.get("latency_cycles"))
            if lat and lat > 0:
                lats.append(lat)

    take(list(enriched.get("modules") or []))
    for region in _dataflow_regions(enriched):
        take(list(region.get("modules") or []))
    return lats


def enforcement_keep_flash_ok(
    *,
    code: Optional[str],
    report: Optional[dict[str, Any]],
    baseline_code: Optional[str] = None,
    baseline_report: Optional[dict[str, Any]] = None,
) -> GateVerdict:
    """Reject ping-pong that dropped flash 512-bit loads or worsened latency.

    Sequential flash (no tile DATAFLOW yet) is not judged here — the overlap
    gate still fails it. No baseline → skip.
    """
    if not baseline_report:
        return GateVerdict(True, "no flash baseline")
    if not tile_loop_around_dataflow(code):
        return GateVerdict(True, "not a ping-pong candidate yet; keep-flash N/A")

    if wide_axi_copy_in_code(baseline_code) and not wide_axi_copy_in_code(code):
        return GateVerdict(
            False,
            "dropped flash LANES=16 / 512-bit load-store; restore k0 += LANES "
            "with an UNROLLed lane loop (do not walk one float per cycle)",
        )

    base_pe, base_simd = pe_simd_macros(baseline_code)
    if base_pe and base_simd:
        cur_pe, cur_simd = pe_simd_macros(code)
        if (cur_pe, cur_simd) != (base_pe, base_simd) or not pe_compute_nest_in_code(
            code
        ):
            return GateVerdict(
                False,
                f"dropped compute-rewrite PE {base_pe} × SIMD {base_simd}; "
                "paste that nest into compute_tile (do not replace with unroll-K)",
            )

    base_lat = _as_int((baseline_report or {}).get("latency_cycles"))
    cur_lat = _as_int((report or {}).get("latency_cycles"))
    if base_lat and cur_lat and cur_lat > base_lat * (1.0 + _FLASH_REGRESS_REL) + 64:
        return GateVerdict(
            False,
            f"enforcement regression: kernel {cur_lat} > flash {base_lat}×1.10; "
            "wrap the flash load/compute, do not replace them",
        )

    base_loads = _load_module_latencies(baseline_report)
    cur_loads = _load_module_latencies(report)
    if base_loads and cur_loads:
        base_max = max(base_loads)
        cur_max = max(cur_loads)
        if base_max <= _WIDE_LOAD_CYCLES and cur_max >= _SCALAR_LOAD_CYCLES:
            return GateVerdict(
                False,
                f"load room {cur_max} looks like a scalar AXI walk; flash loads "
                f"were {base_max} (LANES=16 / ~256 cycles)",
            )

    base_dsp = _as_int((baseline_report or {}).get("dsp"))
    cur_dsp = _as_int((report or {}).get("dsp"))
    if base_dsp and cur_dsp is not None and cur_dsp < 0.85 * base_dsp:
        return GateVerdict(
            False,
            f"dropped flash compute: DSP {cur_dsp} < 0.85× flash {base_dsp}",
        )
    return GateVerdict(
        True,
        "kept flash wide load/compute; latency not worse than flash",
    )
