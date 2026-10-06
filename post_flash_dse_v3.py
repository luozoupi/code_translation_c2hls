"""DSE v3 prompt rules and optional harness checks on top of DSE v2.

``C2HLS_DSE_V3=1`` adds prompt rules, the v3 skill file, and a per-trial float
architecture floor ``floor(5 × PE × SIMD × 0.9)`` (the grid's float DSP
multiplier, default 5). That replaces the constant grid ``min_dsp`` on the
trial. It does not change the grid membership band, csynth TCL, or winner
grouping.

``C2HLS_DSE_V3_HARNESS=1`` implies v3 and also:
- injects ``config_array_partition -complete_threshold 0``
- omits ``config_compile -jobs`` (this Vitis errors on ``-jobs``)
- rejects a 512-bit load/store pipeline whose trip is still one float per cycle
- keeps latency winners inside a K-coverage class

The v2 skill JSON is unchanged. With both flags unset, each trial still uses
the grid ``min_dsp``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

WIDE_BEAT_FLOATS = 16
V3_SKILL_ID = "hls-dse-wide-512-load-store-hoist-b"
HARNESS_COMPLETE_THRESHOLD_TCL = "config_array_partition -complete_threshold 0"

V3_RULE_LINES = (
    "Rewrite load and store as well as the compute nest. Global arrays move in "
    "512-bit beats (16 floats). Contiguous index innermost, PIPELINE II=1, beat "
    "unrolled. Keep the existing m_axi max_widen_bitwidth=512.",
    "If a tile address does not depend on an enclosing loop, load that tile "
    "outside the loop. Specifically B[j0][k] is outside i0.",
    "If the load stage is many times the compute stage, widen the load. Do not "
    "raise PE or SIMD to shorten a scalar load.",
    "Do not add a K loop the seed does not have in order to cut latency.",
    "Do not stream B once per PE-row group across all of J and K. Reuse a B tile "
    "across i0.",
    "Tile-level overlap of load, compute, and store is allowed. Do not emit "
    "AutoSA kernel0 or a C for-loop of mm_pe inside DATAFLOW. A dataflow region "
    "of the three tile stages is allowed.",
)

V3_PROMPT_RULES = (
    "## DSE v3 load, store, and tile reuse (mandatory)\n"
    + "\n".join(V3_RULE_LINES)
    + "\n"
)

_FOR_COND_RE = re.compile(r"for\s*\([^;]*;\s*([^;]+);", re.M)
_UPPER_RE = re.compile(r"(?:<=|<)\s*([A-Za-z_][A-Za-z0-9_]*)")
_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)\b", re.M
)
_FULL_K_BOUNDS = frozenset({"K", "K_P"})
_PARTIAL_K_BOUNDS = frozenset({"TK"})
_LOAD_STORE_RE = re.compile(r"load|store", re.IGNORECASE)


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "on"}


def dse_v3_harness_enabled() -> bool:
    return _env_flag("C2HLS_DSE_V3_HARNESS") is True


def dse_v3_enabled() -> bool:
    """True for v3 prompts. Harness implies v3; v3 alone leaves the harness off.

    ``C2HLS_DSE_V4`` wins: v3 rules are not appended when v4 is on.
    """
    if _env_flag("C2HLS_DSE_V4") is True:
        return False
    if dse_v3_harness_enabled():
        return True
    return _env_flag("C2HLS_DSE_V3") is True


def float_pair_min_dsp(pe: int, simd: int, *, dsp_mul: int = 5) -> int:
    """Architecture floor for one float PE×SIMD pair.

    ``floor(dsp_mul × PE × SIMD × 0.9)``. Exact integer form is
    ``(dsp_mul * pe * simd * 9) // 10``. A product that is not an integer
    truncates toward zero, which matches ``int()`` for non-negative values.
    """
    return (int(dsp_mul) * int(pe) * int(simd) * 9) // 10


def resolve_dse_v3_skills_path() -> Path:
    raw = os.getenv("C2HLS_DSE_V3_SKILL_ENTRIES_JSON", "").strip()
    if raw:
        return Path(raw)
    from c2hls_paths import POST_FLASH_DSE_V3_SKILL_ENTRIES_JSON

    return POST_FLASH_DSE_V3_SKILL_ENTRIES_JSON


def append_v3_prompt_rules(text: str) -> str:
    """Append v3 rules when v3 is on. Unset leaves the prompt object unchanged."""
    if not dse_v3_enabled():
        return text
    return text.rstrip() + "\n\n" + V3_PROMPT_RULES


def harness_csynth_kwargs() -> dict[str, Any]:
    """Extra ``run_hls_synthesis`` kwargs. Empty unless the harness is on."""
    if not dse_v3_harness_enabled():
        return {}
    return {
        "allow_compile_jobs": False,
        "extra_csynth_tcl": HARNESS_COMPLETE_THRESHOLD_TCL + "\n",
    }


def harness_csynth_preamble() -> str:
    """Csynthes TCL prefix actually used for a DSE trial."""
    from hls_eval import csynth_pre_commands

    opts = harness_csynth_kwargs()
    if not opts:
        return csynth_pre_commands()
    return csynth_pre_commands(
        allow_compile_jobs=bool(opts["allow_compile_jobs"]),
        extra_commands=str(opts["extra_csynth_tcl"]),
    )


def _strip_cpp_comments(code: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", code or "", flags=re.S)
    return re.sub(r"//.*?$", "", text, flags=re.M)


def _int_defines(code: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in _DEFINE_RE.finditer(code or "")}


def loop_upper_idents(kernel_code: str) -> list[str]:
    text = _strip_cpp_comments(kernel_code)
    found: list[str] = []
    for match in _FOR_COND_RE.finditer(text):
        upper = _UPPER_RE.search(match.group(1))
        if upper:
            found.append(upper.group(1))
    return found


def classify_k_coverage(kernel_code: str) -> str:
    """``full_k`` if a loop runs to K/K_P, ``partial_k`` if only to TK.

    A loop over all of K wins over an inner TK tile. No K or TK bound is
    ``unknown`` (MAC count is then not claimed).
    """
    bounds = set(loop_upper_idents(kernel_code))
    if bounds & _FULL_K_BOUNDS:
        return "full_k"
    if bounds & _PARTIAL_K_BOUNDS:
        return "partial_k"
    return "unknown"


def assess_k_coverage(
    kernel_code: str,
    *,
    i: Optional[int] = None,
    j: Optional[int] = None,
    k: Optional[int] = None,
) -> dict[str, Any]:
    """Coverage tag plus a MAC count only when the bound is static and obvious.

    ``full_gemm`` is true only for ``full_k``. A partial-K kernel is never a
    full-GEMM win.
    """
    tag = classify_k_coverage(kernel_code)
    defines = _int_defines(_strip_cpp_comments(kernel_code))
    mac: Optional[int] = None
    if tag == "full_k" and i and j and k and i > 0 and j > 0 and k > 0:
        mac = int(i) * int(j) * int(k)
    elif tag == "partial_k":
        ti = defines.get("TI")
        tj = defines.get("TJ")
        tk = defines.get("TK")
        if ti and tj and tk:
            mac = ti * tj * tk
        elif tk and i and j and i > 0 and j > 0:
            bounds = set(loop_upper_idents(kernel_code))
            if ({"I", "I_P"} & bounds) and ({"J", "J_P"} & bounds):
                mac = int(i) * int(j) * tk
    return {
        "coverage": tag,
        "mac_count": mac,
        "full_gemm": tag == "full_k",
    }


def tile_jk(kernel_code: str, *, j: int, k: int) -> tuple[int, int]:
    """B-tile shape: TJ×TK when the kernel defines them, else header J×K."""
    defines = _int_defines(_strip_cpp_comments(kernel_code))
    return int(defines.get("TJ", j)), int(defines.get("TK", k))


def scalar_element_trip_illegal(*, interface_bits: int, trip: int, elements: int) -> bool:
    """True when a 512-bit load/store pipeline still trips once per float.

    A widened trip (elements/16) is legal. A 64×64 tile is 4096 scalar floats
    and 256 beats of 16 floats.
    """
    try:
        bits = int(interface_bits)
        trip_i = int(trip)
        elems = int(elements)
    except (TypeError, ValueError):
        return False
    if bits != 512 or elems <= 0 or elems % WIDE_BEAT_FLOATS != 0:
        return False
    wide = elems // WIDE_BEAT_FLOATS
    return trip_i == elems and trip_i != wide


def scalar_element_trip_error(
    *,
    interface_bits: int,
    trip: int,
    elements: int,
    loop_name: str = "load/store",
) -> str:
    if not scalar_element_trip_illegal(
        interface_bits=interface_bits, trip=trip, elements=elements
    ):
        return ""
    wide = int(elements) // WIDE_BEAT_FLOATS
    return (
        f"illegal scalar {loop_name}: trip count {int(trip)} equals the full "
        f"scalar element count ({int(elements)} floats, one float per cycle) "
        f"on a 512-bit interface; widened trip is {wide} (elements/16)"
    )


def kernel_interface_bits(kernel_code: str) -> Optional[int]:
    found = [
        int(bits)
        for bits in re.findall(r"max_widen_bitwidth\s*=\s*(\d+)", kernel_code or "")
    ]
    if not found:
        return None
    if 512 in found:
        return 512
    return found[0]


def load_store_pipeline_trips(report: Optional[dict[str, Any]]) -> list[tuple[str, int]]:
    """(loop name, trip) for pipelined load/store scopes in a csynth report."""
    scopes = ((report or {}).get("feedback") or {}).get("scopes") or []
    out: list[tuple[str, int]] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        kind = scope.get("kind")
        if kind not in (None, "loop"):
            continue
        name = str(scope.get("name") or "")
        scope_id = str(scope.get("scope_id") or "")
        if not (_LOAD_STORE_RE.search(name) or _LOAD_STORE_RE.search(scope_id)):
            continue
        pipelined = scope.get("pipelined")
        if isinstance(pipelined, str) and pipelined.strip().lower() in {
            "no",
            "none",
            "off",
            "-",
        }:
            continue
        trip = scope.get("trip_count")
        if trip is None:
            continue
        try:
            trip_i = int(trip)
        except (TypeError, ValueError):
            continue
        label = name or scope_id or "load/store"
        out.append((label, trip_i))
    return out


def harness_post_csynth_error(
    *,
    kernel_code: str,
    report: Optional[dict[str, Any]],
    tile_rows: int,
    tile_cols: int,
) -> str:
    """Scalar-load reject. Empty when the harness is off, including v3-only."""
    if not dse_v3_harness_enabled():
        return ""
    if kernel_interface_bits(kernel_code) != 512:
        return ""
    try:
        elements = int(tile_rows) * int(tile_cols)
    except (TypeError, ValueError):
        return ""
    for name, trip in load_store_pipeline_trips(report):
        err = scalar_element_trip_error(
            interface_bits=512,
            trip=trip,
            elements=elements,
            loop_name=name,
        )
        if err:
            return err
    return ""


__all__ = [
    "HARNESS_COMPLETE_THRESHOLD_TCL",
    "V3_PROMPT_RULES",
    "V3_RULE_LINES",
    "V3_SKILL_ID",
    "WIDE_BEAT_FLOATS",
    "append_v3_prompt_rules",
    "assess_k_coverage",
    "classify_k_coverage",
    "dse_v3_enabled",
    "dse_v3_harness_enabled",
    "harness_csynth_kwargs",
    "harness_csynth_preamble",
    "harness_post_csynth_error",
    "kernel_interface_bits",
    "load_store_pipeline_trips",
    "loop_upper_idents",
    "resolve_dse_v3_skills_path",
    "scalar_element_trip_error",
    "scalar_element_trip_illegal",
    "tile_jk",
]
