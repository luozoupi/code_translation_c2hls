"""flash_refined: the c2hls flash step without its hold-back cues.

Applied at runtime by run_flash_cc.py --refined (the c2hls checkout is read only). apply() changes,
for the flash step only:

Prompts (prompt_c2hls)
  * Instruction_c2hls_flash (system): "modest unroll", "array partition on small buffers",
    "DATAFLOW only when buffering stays simple", "narrow host ABI", "Prefer surgical ... edits on the
    existing structure" -> sizing to the device budget, overlap via DATAFLOW / ping-pong, 512-bit
    internal types with the top-level ABI unchanged, full restructuring allowed.
  * q_optimize_flash (user): items 3-6, the "Prefer a compact ... modest unroll" paragraph and the
    "Prefer surgical edits" output rule are replaced the same way. Every other line (ABI, bundles,
    burst legality, loop labels, interface check, output format) is unchanged.
Flow (c2hls)
  * _render_step_resource_constraints: for `flash`, the 12x-per-step RESOURCE CONSTRAINTS block and the
    "Reserve DSP-heavy unrolling for later steps" paragraph become a DEVICE BUDGET block (must be faster,
    every resource < 100% of the device, no growth cap). Other steps keep the original block.
  * C2HLS_STEP_REGRESSION_THRESHOLDS_JSON: the flash regression guard keeps latency (must not get
    slower) and drops the per-step resource ratios; device capacity is enforced by the DSP-redo
    resource-cap gate instead.
  * DSP floor on, as in scripts/pc2/start_autosa_mm_flash_dsp_redo.sh: C2HLS_FLASH_DSP_REDO=1,
    MIN_DSP=300, MAX_DSP=9024, DSP_FILL_PCT=90 (fill gate + < 100% resource-cap gate).
Skills
  * the refined pack in this directory (make_refined_skills.py; source pack unchanged).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_PACKS = {1: HERE / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined.json",
               2: HERE / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined_v2.json"}
NEUTRAL_PACK = HERE / "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2_refined_v2_neutral.json"

# Neutral mode: the hold-back cues stay removed, but nothing asks the model to size the design to the device or to
# use most of the DSP. Every resource only has to stay below 100% (a cap, not a requirement). (old, new) pairs are
# applied to the refined system prompt, user prompt and DEVICE BUDGET block; each must match exactly once.
NEUTRAL_SYSTEM = [
    ("You produce one complete, high-performance kernel in a single rewrite (flash mode), sized to the target device.",
     "You produce one complete, high-performance kernel in a single rewrite (flash mode)."),
    ("unrolling and array partitioning\n  sized to the device budget (DSP, BRAM/URAM, memory ports, timing),",
     "unrolling and array partitioning\n  (any factor, as long as every resource stays below 100% of the device),"),
    ("""- For compute-bound kernels, scale the parallel datapath toward the device budget: a small datapath on a
  large device is a performance bug, not a safe choice.
""", ""),
    ("once the datapath is sized, per-tile compute cycles", "for the datapath you choose, per-tile compute cycles"),
    ("A filled datapath that waits on memory is not done.", "A datapath that waits on memory is not done."),
]
NEUTRAL_USER = [
    ("""to the factor the device budget supports — DSP,
   BRAM/URAM ports and timing — not a fixed small factor.""",
     """with any factor you choose; the only limit is that
   every resource stays below 100% of the device."""),
    ("""Size the design to the target device. For compute-bound kernels (for example
matrix multiply), the parallel datapath should use most of the device's DSP
budget while every resource stays below 100%; leaving most of the device idle
is not the best endpoint. Restructure""", "Restructure"),
    ("Balance compute and data movement once the datapath is sized:", "Balance compute and data movement for the datapath you choose:"),
]
NEUTRAL_BLOCK = [("DEVICE BUDGET for the", "DEVICE LIMITS for the"),
                 ("There is no growth cap relative to the current design: size the datapath to the device.",
                  "There is no growth cap relative to the current design.")]
NEUTRAL_FIX = """HARD REJECT — the flash design is waiting on memory. Do not reuse this kernel.

Csynth: DSP={dsp}, latency {latency_cycles} cycles.

Why it was rejected:
{reject_reason}

Fix the cause named above. The datapath is waiting on memory: keep reused operands resident on chip across tiles
(load each operand once per reuse window), move every array, including C, in 512-bit beats at II=1, and overlap
loads/stores with compute (DATAFLOW or ping-pong). Per-tile compute must exceed per-tile transfers.
Stay strictly below 100% of DSP/BRAM/FF/LUT/URAM. Keep the exact `extern "C"` top and kernel.h ABI.

Previous (rejected) code:
```cpp
{hls_code}
```

{attempt_history}Header:
```cpp
{header_code}
```

Before the fence, one sentence: (a) the cause and (b) your fix. Then the complete kernel in ```cpp.
"""
NEUTRAL_BANNED = ("device budget", "most of the device", "most of the DSP", "sized to the target device",
                  "performance bug", "idle", "fill target", "filled datapath", "size the datapath to the device")


def _neutralize(text: str, pairs) -> str:
    for old, new in pairs:
        assert text.count(old) == 1, f"neutral edit not found once: {old[:60]!r}"
        text = text.replace(old, new)
    return text
REFINED_SKILLS = SKILL_PACKS[1]

# Version 2: compute/transfer balance. A filled datapath must not starve on memory.
BALANCE_RATIO = float(os.environ.get("FLASH_REFINED_BALANCE_RATIO", "1.5"))
DSP_PER_MAC = float(os.environ.get("FLASH_REFINED_DSP_PER_MAC", "5"))  # fp32 multiply-add on UltraScale+
SYSTEM_BALANCE = """
- Balance compute and data movement: once the datapath is sized, per-tile compute cycles must exceed
  per-tile transfer cycles. Keep reused operands resident on chip across tiles, move every array
  (including the output read-modify-write) in 512-bit beats at II=1, and overlap loads and stores with
  compute. A filled datapath that waits on memory is not done."""
USER_BALANCE_ANCHOR = """is not the best endpoint. Restructure the code as much as needed — a full
rewrite is allowed."""
USER_BALANCE = """is not the best endpoint. Restructure the code as much as needed — a full
rewrite is allowed.

Balance compute and data movement once the datapath is sized: estimate per-tile
compute cycles (work per tile / parallel MACs) and per-tile transfer cycles
(words moved / 16 floats per beat). Transfers must stay below compute: keep the
operand that is reused across tiles resident on chip instead of reloading it
every tile, move every array — including C's read-modify-write — in 512-bit
beats at II=1, and overlap the next tile's loads and the previous tile's stores
with the current tile's compute (DATAFLOW or ping-pong). The flow rejects a
design whose latency is far above total work / parallel MACs."""
BALANCE_FIX = """HARD REJECT — the flash design does not use the device efficiently. Do not reuse this kernel.

Csynth: DSP={dsp}, latency {latency_cycles} cycles. Device DSP budget {max_dsp} (fill target about {fill_pct}%).

Why it was rejected:
{reject_reason}

Fix the cause named above:
- DSP not filled: add parallel MACs (spatial PE rows/columns and k lanes) until DSP is near the fill
  target or another resource would exceed 100%.
- Compute/transfer imbalance: the datapath is waiting on memory. Keep reused operands resident on chip
  across tiles (load each operand once per reuse window), move every array, including C, in 512-bit
  beats at II=1, and overlap loads/stores with compute (DATAFLOW or ping-pong). Per-tile compute must
  exceed per-tile transfers.
Stay strictly below 100% of DSP/BRAM/FF/LUT/URAM. Keep the exact `extern "C"` top and kernel.h ABI.

Previous (rejected) code:
```cpp
{hls_code}
```

{attempt_history}Header:
```cpp
{header_code}
```

Before the fence, one sentence: (a) the cause and (b) your fix. Then the complete kernel in ```cpp.
"""


def _total_macs() -> int | None:
    raw = os.environ.get("FLASH_REFINED_TOTAL_MACS", "").strip()
    return int(raw) if raw else None


def balance_error(report: dict) -> str | None:
    """Reject when latency is far above the compute floor implied by the design's own DSP count."""
    macs, lat, dsp = _total_macs(), (report or {}).get("latency_cycles"), (report or {}).get("dsp")
    if not macs or not isinstance(lat, (int, float)) or not dsp:
        return None
    parallel = float(dsp) / DSP_PER_MAC
    floor = macs / parallel
    # Non-mm kernels: the design can be no faster than moving its data once through a 512-bit port.
    mem = float(os.environ.get("FLASH_REFINED_MEM_CYCLES", "0") or 0)
    floor = max(floor, mem)
    ratio = float(lat) / floor
    if ratio <= BALANCE_RATIO:
        return None
    scopes = ((report.get("feedback") or {}).get("scopes")) or []
    loops = [x for x in scopes if x.get("kind") == "loop" and x.get("pipelined") == "yes"]
    key = lambda x: x.get("latency_cycles") or x.get("latency") or 0
    top = "; ".join(f"{x.get('name')} latency={key(x)} II={x.get('pipeline_ii')} trip={x.get('trip_count')}"
                    for x in sorted(loops, key=key, reverse=True)[:5])
    return (f"compute/transfer imbalance: latency {int(lat)} cycles is {ratio:.2f}x the compute floor "
            f"{int(floor)} cycles (= max({macs} MACs / ({int(dsp)} DSP / {DSP_PER_MAC:g} DSP per MAC), memory floor "
            f"{int(mem)} cycles)); the limit is "
            f"{BALANCE_RATIO:g}x. The datapath is waiting on memory. Longest pipelined loops per call: {top}.")

REFINED_SYSTEM = """You are an expert in FPGA High-Level Synthesis (HLS) using Xilinx Vitis HLS.
You produce one complete, high-performance kernel in a single rewrite (flash mode), sized to the target device.

Key optimization techniques you may combine:
- Tiling and on-chip staging, pipelining (`#pragma HLS PIPELINE II=1`), unrolling and array partitioning
  sized to the device budget (DSP, BRAM/URAM, memory ports, timing), DATAFLOW / ping-pong buffering to
  overlap load, compute and store, and wide 512-bit m_axi access (`max_widen_bitwidth=512`, or packed
  internal types such as `ap_uint<512>` / `hls::vector`) while the top-level argument types stay unchanged.
- For compute-bound kernels, scale the parallel datapath toward the device budget: a small datapath on a
  large device is a performance bug, not a safe choice.

## m_axi bundle assignment (MANDATORY in flash)
- Every top-level **pointer** port needs its own memory bundle: `bundle=gmem0`,
  `bundle=gmem1`, `bundle=gmem2`, … assigned in **kernel argument order**.
- **Never** put multiple pointer ports on `bundle=gmem` or reuse the same `gmemN`
  for two different ports.
- Keep all `s_axilite` ports on `bundle=control` (unchanged from baseline).
- When splitting bundles, preserve legal adapter settings (`latency`, burst lengths,
  outstanding counts) — ports sharing one bundle must match; distinct bundles may
  each carry the same adapter values copied from the baseline.

Rules:
- Preserve algorithm correctness and the single `extern "C"` testbench-visible top and its argument types.
- Label every `for` loop with a descriptive C loop label before the `for`.
- Always provide the COMPLETE kernel in a ```cpp fence that is opened AND closed.
  Restructure as much as the design needs; you are not limited to edits of the existing structure.
  Truncated partial kernels are unusable; if cut off, continue from where you stopped."""

# (old, new) exact replacements in prompt_c2hls.q_optimize_flash; each must match once.
USER_EDITS = [
    ("""3. Unroll small data-parallel loops with a bounded power-of-two factor,
   usually 2, 4, 8, 16, 32, 64, or 128 if the trip count, memory ports,
   and device budget allow it, and match any local-buffer array
   partitioning to the unroll factor.""",
     """3. Unroll data-parallel loops, and add spatial parallelism (rows/columns of
   processing elements), to the factor the device budget supports — DSP,
   BRAM/URAM ports and timing — not a fixed small factor. Match local-buffer
   array partitioning to the unroll factor."""),
    ("""4. Use local scalar or array staging for repeated accesses; partition only
   small hot buffers, not large global-size arrays.""",
     """4. Stage reused data on chip (tiles, line buffers, or whole arrays when they
   fit the device's BRAM/URAM); partition the dimensions that are accessed in
   parallel."""),
    ("""5. Add DATAFLOW/load-compute-store structure only when the added buffering
   is simple and the kernel has clear streaming phases.""",
     """5. Overlap memory traffic with compute: DATAFLOW load-compute-store stages,
   ping-pong (double) buffers or streams when the kernel has separable phases."""),
    ("""6. Improve contiguous AXI access with legal burst/outstanding m_axi pragmas
   and local staging while preserving the narrow host/testbench ABI by
   default. Do NOT introduce `ap_uint<512>`, `MARS_WIDE_BUS_TYPE`, or
   `memcpy_wide_bus_*` helpers unless the benchmark context explicitly says
   the active harness supports a wide-bus ABI.""",
     """6. Make AXI access wide and contiguous: legal burst/outstanding m_axi
   pragmas, `max_widen_bitwidth=512` and 16-float (512-bit) beats. Keep the
   top-level argument types (the host/testbench ABI) unchanged; inside the
   kernel you may pack data into wide types such as `ap_uint<512>` or
   `hls::vector`."""),
    ("""Prefer a compact, synthesizable implementation over a heroic rewrite. For
small PolyBench/HLSFactory-style kernels, simple pipelining, modest unroll,
and a few local buffers often beat complex tiling/dataflow.""",
     """Size the design to the target device. For compute-bound kernels (for example
matrix multiply), the parallel datapath should use most of the device's DSP
budget while every resource stays below 100%; leaving most of the device idle
is not the best endpoint. Restructure the code as much as needed — a full
rewrite is allowed."""),
    ("""- Prefer surgical edits (pragmas, bundles, II, modest unroll, local buffers)
  on the existing structure over inventing a shorter incomplete rewrite.""",
     """- Restructure freely (new tiling, dataflow stages, wide buffers), but keep the
  result complete and synthesizable; a shorter incomplete rewrite is rejected."""),
]

FLOW_ENV = {
    "C2HLS_FLASH_DSP_REDO": "1",
    "C2HLS_FLASH_MIN_DSP": "300",
    "C2HLS_FLASH_MAX_DSP": "9024",
    "C2HLS_FLASH_DSP_FILL_PCT": "90",
    # Flash keeps "must not get slower"; per-step resource ratios are removed (device cap enforced by the
    # resource-cap gate). Other steps are untouched (no "_default" key).
    "C2HLS_STEP_REGRESSION_THRESHOLDS_JSON": json.dumps({"flash": {"latency": 1.0, "resources": {"default": 1e9}}}),
}


def refined_user_prompt(original: str) -> str:
    text = original
    for old, new in USER_EDITS:
        assert text.count(old) == 1, f"flash prompt changed upstream; cannot find: {old[:60]!r}"
        text = text.replace(old, new)
    return text


def _device_budget_block(step_name: str, current_report: dict, part: str, capacity: dict) -> str:
    lines = [f"DEVICE BUDGET for the `{step_name}` step ({part or 'xcu280-fsvh2892-2L-e'}):"]
    cur_lat = current_report.get("latency_ns")
    if cur_lat:
        lines.append(f"  latency_ns: current={float(cur_lat):.0f} — the new design must be faster")
    for key in ("dsp", "bram", "uram", "ff", "lut"):
        cap = (capacity or {}).get(key)
        cur = current_report.get(key)
        if cap:
            cur_s = f"current={int(cur)}  " if cur is not None else ""
            lines.append(f"  {key:<5}: {cur_s}device_capacity={int(cap)} (must stay below 100%)")
    lines.append("Accept rule: latency improves AND every resource stays strictly below 100% of the device. "
                 "There is no growth cap relative to the current design: size the datapath to the device.")
    return "\n".join(lines)


def apply(log=print, version: int = 2, neutral: bool = False) -> dict:
    import prompt_c2hls
    import c2hls

    system, user = REFINED_SYSTEM, refined_user_prompt(prompt_c2hls.q_optimize_flash)
    if version >= 2:
        anchor = "  large device is a performance bug, not a safe choice."
        assert system.count(anchor) == 1
        system = system.replace(anchor, anchor + SYSTEM_BALANCE)
        assert user.count(USER_BALANCE_ANCHOR) == 1
        user = user.replace(USER_BALANCE_ANCHOR, USER_BALANCE)
    if neutral:
        assert version >= 2, "neutral mode is defined on version 2"
        system = _neutralize(system, NEUTRAL_SYSTEM)
        user = _neutralize(user, NEUTRAL_USER)
        for bad in NEUTRAL_BANNED:
            assert bad not in system and bad not in user, f"neutral prompt still says {bad!r}"
    prompt_c2hls.Instruction_c2hls_flash = system
    prompt_c2hls.q_optimize_flash = user

    original_block = c2hls._render_step_resource_constraints

    def refined_block(step_name: str, current_report: dict, part: str = "") -> str:
        if step_name != "flash":
            return original_block(step_name, current_report, part)
        if not current_report:
            return ""
        cap = dict(c2hls._resource_capacity_for(part or os.getenv("C2HLS_PART", "")) or {})
        cap.setdefault("uram", 960)
        block = _device_budget_block(step_name, current_report, part, cap)
        return _neutralize(block, NEUTRAL_BLOCK) if neutral else block

    c2hls._render_step_resource_constraints = refined_block
    if version >= 2:
        original_fill = c2hls.flash_dsp_fill_reject_error

        def fill_or_balance(step_name, report=None):
            err = original_fill(step_name, report)
            if err or step_name != "flash":
                return err
            return balance_error(report or {})

        c2hls.flash_dsp_fill_reject_error = fill_or_balance
        c2hls.hls_flash_dsp_fill_fix = NEUTRAL_FIX if neutral else BALANCE_FIX
    os.environ.update(FLOW_ENV)
    if neutral:  # no DSP floor or fill gate; the 9024 cap and the balance gate stay
        for key in ("C2HLS_FLASH_DSP_REDO", "C2HLS_FLASH_MIN_DSP", "C2HLS_FLASH_DSP_FILL_PCT"):
            os.environ.pop(key, None)
    os.environ["C2HLS_PACKAGED_SKILLS_JSON"] = str(NEUTRAL_PACK if neutral else SKILL_PACKS[version])
    os.environ["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
    info = {"version": version, "system_prompt_chars": len(system), "user_edits": len(USER_EDITS),
            "balance": version >= 2, "balance_ratio": BALANCE_RATIO, "total_macs": _total_macs(),
            "flow_env": FLOW_ENV, "neutral": neutral, "skills": str(NEUTRAL_PACK if neutral else SKILL_PACKS[version])}
    log(f"flash_refined applied: {json.dumps(info)}")
    return info
