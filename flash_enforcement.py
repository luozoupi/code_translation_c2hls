"""Flash-end ping-pong + DATAFLOW enforcement with a csynth overlap judge.

Opt-in via ``--enforcement`` / ``C2HLS_ENFORCEMENT=1``. After flash produces a
synthesizing kernel, a pass requires **all** of:

1. Source: **explicit** ping-pong. Tile loop around DATAFLOW, ``buf[2]`` (or
   ping/pong pair) **and** ``t & 1`` / ``t%2``, INLINE-off load / compute /
   store, B loaded once outside the tile loop (default). ``C2HLS_PP_LOAD_B_IN_DF``
   opts in to ``load_B`` as a tile DATAFLOW task. Arrays declared inside the
   DATAFLOW loop without ``buf[2]`` are **not** ping-pong (101836 wrap).
2. Csynth: one-call / tile-parent latency tracks ``max(load, compute, store)``
   (plus fill), **not** the LCST sum. Interval < latency is the next kernel
   launch, not in-GEMM overlap.
3. Keep flash: LANES=16 / 512-bit load-store and the flash compute nest must
   survive. Kernel latency may not exceed flash × 1.10 (canonical fail:
   4808 → 12642 after a scalar load rewrite).

``#pragma HLS DATAFLOW`` + arrays inside the tile loop is not a pass. The
12893 enforcement kernel is the canonical one-shot fail. Frozen campaigns
are not re-judged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import autosa_flow_gates as _gates

_LOG = logging.getLogger(__name__)

SCHEMA = "flash_overlap_enforcement_v1"
DEFAULT_ROUNDS = 20
ENFORCEMENT_ENV = "C2HLS_ENFORCEMENT"
ENFORCEMENT_ROUNDS_ENV = "C2HLS_ENFORCEMENT_ROUNDS"
AUTOSA_FLOW_ENV = "C2HLS_AUTOSA_FLOW"
SKIP_FLASH_ENV = "C2HLS_SKIP_FLASH"
FLASH_SEED_DIR_ENV = "C2HLS_FLASH_SEED_DIR"
LOADB_IN_DF_ENV = "C2HLS_PP_LOAD_B_IN_DF"
_AUTOSA_SKIP_LOGGED = False

# Kept for log formatting of old reports. Overlap is no longer interval/latency.
_INTERVAL_SLACK = 8
_OVERLAP_RATIO = 0.85

_DATAFLOW_RE = re.compile(r"#\s*pragma\s+HLS\s+DATAFLOW\b", re.IGNORECASE)
_PINGPONG_RE = re.compile(
    r"\b(?:ping|pong|buf_a|buf_b|buf0|buf1)\b"
    r"|\[\s*2\s*\]"
    r"|\[\s*(?:t|tile|i0|ping)\s*&\s*1\s*\]",
    re.IGNORECASE,
)

_JUDGE_SYSTEM = """You are a Vitis HLS 2023.2 **overlap judge**.

Decide whether this kernel is **overlapping ping-pong DATAFLOW**. Decorative
pragmas do not count. Interval < latency does **not** count.

## 1) Code intent
PASS if ALL of:
- A **tile (or iteration) loop around DATAFLOW**, not DATAFLOW wrapping one
  full-matrix `load(); compute(); store();`.
- **Explicit ping-pong**: double buffers `buf[2]` / `[2][...]` (or a ping/pong
  pair) **and** index `t & 1` / `t%2` / ping-pong names. Pattern
  `load(t+1)`, `compute(t)`, `store(t-1)` (or NT>=2 + buf[2] equivalent).
- Separate INLINE-off load / compute / store tasks.
- Load full B **once outside** the tile loop; ping-pong A/C (or A/B/C tiles)
  inside. Reloading full B every tile is a FAIL unless the opt-in note below.
FAIL: no tile loop; DATAFLOW + arrays inside the loop **without** `buf[2]`
(that is not ping-pong; canonical 101836 wrap); one-shot LCST DATAFLOW;
`load_B` inside the tile loop (default).

## 2) Csynth evidence
PASS if one-call (or tile-parent) latency tracks **max(load, compute, store)**
plus a small fill/depth, **not** the sum of those rooms.
FAIL if kernel latency is within ~10% of sum(module latencies) — that is
serial LCST (canonical fail: 12893 ≈ 4170+4552+4169).
FAIL: one big DATAFLOW chunk with no overlapping processes; DATAFLOW with no
tile loop. Do not treat `interval << latency` as overlap (next kernel launch).

## 3) Keep the flash kernel
If a flash baseline is given (LANES=16 loads ~256, compute ~4k, kernel ~4808):
FAIL when the candidate dropped `k0 += LANES` / UNROLL lanes (load_B ~4096).
FAIL when kernel latency > flash × 1.10 (canonical fail: 12642 after 4808).
Overlap of slow scalar rooms is not a pass. Repair by wrapping the flash
load/compute, not by replacing them.

## Output — JSON only (one fenced block)
```json
{
  "schema": "flash_overlap_enforcement_v1",
  "passed": false,
  "code_intended": {"ok": false, "dataflow": false, "ping_pong": false, "reason": "..."},
  "csynth_shows": {"ok": false, "reason": "..."},
  "repair_focus": "one sentence: what to change next"
}
```
Do not return kernel code.
"""

_JUDGE_SYSTEM_LOADB_DF = """
## Opt-in C2HLS_PP_LOAD_B_IN_DF
This run **allows** INLINE-off `load_B` inside the tile DATAFLOW loop.
PASS that shape if A/C still have `buf[2]` + `t & 1`, B_local is declared
inside the DATAFLOW region (channel; avoids HLS 200-976), and load_B keeps
LANES=16. Still FAIL: no buf[2], scalar AXI B walk, one-shot full-matrix
DATAFLOW, systolic mm_pe/FIFOs.
"""

_WRAP_RECIPE_PREFIX = """## How to wrap (NT=2, explicit buf[2] + t & 1, B loaded once outside)
```cpp
#ifndef NT
#define NT 2
#endif
#ifndef TI
#define TI (I / NT)
#endif
static void load_B(/* same ports as flash */) {
#pragma HLS INLINE off
  // PASTE flash load-B body: LANES=16, k0 += LANES, UNROLL u
}
static void load_A_tile(/* A, A_buf[TI][K], t */) {
#pragma HLS INLINE off
  // PASTE flash load-A body; outer trip is TI; inner is still k0 += LANES
}
static void compute_tile(/* A_buf, B_local, C_buf */) {
#pragma HLS INLINE off
  // PASTE flash compute body (dot64 or PE nest); i runs over TI
}
static void store_C_tile(/* C_buf, C, t */) {
#pragma HLS INLINE off
  // PASTE flash store-C body; outer trip is TI; inner j0 += LANES
}
extern "C" void TOP(/* flash ABI */) {
  // keep flash INTERFACE lines
  data_t B_local[J][K];
  data_t A_buf[2][TI][K];
  data_t C_buf[2][TI][J];
  load_B(B, B_local);
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    load_A_tile(A, A_buf[(t + 1) & 1], t + 1);
    compute_tile(A_buf[t & 1], B_local, C_buf[t & 1]);
    store_C_tile(C_buf[(t - 1) & 1], C, t - 1);
  }
}
```
Replace `TOP` with the flash function name. NT=2 large tiles (or 4).
Declaring a single `A_loc` inside the loop is **not** ping-pong.
"""

_WRAP_RECIPE_LOADB_DF = """## How to wrap (NT=2/4, buf[2] + t&1, load_B INSIDE tile DATAFLOW)
`C2HLS_PP_LOAD_B_IN_DF=1`. Do **not** prefix load_B before the tile loop.

```cpp
#ifndef NT
#define NT 4
#endif
#ifndef TI
#define TI (I / NT)
#endif
static void load_B(/* flash LANES=16 body */) {
#pragma HLS INLINE off
}
static void load_A_tile(...) { #pragma HLS INLINE off }
static void compute_tile(...) { #pragma HLS INLINE off }
static void store_C_tile(...) { #pragma HLS INLINE off }
extern "C" void TOP(/* flash ABI */) {
  tile_loop: for (int t = 0; t < NT; ++t) {
#pragma HLS DATAFLOW
    data_t B_local[J][K];
    data_t A_buf[2][TI][K];
    data_t C_buf[2][TI][J];
    load_B(B, B_local);
    load_A_tile(A, A_buf, t & 1, t);
    compute_tile(A_buf, t & 1, B_local, C_buf, t & 1);
    store_C_tile(C_buf, t & 1, C, t);
  }
}
```
B_local must be declared **inside** the DATAFLOW body (channel). Keep LANES=16.
A/C still use buf[2] + t&1. Reloading B every tile is the point: overlap it
with load_A (gmem1 vs gmem0) and compute.
"""

_JUDGE_USER = """Judge ping-pong + DATAFLOW on this flash kernel.

## Kernel
```cpp
{kernel_code}
```

## Csynth (cycles + resources)
{report_blob}

Return the JSON verdict only.
"""

_REPAIR_USER = """The overlap **enforcement judge failed**. This is NOT the flash step.

**Wrap the flash kernel.** Do not replace its load/store or compute.
Flash already has the iso-compute nest and LANES=16 (512-bit) copies. A legal
tile DATAFLOW that drops those and walks one float per cycle is a FAIL
(canonical: flash 4808 / load 259 → enforcement 12642 / load_B 4171).

One-call latency must stay **≤ flash × 1.10** and track max(load, compute,
store), not the LCST sum. `interval < latency` is the next kernel launch.

Keep the exact top-level `extern "C"` name, ports, ranks, and INTERFACE pragmas
(split `bundle=gmem` into gmem0/gmem1/gmem2). Keep csim-correctness.
`#include "kernel.h"` at the top. Keep `max_widen_bitwidth=512`.

## Keep (copy from the flash kernel below — do not drop)
- `const int LANES = 16` and inner `k0 += LANES` / `j0 += LANES` with
  `#pragma HLS UNROLL` on the lane loop. Tile A/C only shrinks the **outer**
  trip (TI rows). load_B stays a full-matrix LANES walk (~256 cycles).
- Flash compute (`dot64` adder tree, or PE×SIMD if that is what flash emitted).
  Paste that body into `compute_tile`; only the i range becomes the tile.
- ARRAY_PARTITION / DSP class of flash. DSP must stay within 15% of flash.

## Do NOT
- Scalar `for (j) for (k) {{ PIPELINE II=1; B_local[j][k] = B[j][k]; }}`
  (that is how 259 became 4171).
- Systolic `mm_pe` / `C_ring` / FIFO packs (stream I/O, a different job).
- DATAFLOW wrapping one full-matrix `load(); compute(); store();`.
- Arrays inside the DATAFLOW tile loop **without** `buf[2]` / `t & 1`
  (not ping-pong; 101836 wrap). {load_b_rule}
- Starting from a new skeleton instead of the flash kernel.
- Returning JSON, comments-only, or truncated code.

{wrap_recipe}

## Keep-flash skills (enforcement only)
{skills_block}

## Judge verdict
```json
{verdict_json}
```

Repair focus: {repair_focus}

## Flash baseline (do not regress)
{flash_report_blob}

## Benchmark context
{benchmark_context}

## Header ({header_name})
```cpp
{header_code}
```

## Flash kernel to wrap (source of truth for load + compute)
```cpp
{flash_kernel_code}
```

## Last attempt (failed — do not iterate on a scalar rewrite)
```cpp
{kernel_code}
```

## Last attempt csynth
{report_blob}

Return one complete ```cpp kernel.
"""


def _skills_from_json(path: Path) -> list[Any]:
    from skill_library import _coerce_skill_entry

    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    skills = []
    for entry in data.get("skills") or []:
        skill = _coerce_skill_entry(entry)
        if skill is not None:
            skills.append(skill)
    return skills


def load_enforcement_keep_flash_skills() -> tuple[str, list[str]]:
    from c2hls_paths import (
        FLASH_ENFORCEMENT_KEEP_FLASH_SKILL_ENTRIES_JSON,
        FLASH_ENFORCEMENT_LOADB_IN_DF_SKILL_ENTRIES_JSON,
    )
    from skill_library import render_skill_set_for_prompt_full

    skills = _skills_from_json(FLASH_ENFORCEMENT_KEEP_FLASH_SKILL_ENTRIES_JSON)
    if _gates.load_b_in_dataflow_allowed():
        skills.extend(_skills_from_json(FLASH_ENFORCEMENT_LOADB_IN_DF_SKILL_ENTRIES_JSON))
    if not skills:
        return "", []
    return render_skill_set_for_prompt_full(skills), [sk.id for sk in skills]


def judge_system_prompt() -> str:
    if _gates.load_b_in_dataflow_allowed():
        return _JUDGE_SYSTEM + _JUDGE_SYSTEM_LOADB_DF
    return _JUDGE_SYSTEM


def load_b_repair_rule() -> str:
    if _gates.load_b_in_dataflow_allowed():
        return (
            "Prefix load_B before the tile loop is a FAIL for this overlay; "
            "put LANES=16 load_B inside the tile DATAFLOW with B_local declared "
            "in that region."
        )
    return "Reloading full B every tile is a FAIL."


def wrap_recipe_for_prompt() -> str:
    if _gates.load_b_in_dataflow_allowed():
        return _WRAP_RECIPE_LOADB_DF
    return _WRAP_RECIPE_PREFIX


def build_repair_prompt(
    *,
    verdict_json: str,
    repair_focus: str,
    benchmark_context: str,
    header_name: str,
    header_code: str,
    kernel_code: str,
    report_blob: str,
    flash_kernel_code: str = "",
    flash_report_blob: str = "",
    skills_block: str = "",
) -> str:
    block = skills_block
    if not block:
        block, _ids = load_enforcement_keep_flash_skills()
    wrap = flash_kernel_code or kernel_code
    flash_blob = flash_report_blob or report_blob
    return _REPAIR_USER.format(
        skills_block=block or "(keep-flash skill file missing)",
        wrap_recipe=wrap_recipe_for_prompt(),
        load_b_rule=load_b_repair_rule(),
        verdict_json=verdict_json,
        repair_focus=repair_focus,
        flash_report_blob=flash_blob,
        benchmark_context=benchmark_context,
        header_name=header_name,
        header_code=header_code,
        flash_kernel_code=wrap,
        kernel_code=kernel_code,
        report_blob=report_blob,
    )


@dataclass
class JudgeVerdict:
    passed: bool
    code_intended: bool
    csynth_shows: bool
    reason: str = ""
    source: str = "static"
    repair_focus: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "passed": self.passed,
            "code_intended": self.code_intended,
            "csynth_shows": self.csynth_shows,
            "reason": self.reason,
            "source": self.source,
            "repair_focus": self.repair_focus,
            "details": self.details,
        }


def _env_flag(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enforcement_enabled() -> bool:
    global _AUTOSA_SKIP_LOGGED
    if _env_flag(AUTOSA_FLOW_ENV):
        if not _AUTOSA_SKIP_LOGGED:
            _LOG.info(
                "[enforcement] skipped: C2HLS_AUTOSA_FLOW is on; "
                "compute+I/O chain owns overlap (ping-pong enforcement off)"
            )
            _AUTOSA_SKIP_LOGGED = True
        return False
    return _env_flag(ENFORCEMENT_ENV)


def enforcement_round_limit() -> int:
    raw = os.getenv(ENFORCEMENT_ROUNDS_ENV, "").strip()
    if not raw:
        return DEFAULT_ROUNDS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_ROUNDS


def add_enforcement_arguments(parser: Any) -> None:
    parser.add_argument(
        "--enforcement",
        action="store_true",
        help="After flash, require ping-pong + DATAFLOW (LLM judge on code and csynth).",
    )
    parser.add_argument(
        "--enforcement-rounds",
        "--enforcement_rounds",
        dest="enforcement_rounds",
        type=int,
        default=None,
        help="Max ping-pong/DATAFLOW repair-or-generate rounds (default: 20).",
    )


def apply_enforcement_env(args: Any) -> None:
    if getattr(args, "enforcement", False):
        os.environ[ENFORCEMENT_ENV] = "1"
    rounds = getattr(args, "enforcement_rounds", None)
    if rounds is not None:
        os.environ[ENFORCEMENT_ROUNDS_ENV] = str(int(rounds))
    elif getattr(args, "enforcement", False) and not os.getenv(ENFORCEMENT_ROUNDS_ENV, "").strip():
        os.environ[ENFORCEMENT_ROUNDS_ENV] = str(DEFAULT_ROUNDS)


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def apply_enforcement_from_campaign(campaign: Optional[dict[str, Any]]) -> None:
    """Honor campaign.json enforcement even when Slurm dropped the env var."""
    if not isinstance(campaign, dict):
        return
    if _truthy_flag(campaign.get("enforcement")):
        os.environ[ENFORCEMENT_ENV] = "1"
    rounds = campaign.get("enforcement_rounds")
    if rounds is not None and str(rounds).strip():
        try:
            os.environ[ENFORCEMENT_ROUNDS_ENV] = str(max(1, int(rounds)))
        except (TypeError, ValueError):
            pass
    if _truthy_flag(campaign.get("pp_load_b_in_dataflow")):
        os.environ[LOADB_IN_DF_ENV] = "1"
    apply_skip_flash_from_campaign(campaign)


def skip_flash_enabled() -> bool:
    return _env_flag(SKIP_FLASH_ENV)


def flash_seed_dir() -> Optional[Path]:
    raw = (os.getenv(FLASH_SEED_DIR_ENV) or "").strip()
    if not raw:
        return None
    return Path(raw)


def load_flash_seed(bench: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Load ``{bench}_flash_opt.cpp`` + report. Never fall back to selected.

    mmflow ``selected`` is the stream kernel (4292/320), not flash 139484/10.
    """
    root = flash_seed_dir()
    if root is None or not root.is_dir():
        return None
    name = (bench or "").strip() or "autosa_mm"
    cpp = root / f"{name}_flash_opt.cpp"
    report_path = root / f"{name}_flash_opt_report.json"
    if not cpp.is_file() or not report_path.is_file():
        _LOG.warning(
            "[flash] skip-flash seed missing %s or %s", cpp, report_path
        )
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[flash] skip-flash report unreadable: %s", exc)
        return None
    if not isinstance(report, dict):
        return None
    code = cpp.read_text(encoding="utf-8")
    if not code.strip():
        return None
    return code, report


def persist_flash_seed(
    cell_dir: Any,
    bench: str,
    code: str,
    report: dict[str, Any],
) -> None:
    dest = Path(str(cell_dir))
    dest.mkdir(parents=True, exist_ok=True)
    name = (bench or "").strip() or "autosa_mm"
    (dest / f"{name}_flash_opt.cpp").write_text(code, encoding="utf-8")
    (dest / f"{name}_flash_opt_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (dest / f"{name}_flash_seed.cpp").write_text(code, encoding="utf-8")
    (dest / f"{name}_flash_seed_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )


def apply_flash_seed_to_orch(
    orch: Any,
    bench: str,
    cell_dir: Any = None,
) -> bool:
    """Install seeded flash as the accepted kernel. No LLM, no re-synth."""
    if not skip_flash_enabled():
        return False
    loaded = load_flash_seed(bench)
    if loaded is None:
        return False
    code, report = loaded
    orch.hls_code = code
    orch.synth_report = report
    ctx = dict(getattr(orch, "_pipelined_ctx", None) or {})
    ctx["flash_pending_code"] = code
    ctx["flash_step_result"] = {
        "success": True,
        "step_name": "flash",
        "report": report,
        "code": code,
        "seeded": True,
        "skip_flash": True,
    }
    ctx["flash_done"] = True
    orch._pipelined_ctx = ctx
    dest = cell_dir or getattr(orch, "_artifact_output_dir", None)
    if dest:
        persist_flash_seed(dest, bench, code, report)
    _LOG.info(
        "[flash] skip-flash seed accepted latency=%s dsp=%s from %s",
        report.get("latency_cycles"),
        report.get("dsp"),
        flash_seed_dir(),
    )
    return True


def apply_skip_flash_from_campaign(campaign: Optional[dict[str, Any]]) -> None:
    if not isinstance(campaign, dict):
        return
    if _truthy_flag(campaign.get("skip_flash")):
        os.environ[SKIP_FLASH_ENV] = "1"
        os.environ["C2HLS_SKIP_PHASE_B"] = "1"
    seed = campaign.get("flash_seed_dir")
    if isinstance(seed, str) and seed.strip():
        os.environ[FLASH_SEED_DIR_ENV] = seed.strip()


def apply_enforcement_from_campaign_root() -> None:
    root = (os.getenv("BATCH_PARALLEL_CAMPAIGN_ROOT") or "").strip()
    if not root:
        return
    path = os.path.join(root, "campaign.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            campaign = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    apply_enforcement_from_campaign(campaign)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def code_intent(code: str) -> dict[str, Any]:
    """Tile loop + arrays inside DATAFLOW. Pragma + buf[2] is not enough."""
    return _gates.pingpong_dataflow_code_intent(code)


def csynth_overlap(
    report: Optional[dict[str, Any]],
    *,
    tile_loop: bool = False,
    code: Optional[str] = None,
) -> dict[str, Any]:
    """Overlap iff csynth latency tracks max(rooms), not the LCST sum."""
    enriched = _gates.enrich_report_with_csynth_modules(report)
    verdict = _gates.pingpong_dataflow_csynth_ok(
        enriched, tile_loop=tile_loop, code=code
    )
    latency = _as_int(enriched.get("latency_cycles"))
    interval = _as_int(enriched.get("interval"))
    return {
        "ok": verdict.ok,
        "reason": verdict.reason,
        "latency_cycles": latency,
        "interval": interval,
        "dsp": enriched.get("dsp"),
        "bram": enriched.get("bram"),
        "ff": enriched.get("ff"),
        "lut": enriched.get("lut"),
        "modules": enriched.get("modules"),
        "dataflow_regions": enriched.get("dataflow_regions"),
        "pipeline_type": enriched.get("pipeline_type"),
    }


def static_verdict(
    code: str,
    report: Optional[dict[str, Any]],
    *,
    baseline_code: Optional[str] = None,
    baseline_report: Optional[dict[str, Any]] = None,
) -> JudgeVerdict:
    intent = code_intent(code)
    shown = csynth_overlap(
        report, tile_loop=bool(intent.get("tile_loop")), code=code
    )
    keep = _gates.enforcement_keep_flash_ok(
        code=code,
        report=report,
        baseline_code=baseline_code,
        baseline_report=baseline_report,
    )
    reasons = []
    if not intent.get("ok"):
        reasons.append(str(intent.get("reason") or "illegal ping-pong DATAFLOW source"))
    if not keep.ok:
        reasons.append(keep.reason)
    if not shown["ok"]:
        reasons.append(shown.get("reason") or "csynth does not show overlapping rooms")
    overlap_ok = bool(shown["ok"] and keep.ok)
    passed = bool(intent.get("ok") and overlap_ok)
    reason = "; ".join(reasons) or (
        shown.get("reason")
        or intent.get("reason")
        or keep.reason
        or "tile-loop ping-pong DATAFLOW; csynth ≈ max(rooms); flash structure kept"
    )
    return JudgeVerdict(
        passed=passed,
        code_intended=bool(intent.get("ok")),
        csynth_shows=overlap_ok,
        reason=reason,
        source="static",
        repair_focus=reasons[0] if reasons else "",
        details={
            "code_intent": intent,
            "csynth_overlap": shown,
            "keep_flash": {"ok": keep.ok, "reason": keep.reason},
        },
    )


def format_judge_report(report: Optional[dict[str, Any]]) -> str:
    shown = csynth_overlap(report)
    report = report or {}
    lines = [
        f"- latency_cycles: {shown['latency_cycles']}",
        f"- interval: {shown['interval']}",
        f"- BRAM: {report.get('bram')}  DSP: {report.get('dsp')}  "
        f"FF: {report.get('ff')}  LUT: {report.get('lut')}",
        f"- fmax_mhz: {report.get('fmax_mhz')}",
        f"- overlap_reason: {shown.get('reason')}",
    ]
    mods = shown.get("modules") or report.get("modules") or []
    if mods:
        blob = ", ".join(
            f"{m.get('name')}={m.get('latency_cycles')}" for m in mods[:8]
        )
        lines.append(f"- modules: {blob}")
    if shown["ok"]:
        lines.append("- overlap_hint: latency tracks max(load,compute,store), not the sum")
    else:
        lines.append(
            "- overlap_hint: need tile-loop DATAFLOW whose csynth latency is "
            "≈ max(rooms). interval<latency is not overlap."
        )
    return "\n".join(lines)


def _nested_ok(raw: Any, key: str) -> Optional[bool]:
    if not isinstance(raw, dict):
        return None
    nested = raw.get(key)
    if isinstance(nested, dict) and "ok" in nested:
        return bool(nested.get("ok"))
    if isinstance(nested, bool):
        return nested
    return None


def parse_llm_verdict(text: str) -> JudgeVerdict:
    data = _extract_judge_json(text)
    if not data:
        return JudgeVerdict(
            passed=False,
            code_intended=False,
            csynth_shows=False,
            reason="LLM judge reply missing JSON verdict",
            source="llm_parse_error",
            repair_focus="return a flash_overlap_enforcement_v1 JSON verdict",
        )
    code_ok = _nested_ok(data, "code_intended")
    synth_ok = _nested_ok(data, "csynth_shows")
    if code_ok is None:
        code_ok = False
    if synth_ok is None:
        synth_ok = False
    code_blob = data.get("code_intended") if isinstance(data.get("code_intended"), dict) else {}
    synth_blob = data.get("csynth_shows") if isinstance(data.get("csynth_shows"), dict) else {}
    reasons = [
        str(code_blob.get("reason") or "").strip(),
        str(synth_blob.get("reason") or "").strip(),
    ]
    reason = "; ".join(r for r in reasons if r) or str(data.get("repair_focus") or "")
    repair = str(data.get("repair_focus") or synth_blob.get("reason") or reason).strip()
    passed = bool(code_ok and synth_ok)
    if data.get("passed") is False:
        passed = False
    return JudgeVerdict(
        passed=passed,
        code_intended=bool(code_ok),
        csynth_shows=bool(synth_ok),
        reason=reason,
        source="llm",
        repair_focus=repair,
        details=data,
    )


def _extract_judge_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    for raw in blocks:
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("schema") == SCHEMA or "code_intended" in data or "csynth_shows" in data:
            return data
    return {}


def combine_verdicts(static: JudgeVerdict, llm: Optional[JudgeVerdict]) -> JudgeVerdict:
    """Static measured overlap is the pass gate. LLM cannot veto it.

    Syntax-only DATAFLOW + buf[2] is not a pass. Serial csynth (kernel ≈
    sum of rooms) vetoes an LLM 'pass'. LLM is only used for repair_focus
    when static has neither overlap nor legal structure.
    """
    if static.passed:
        return JudgeVerdict(
            passed=True,
            code_intended=True,
            csynth_shows=True,
            reason=static.reason or "tile-loop ping-pong DATAFLOW; csynth ≈ max(rooms)",
            source="static" if llm is None else "hybrid",
            repair_focus="",
            details={"static": static.to_dict(), "llm": None if llm is None else llm.to_dict()},
        )
    if static.code_intended:
        return JudgeVerdict(
            passed=False,
            code_intended=True,
            csynth_shows=bool(static.csynth_shows),
            reason=static.reason or "legal tile-loop DATAFLOW but csynth is still the LCST sum",
            source="static" if llm is None else "hybrid",
            repair_focus=static.repair_focus or (
                "csynth latency is the serial LCST sum, not max(load,compute,store). "
                "Keep the tile loop; keep flash LANES=16 loads and flash compute; "
                "do not treat interval<latency as overlap."
            ),
            details={"static": static.to_dict(), "llm": None if llm is None else llm.to_dict()},
        )
    if llm is None:
        return static
    repair = llm.repair_focus or static.repair_focus
    if llm.source == "llm_parse_error":
        repair = static.repair_focus or repair
    reasons = [r for r in (static.reason, llm.reason) if r]
    return JudgeVerdict(
        passed=False,
        code_intended=bool(static.code_intended and llm.code_intended),
        csynth_shows=bool(static.csynth_shows and llm.csynth_shows),
        reason="; ".join(reasons),
        source="hybrid",
        repair_focus=repair,
        details={"static": static.to_dict(), "llm": llm.to_dict()},
    )


def _mark_structure_stop(summary: dict[str, Any], verdict: JudgeVerdict) -> None:
    """Commit only when csynth shows overlapping rooms, not syntax."""
    overlap = bool(verdict.csynth_shows) and bool(verdict.passed)
    summary["passed"] = bool(verdict.passed)
    summary["applied"] = bool(verdict.passed)
    summary["code_intended"] = bool(verdict.code_intended)
    summary["overlap"] = overlap
    summary["needs_latency_opt"] = False
    if verdict.passed:
        summary["reason"] = verdict.reason or "tile-loop ping-pong DATAFLOW; csynth ≈ max(rooms)"
    else:
        summary["reason"] = verdict.reason or "ping-pong DATAFLOW overlap not shown on csynth"


def run_enforcement_loop(
    *,
    kernel_code: str,
    synth_report: Optional[dict[str, Any]],
    rounds: int,
    judge_fn: Callable[[str, Optional[dict[str, Any]]], JudgeVerdict],
    generate_fn: Callable[[str, Optional[dict[str, Any]], JudgeVerdict, int], str],
    evaluate_fn: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    current_code = kernel_code or ""
    current_report = synth_report or {}
    attempts: list[dict[str, Any]] = []
    verdict = judge_fn(current_code, current_report)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "attempted": True,
        "passed": bool(verdict.passed),
        "code_intended": bool(verdict.code_intended),
        "overlap": bool(verdict.csynth_shows),
        "needs_latency_opt": False,
        "rounds_used": 0,
        "rounds_limit": int(rounds),
        "initial": verdict.to_dict(),
        "attempts": attempts,
        "code": current_code,
        "report": current_report,
    }
    if verdict.passed:
        _mark_structure_stop(summary, verdict)
        return summary

    for round_i in range(max(1, int(rounds))):
        proposed = generate_fn(current_code, current_report, verdict, round_i) or ""
        attempt: dict[str, Any] = {"round": round_i, "judge_before": verdict.to_dict()}
        if not proposed.strip():
            attempt["status"] = "no_code"
            attempts.append(attempt)
            continue
        evaluated = evaluate_fn(proposed) or {}
        if not evaluated.get("success"):
            attempt["status"] = "eval_failed"
            attempt["error"] = evaluated.get("error", "")
            attempts.append(attempt)
            continue
        current_code = evaluated.get("code") or proposed
        current_report = evaluated.get("report") or current_report
        verdict = judge_fn(current_code, current_report)
        if verdict.passed:
            attempt["status"] = "accepted"
        else:
            attempt["status"] = "judge_fail"
        attempt["judge_after"] = verdict.to_dict()
        attempt["latency_cycles"] = (current_report or {}).get("latency_cycles")
        attempt["interval"] = (current_report or {}).get("interval")
        attempts.append(attempt)
        summary["rounds_used"] = round_i + 1
        summary["code"] = current_code
        summary["report"] = current_report
        if verdict.passed:
            _mark_structure_stop(summary, verdict)
            return summary

    summary["passed"] = False
    summary["applied"] = False
    summary["code_intended"] = bool(verdict.code_intended)
    summary["overlap"] = bool(verdict.csynth_shows)
    summary["needs_latency_opt"] = False
    summary["rounds_used"] = len(
        [a for a in attempts if a.get("status") in {
            "accepted", "structure_ok", "judge_fail", "eval_failed", "no_code",
        }]
    )
    if summary["rounds_used"] == 0:
        summary["rounds_used"] = int(rounds)
    summary["final"] = verdict.to_dict()
    return summary


def _call_judge_llm(orch: Any, code: str, report: Optional[dict[str, Any]]) -> str:
    user = _JUDGE_USER.format(
        kernel_code=(code or "")[:120000],
        report_blob=format_judge_report(report),
    )
    messages = [
        {"role": "system", "content": judge_system_prompt()},
        {"role": "user", "content": user},
    ]
    call = getattr(orch, "_call_llm", None)
    if call is None:
        return ""
    return call(messages) or ""


def _generate_repair(orch: Any, code: str, report: Optional[dict[str, Any]],
                     verdict: JudgeVerdict, round_i: int) -> str:
    flash_code = getattr(orch, "enforcement_flash_code", None) or code
    flash_report = getattr(orch, "enforcement_flash_report", None) or report
    prompt = build_repair_prompt(
        verdict_json=json.dumps(verdict.to_dict(), indent=2),
        repair_focus=verdict.repair_focus or verdict.reason,
        benchmark_context=getattr(orch, "benchmark_context", "") or "",
        header_name=getattr(orch, "header_name", None) or "kernel.h",
        header_code=(getattr(orch, "header_code", "") or "")[:12000],
        kernel_code=(code or "")[:120000],
        report_blob=format_judge_report(report),
        flash_kernel_code=(flash_code or "")[:120000],
        flash_report_blob=format_judge_report(flash_report),
    )
    request = getattr(orch, "_request_code_revision", None)
    if request is None:
        return ""
    _LOG.info("[enforcement] generate/repair round %d", round_i)
    return request(prompt) or ""


def _evaluate_on_orch(orch: Any, code: str) -> dict[str, Any]:
    evaluate = getattr(orch, "_evaluate_candidate_with_repairs", None)
    if evaluate is None:
        return {"success": False, "error": "orchestrator cannot evaluate candidates"}
    candidate = evaluate(code, "[Enforcement]")
    if not candidate.get("success"):
        return candidate
    csim = candidate.get("csim")
    prior = getattr(orch, "generated_csim", None)
    if isinstance(prior, dict) and prior.get("passed") is True:
        if isinstance(csim, dict) and csim.get("passed") is False:
            candidate = dict(candidate)
            candidate["success"] = False
            candidate["error"] = "enforcement candidate failed csim"
    return candidate


def attach_enforcement_after_flash(orch: Any) -> Optional[dict[str, Any]]:
    """Run enforcement after a successful flash synth and stash the summary on ctx.

    Idempotent. This is the single entry point every flash success path must call
    (AutoSA/tier_a ``_run_synth_flash``, pipelined handle_job, and ``_finalize_success``).
    """
    ctx = getattr(orch, "_pipelined_ctx", None)
    if not isinstance(ctx, dict):
        ctx = {}
        orch._pipelined_ctx = ctx
    if ctx.get("enforcement_ran"):
        return ctx.get("enforcement")
    ctx["enforcement_ran"] = True
    orch._pipelined_ctx = ctx
    try:
        enf = maybe_run_enforcement(orch)
    except Exception as exc:
        _LOG.warning("[enforcement] attach failed: %s", exc)
        enf = {"attempted": True, "passed": False, "error": str(exc)}
    if enf is None:
        _LOG.warning(
            "[enforcement] attach: disabled for %s",
            getattr(orch, "benchmark_name", ""),
        )
        return None
    slim = {k: v for k, v in enf.items() if k != "code"}
    ctx["enforcement"] = slim
    if enf.get("applied"):
        step = dict(ctx.get("flash_step_result") or {})
        step["code"] = getattr(orch, "hls_code", None)
        step["report"] = getattr(orch, "synth_report", None)
        step["enforcement"] = slim
        ctx["flash_step_result"] = step
    orch._pipelined_ctx = ctx
    return enf


def maybe_run_enforcement(orch: Any) -> Optional[dict[str, Any]]:
    """Run ping-pong/DATAFLOW enforcement on a finished flash orchestrator.

    No-op when ``C2HLS_ENFORCEMENT`` is off. Commits the new kernel onto
    ``orch`` only when the judge passes.
    """
    existing = getattr(orch, "enforcement_result", None)
    if isinstance(existing, dict) and existing.get("attempted") is not None:
        return existing
    apply_enforcement_from_campaign_root()
    if not enforcement_enabled():
        _LOG.warning(
            "[enforcement] skipped: C2HLS_ENFORCEMENT=%s campaign_root=%s",
            os.getenv(ENFORCEMENT_ENV, ""),
            os.getenv("BATCH_PARALLEL_CAMPAIGN_ROOT", ""),
        )
        return None
    _LOG.info("[enforcement] starting rounds=%s", enforcement_round_limit())
    code = getattr(orch, "hls_code", None) or ""
    report = getattr(orch, "synth_report", None)
    orch.enforcement_flash_code = code
    orch.enforcement_flash_report = report if isinstance(report, dict) else {}
    if not code or not report:
        summary = {
            "attempted": False,
            "passed": False,
            "reason": "missing kernel or synth report",
        }
        orch.enforcement_result = summary
        return summary

    def judge_fn(cur_code: str, cur_report: Optional[dict[str, Any]]) -> JudgeVerdict:
        static = static_verdict(
            cur_code,
            cur_report,
            baseline_code=orch.enforcement_flash_code,
            baseline_report=orch.enforcement_flash_report,
        )
        if static.passed:
            return static
        reply = _call_judge_llm(orch, cur_code, cur_report)
        llm = parse_llm_verdict(reply)
        return combine_verdicts(static, llm)

    summary = run_enforcement_loop(
        kernel_code=code,
        synth_report=report if isinstance(report, dict) else {},
        rounds=enforcement_round_limit(),
        judge_fn=judge_fn,
        generate_fn=lambda c, r, v, i: _generate_repair(orch, c, r, v, i),
        evaluate_fn=lambda c: _evaluate_on_orch(orch, c),
    )
    if summary.get("passed") and summary.get("code"):
        orch.hls_code = summary["code"]
        if summary.get("report"):
            orch.synth_report = summary["report"]
        summary["applied"] = True
        if summary.get("needs_latency_opt"):
            _LOG.info(
                "[enforcement] passed after %s round(s) with overlap; "
                "interval=%s latency=%s",
                summary.get("rounds_used"),
                (summary.get("report") or {}).get("interval"),
                (summary.get("report") or {}).get("latency_cycles"),
            )
        else:
            _LOG.info("[enforcement] passed after %s round(s)", summary.get("rounds_used"))
    else:
        summary["applied"] = False
        _LOG.warning(
            "[enforcement] not passed after %s/%s rounds",
            summary.get("rounds_used"),
            summary.get("rounds_limit"),
        )
    orch.enforcement_result = summary
    _persist_enforcement(orch, summary)
    return summary


def _persist_enforcement(orch: Any, summary: dict[str, Any]) -> None:
    out_dir = getattr(orch, "_artifact_output_dir", None)
    bench = getattr(orch, "benchmark_name", "") or "kernel"
    if not out_dir:
        return
    try:
        path = os.path.join(str(out_dir), f"{bench}_enforcement.json")
        slim = dict(summary)
        slim.pop("code", None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(slim, handle, indent=2, default=str)
    except OSError as exc:
        _LOG.warning("[enforcement] failed to write result json: %s", exc)
