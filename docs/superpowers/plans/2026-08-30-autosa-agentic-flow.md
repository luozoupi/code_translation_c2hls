# AutoSA-ordered agentic mm flow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `autosa_mm` as flash → **compute architecture** → **I/O architecture**, with gates that match AutoSA’s paper (workers first, then DRAM overlapping compute), not DATAFLOW-enforcement-before-workers.

**Architecture:** Keep the two existing LLM steps (`post_flash_dse.py` = compute, `post_flash_stream.py` = I/O). Do **not** add eight LLM hops. Add a small pure `autosa_flow_gates.py` so pass/fail is “many workers / DSP match / one-run time ≈ slowest room,” not “pragma present.” Skip flash-end ping-pong enforcement when this flow is on. Knobs stay in `post_flash_pe_recipe.py` (16×4, 128-bit for this mm). Flattening stays a stream **repair skill**, not a stage.

**Tech Stack:** Python 3, existing pipelined flash chain, pytest, Vitis HLS 2023.2, U280 3.33 ns, DeepSeek-v4-flash.

**Rank1 (do not change):** 4228 cycles, 320 DSP, `xcu280-fsvh2892-2L-e`, 3.33 ns. Success: agent csynth **latency** ≤ 4228 × 1.02, original `autosa_mm` ABI, csim+csynth. Do not emit AutoSA `kernel0`.

---

## Supervisor view (what the flow is)

AutoSA is two jobs, then knobs:

1. **Many small calculators** that do not wait on one running total (compute).
2. **Memory filling the next chapter while they work**, with wide parcels and A private / B passed along the line / C drained (I/O).

The agent does **those two rewrites**. It does not search PE×SIMD (recipe is fixed). It does **not** start with “two folders + DATAFLOW” on a one-worker kernel — that is why the last mm-only job finished at load+compute+store.

**Do not build:** eight sequential LLM stages named after every paper subsection.

---

## File map

| Path | Role |
|------|------|
| `autosa_flow_gates.py` | Pure gates: compute pass, I/O overlap pass, human `reason` strings |
| `tests/test_autosa_flow_gates.py` | No Vitis: enforcement kernel vs rank1-like reports |
| `post_flash_dse.py` | Compute step: still DSP recipe; **do not fail** because kernel latency is still ~3× |
| `post_flash_stream.py` | After architecture_ok, also require overlap gate (latency ≈ interval) |
| `flash_enforcement.py` | Skip `attach_enforcement_after_flash` when `C2HLS_AUTOSA_FLOW=1` |
| `scripts/pc2/flash_pipelined_bench.py` | Already chains dse then stream; no order change if enforcement skipped |
| `scripts/pc2/batch_parallel_autosa_mm_flow.json` | mm-only campaign: enforcement off, dse+stream chain on |
| `scripts/pc2/start_autosa_mm_flow.sh` | Launcher (copy mm-enforcement, different config/env) |
| `docs/pc2/2026-08-30-autosa-agentic-flow-one-pager.md` | One page for the supervisor |

---

## Locked behavior

| Step | LLM job | Pass when | Fail when (keep repairing) |
|------|---------|-----------|------------------------------|
| Flash | Legal kernel, ABI, csim | csim+csynth | Wrong ports / csim fail |
| Compute (`post_flash_dse`) | N workers, short adder, small SIMD, on-chip tiles | DSP ≥ recipe `min_dsp` (320-class); csim | One fat 64-wide tree only; 6 DSP; clone `kernel0` |
| I/O (`post_flash_stream`) | A per worker, B along the line, C drain, packed beats, chapter handshake | DSP still recipe; streams+pack+flatten skills; **`latency / interval ≤ 1.15`** (one run ≈ slowest room) | DATAFLOW on bulk locals; interval ≪ latency with latency still load+compute+store |

Compute **success example (keep):** Aug 18 multi-PE ~13160 kernel, compute ~4816, DSP 352. That is **done with compute**. I/O is the next step.

I/O **success example:** 4285 vs 4228. Overlap gate would pass (4285 vs interval ~4173).

**Enforcement kernel (do not treat as success):** 12893 latency / 4553 interval. Overlap gate **fails**.

---

### Task 1: Pure gates (TDD)

**Files:**
- Create: `autosa_flow_gates.py`
- Create: `tests/test_autosa_flow_gates.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_autosa_flow_gates.py
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autosa_flow_gates as g


def test_compute_pass_dsp_even_if_kernel_still_serial_io():
    report = {"dsp": 352, "latency_cycles": 13160, "interval": 13161}
    v = g.compute_architecture_ok(report, min_dsp=200)
    assert v.ok is True
    assert "workers" in v.reason.lower() or "dsp" in v.reason.lower()


def test_compute_fail_flash_leftover_dsp():
    report = {"dsp": 6, "latency_cycles": 149082, "interval": 149083}
    v = g.compute_architecture_ok(report, min_dsp=200)
    assert v.ok is False


def test_overlap_fail_enforcement_lcst():
    # 12893 vs 4553: second launch could start, this C is not done
    report = {"dsp": 320, "latency_cycles": 12893, "interval": 4553}
    v = g.io_overlap_ok(report, max_latency_over_interval=1.15)
    assert v.ok is False
    assert "overlap" in v.reason.lower() or "chapter" in v.reason.lower() or "interval" in v.reason.lower()


def test_overlap_pass_rank1_like():
    report = {"dsp": 320, "latency_cycles": 4228, "interval": 4133}
    v = g.io_overlap_ok(report, max_latency_over_interval=1.15)
    assert v.ok is True


def test_overlap_pass_locked_stream():
    report = {"dsp": 320, "latency_cycles": 4285, "interval": 4173}
    v = g.io_overlap_ok(report, max_latency_over_interval=1.15)
    assert v.ok is True


def test_rank1_success_gate():
    assert g.within_rank1(4285, rank1=4228, tol=1.02) is True
    assert g.within_rank1(4553, rank1=4228, tol=1.02) is False
```

- [ ] **Step 2: Run tests, expect import fail**

Run: `python -m pytest tests/test_autosa_flow_gates.py -v`

Expected: `ModuleNotFoundError: autosa_flow_gates`

- [ ] **Step 3: Implement gates**

```python
# autosa_flow_gates.py
"""Behavioral gates for AutoSA-ordered flash → compute → I/O.

No Vitis. Reports are csynth summaries: latency_cycles, interval, dsp.
"""
from __future__ import annotations

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


def within_rank1(agent_cycles: float, *, rank1: int, tol: float = 1.02) -> bool:
    return agent_cycles <= rank1 * tol
```

- [ ] **Step 4: Re-run tests**

Run: `python -m pytest tests/test_autosa_flow_gates.py -v`

Expected: 6 passed

- [ ] **Step 5: Commit** (only if the user asked to commit)

```bash
git add autosa_flow_gates.py tests/test_autosa_flow_gates.py
git commit -m "$(cat <<'EOF'
Add AutoSA-flow compute vs I/O overlap gates.

EOF
)"
```

---

### Task 2: Skip flash-end DATAFLOW enforcement on this flow

**Files:**
- Modify: `flash_enforcement.py` (`maybe_run_enforcement` / `attach_enforcement_after_flash`)
- Modify: `tests/test_flash_enforcement.py`

Why: `attach_enforcement_after_flash` currently rewrites a good flash into LCST DATAFLOW **before** compute. That is paper §6 on a one-worker design.

- [ ] **Step 1: Test skip**

```python
def test_enforcement_skipped_when_autosa_flow(monkeypatch):
    monkeypatch.setenv("C2HLS_AUTOSA_FLOW", "1")
    monkeypatch.setenv("C2HLS_ENFORCEMENT", "1")
    assert fe.enforcement_enabled() is False
```

- [ ] **Step 2: Implement**

In `enforcement_enabled()`, if `_env_flag("C2HLS_AUTOSA_FLOW")` is true, return False (log once: skip ping-pong enforcement; compute+I/O chain owns overlap).

Keep `C2HLS_ENFORCEMENT=1` working when `C2HLS_AUTOSA_FLOW` is unset (old mm-enforcement launcher).

- [ ] **Step 3: Run** `python -m pytest tests/test_flash_enforcement.py tests/test_autosa_flow_gates.py -v`

---

### Task 3: Stream step requires overlap, not only streams+DSP

**Files:**
- Modify: `post_flash_stream.py` `architecture_ok` / `architecture_miss_message`
- Modify: `tests/test_post_flash_stream.py`

- [ ] **Step 1: Failing test** — existing packed+DSP+II=1 report with `latency=12893, interval=4553` must make `architecture_ok` False.

- [ ] **Step 2: Call `autosa_flow_gates.io_overlap_ok` at the end of `architecture_ok`.** If fail, `architecture_miss_message` must say the one-run vs room-time numbers (no “pragma missing”).

- [ ] **Step 3: Repair prompt** — one extra bullet: handshake **per chapter** (tile or stream beat). Do not PIPO the whole matrix after the whole loop. Do not mention “DSE.”

- [ ] **Step 4: Run** `python -m pytest tests/test_post_flash_stream.py tests/test_autosa_flow_gates.py -v`

---

### Task 4: Compute step must not demand kernel ≈ rank1

**Files:**
- Modify: `post_flash_dse.py` only if any existing check compares kernel latency to rank1.

Today `architecture_ok` is DSP-only. **Leave that.** Add a comment + log via `compute_architecture_ok` so a 13160 kernel with DSP 352 is promoted.

- [ ] **Step 1: Grep `post_flash_dse.py` for `4228`, `rank1`, `latency_cycles` used as a fail.** Remove any such fail. Log `compute_architecture_ok(...).reason`.

- [ ] **Step 2: Run** `python -m pytest tests/test_post_flash_dse.py -v`

---

### Task 5: mm-only campaign = flash → compute → I/O

**Files:**
- Create: `scripts/pc2/batch_parallel_autosa_mm_flow.json` (copy `batch_parallel_autosa_flash_enforcement_mm.json`: one bench `autosa_mm`, `job_prefix` e.g. `mmflow`)
- Create: `scripts/pc2/start_autosa_mm_flow.sh`

Env the launcher **must** export (and pin in sbatch `--export` the same way enforcement was pinned):

```bash
export C2HLS_AUTOSA_FLOW=1
export C2HLS_ENFORCEMENT=0
export C2HLS_POST_FLASH_DSE=1
export C2HLS_DSE_CHAIN_FLASH=1
export C2HLS_POST_FLASH_STREAM=1
export C2HLS_STREAM_CHAIN_FLASH=1
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_RUN_COSIM=0
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_SYNTH_TIMEOUT=3600
```

`start_autosa_mm_flow.sh` mirrors `start_autosa_mm_enforcement.sh` but points at the new json and **does not** call `start_autosa_flash_enforcement.sh`.

- [ ] **Step 1: Unit test** that campaign json + helper export sets `C2HLS_AUTOSA_FLOW=1` and `C2HLS_ENFORCEMENT=0` (extend `tests/test_batch_parallel_job_prefix.py` or a small new test).

- [ ] **Step 2: Dry-run** `./scripts/pc2/start_autosa_mm_flow.sh --dry-run` (or whatever the sibling scripts use) and confirm the printed env.

---

### Task 6: One-pager for the supervisor

**Files:**
- Create: `docs/pc2/2026-08-30-autosa-agentic-flow-one-pager.md`

Contents (keep human; no loop names):

- AutoSA = many workers, then memory overlapping them.
- Agent: flash (correct engine) → compute rewrite → I/O rewrite. Knobs fixed (16 workers × 4-wide for this mm).
- We **do not** start with two folders on one worker.
- Pass compute: lots of DSP / many units even if the whole job is still load-then-compute-then-store.
- Pass I/O: finish time of one run ≈ the slowest room.
- Measured path we already have: ~13k (compute done) → ~7k (overlap but refill) → **4285** (continuous inner work) vs AutoSA **4228**.

---

## Out of scope (this plan)

- Eight LLM stages named space-time / tiling / SIMD / packing / …
- Re-running AutoSA’s auto-tuner
- Cloning `kernel0`
- Changing the locked Aug 18 slide numbers unless a **new** cell beats 4285
- Other six GEMMs (user said leave them unless a later plan)

---

## Spec coverage

| Requirement | Task |
|------------|------|
| Two construction jobs not eight | Header + Task 5 (reuse dse+stream) |
| Workers before overlap | Task 2 skip enforcement; Task 5 order dse then stream |
| Compute may stay ~3× | Task 4 |
| I/O = one run ≈ slowest room | Task 1 + Task 3 |
| Flattening is a skill not a stage | Stream already has it; Task 3 repair bullet |
| Recipe knobs | Unchanged `post_flash_pe_recipe.py` |
| Supervisor language | Task 6 |
| mm-only runner | Task 5 |
