# Post-flash Latency-Opt Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional post-flash/post-dataflow latency-opt pass that improves `latency_cycles` under ≤100% device budget via analysis pack → LLM plan → LLM modify → validate/repair, with full trajectory artifacts and env toggles for A/B.

**Architecture:** New `post_flash_latency_opt.py` mirrors `post_flash_pragma_opt.py`. Deterministic renderers build cycle/bottleneck/template advice from `hls_feedback` + `rubric._device_limits_for_part`. Each of N rounds: plan LLM call, modify LLM call, csim+csynth gate (accept only lower latency and under budget), up to R repairs on failure, always restore `best_so_far`. Chain after pragma_opt at flash and dataflow sites; dataflow source resolution prefers successful latency_opt kernels.

**Tech Stack:** Python 3, existing `c2hls` orchestrator `_call_llm` / `_run_synth_csim_cosim`, `hls_feedback`, `rubric` device table, pytest-style `tests/*.py` (repo often runs tests as scripts).

**Spec:** `docs/superpowers/specs/2026-07-19-c2hls-post-flash-latency-opt-design.md`

---

## File map

| Path | Role |
|------|------|
| `post_flash_latency_opt.py` | Env, analysis pack, templates, plan/modify/repair prompts, round loop, accept logic, trajectory I/O, `maybe_chain_latency_opt` |
| `tests/test_post_flash_latency_opt.py` | Unit tests (no real Vitis): env, budget, accept/reject, analysis render, seed legalization |
| `post_flash_mem_parallel.py` | Extend `resolve_selected_kernel` to prefer successful `*_latency_opt.cpp` then `*_pragma_opt.cpp` |
| `c2hls.py` | After `maybe_chain_pragma_opt` flash chain, call `maybe_chain_latency_opt` |
| `post_flash_dataflow.py` | After dataflow pragma_opt chain, call `maybe_chain_latency_opt` |
| `scripts/pc2/run_post_flash_latency_opt.py` | Standalone batch runner (mirror pragma_opt) |
| `scripts/pc2/start_post_flash_latency_opt.sh` | PC2 starter |

---

### Task 1: Env helpers + budget/accept pure functions (TDD)

**Files:**
- Create: `post_flash_latency_opt.py` (stubs + pure helpers first)
- Create: `tests/test_post_flash_latency_opt.py`

- [ ] **Step 1: Write failing tests for env + budget + accept**

```python
# tests/test_post_flash_latency_opt.py
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import post_flash_latency_opt as plo


def test_enabled_default_off():
    os.environ.pop("C2HLS_POST_FLASH_LATENCY_OPT", None)
    assert plo.latency_opt_enabled() is False


def test_enabled_on():
    os.environ["C2HLS_POST_FLASH_LATENCY_OPT"] = "1"
    assert plo.latency_opt_enabled() is True
    del os.environ["C2HLS_POST_FLASH_LATENCY_OPT"]


def test_rounds_defaults():
    os.environ.pop("C2HLS_LATENCY_OPT_ROUNDS", None)
    os.environ.pop("C2HLS_LATENCY_OPT_REPAIR_ROUNDS", None)
    assert plo.latency_round_limit() == 3
    assert plo.repair_round_limit() == 3


def test_under_budget_u280():
    part = "xcu280-fsvh2892-2L-e"
    report = {"lut": 1000, "dsp": 10, "ff": 2000, "bram": 0, "uram": 0}
    assert plo.under_device_budget(report, part, budget_pct=100.0) is True
    report_over = {"lut": 1000, "dsp": 20000, "ff": 2000, "bram": 0, "uram": 0}
    assert plo.under_device_budget(report_over, part, budget_pct=100.0) is False


def test_should_accept_candidate():
    best = {"latency_cycles": 1000, "report": {"lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}}
    cand_ok = {"latency_cycles": 800, "report": {"lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}}
    assert plo.should_accept(cand_ok, best, part="xcu280-fsvh2892-2L-e") is True
    cand_worse = {"latency_cycles": 1200, "report": cand_ok["report"]}
    assert plo.should_accept(cand_worse, best, part="xcu280-fsvh2892-2L-e") is False
    cand_over = {"latency_cycles": 100, "report": {"lut": 1, "dsp": 99999, "ff": 1, "bram": 0, "uram": 0}}
    assert plo.should_accept(cand_over, best, part="xcu280-fsvh2892-2L-e") is False


def test_should_accept_legalization_when_no_best():
    cand = {"latency_cycles": 5000, "report": {"lut": 1, "dsp": 1, "ff": 1, "bram": 0, "uram": 0}}
    assert plo.should_accept(cand, best=None, part="xcu280-fsvh2892-2L-e") is True


if __name__ == "__main__":
    test_enabled_default_off()
    test_enabled_on()
    test_rounds_defaults()
    test_under_budget_u280()
    test_should_accept_candidate()
    test_should_accept_legalization_when_no_best()
    print("test_post_flash_latency_opt: ok")
```

- [ ] **Step 2: Run tests — expect fail (module missing)**

Run: `cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls && python3 tests/test_post_flash_latency_opt.py`  
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement pure helpers in `post_flash_latency_opt.py`**

```python
"""Post-pass constrained latency optimization with trajectory tracking."""
from __future__ import annotations

import os
from typing import Any, Optional

DEFAULT_ROUNDS = 3
DEFAULT_REPAIR_ROUNDS = 3
DEFAULT_BUDGET_PCT = 100.0
STEP_TAG = "latency_opt"
DATAFLOW_STEP_TAG = "dataflow"


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def latency_opt_enabled() -> bool:
    return _truthy("C2HLS_POST_FLASH_LATENCY_OPT")


def chain_after_flash() -> bool:
    raw = os.getenv("C2HLS_LATENCY_OPT_CHAIN_FLASH", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return latency_opt_enabled()


def chain_after_dataflow() -> bool:
    raw = os.getenv("C2HLS_LATENCY_OPT_CHAIN_DATAFLOW", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return latency_opt_enabled()


def latency_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_LATENCY_OPT_ROUNDS", str(DEFAULT_ROUNDS))))
    except ValueError:
        return DEFAULT_ROUNDS


def repair_round_limit() -> int:
    try:
        return max(1, int(os.getenv("C2HLS_LATENCY_OPT_REPAIR_ROUNDS", str(DEFAULT_REPAIR_ROUNDS))))
    except ValueError:
        return DEFAULT_REPAIR_ROUNDS


def budget_pct() -> float:
    try:
        return float(os.getenv("C2HLS_LATENCY_OPT_BUDGET_PCT", str(DEFAULT_BUDGET_PCT)))
    except ValueError:
        return DEFAULT_BUDGET_PCT


def _device_capacity(part: str) -> dict[str, float]:
    from rubric import _device_limits_for_part
    caps = dict(_device_limits_for_part(part) or {})
    caps.pop("_fallback_reason", None)
    return {k: float(v) for k, v in caps.items() if isinstance(v, (int, float))}


def under_device_budget(
    report: Optional[dict[str, Any]],
    part: str,
    *,
    budget_pct: float = 100.0,
) -> bool:
    if not report:
        return False
    caps = _device_capacity(part)
    limit = budget_pct / 100.0
    for key in ("lut", "dsp", "ff", "bram", "uram"):
        used = report.get(key)
        if used is None:
            used = 0
        try:
            used_f = float(used)
        except (TypeError, ValueError):
            used_f = 0.0
        cap = caps.get(key)
        if not cap or cap <= 0:
            continue
        if used_f / cap > limit + 1e-12:
            return False
    return True


def should_accept(
    candidate: dict[str, Any],
    best: Optional[dict[str, Any]],
    *,
    part: str,
    budget_pct_value: Optional[float] = None,
) -> bool:
    pct = budget_pct() if budget_pct_value is None else budget_pct_value
    report = candidate.get("report") or {}
    if not under_device_budget(report, part, budget_pct=pct):
        return False
    lat = candidate.get("latency_cycles")
    if lat is None:
        return False
    try:
        lat_i = int(lat)
    except (TypeError, ValueError):
        return False
    if best is None:
        return True  # legalization: first under-budget validated design
    try:
        best_lat = int(best["latency_cycles"])
    except (KeyError, TypeError, ValueError):
        return True
    return lat_i < best_lat
```

- [ ] **Step 4: Re-run tests — expect pass**

Run: `python3 tests/test_post_flash_latency_opt.py`  
Expected: `test_post_flash_latency_opt: ok`

- [ ] **Step 5: Commit** (only if user requested commits in this session)

```bash
git add post_flash_latency_opt.py tests/test_post_flash_latency_opt.py
git commit -m "$(cat <<'EOF'
feat: add latency-opt env helpers and constrained accept gate

EOF
)"
```

---

### Task 2: Analysis pack renderer + template actions (TDD)

**Files:**
- Modify: `post_flash_latency_opt.py`
- Modify: `tests/test_post_flash_latency_opt.py`

- [ ] **Step 1: Add failing tests for analysis pack**

```python
def test_render_analysis_pack_includes_scopes_and_templates():
    report = {
        "latency_cycles": 10000,
        "lut": 100, "dsp": 8, "ff": 200, "bram": 0, "uram": 0,
        "feedback": {
            "summary": {
                "loop_count": 2, "pipelined_loops": 1, "bottleneck_count": 1,
                "scopes_with_negative_slack": 0, "high_severity_bottlenecks": 1,
            },
            "scopes": [
                {
                    "scope_id": "k/outer", "kind": "loop", "latency_cycles": 9000,
                    "trip_count": 64, "interval": 64, "pipelined": "no", "pipeline_ii": None,
                    "dsp": 0, "lut": 50,
                },
                {
                    "scope_id": "k/inner", "kind": "loop", "latency_cycles": 128,
                    "trip_count": 64, "interval": 2, "pipelined": "yes", "pipeline_ii": 2,
                    "dsp": 8, "lut": 40,
                },
            ],
            "bottlenecks": [
                {
                    "kind": "ii_target_miss", "severity": "high",
                    "scope_id": "k/inner", "evidence": "II=2 target=1",
                },
                {
                    "kind": "non_pipelined_hot_loop", "severity": "high",
                    "scope_id": "k/outer", "evidence": "not pipelined",
                },
            ],
        },
    }
    text = plo.render_latency_analysis_pack(report, part="xcu280-fsvh2892-2L-e")
    assert "10000" in text
    assert "k/outer" in text and "k/inner" in text
    assert "ii_target_miss" in text
    assert "pipeline" in text.lower()
    assert "guided" in text.lower()


def test_template_actions_resource_pressure():
    report = {
        "latency_cycles": 10,
        "lut": 1200000, "dsp": 1, "ff": 1, "bram": 0, "uram": 0,
        "feedback": {"scopes": [], "bottlenecks": [], "summary": {}},
    }
    text = plo.render_latency_analysis_pack(report, part="xcu280-fsvh2892-2L-e")
    assert "80" in text or "pressure" in text.lower() or "budget" in text.lower()
```

- [ ] **Step 2: Run — expect fail on missing `render_latency_analysis_pack`**

- [ ] **Step 3: Implement renderer + templates**

In `post_flash_latency_opt.py` add:

- `template_actions_for_report(report, part) -> list[str]` mapping:
  - `non_pipelined_hot_loop` → `#pragma HLS PIPELINE II=1` on that scope
  - `ii_target_miss` → lower II via dependence / partition / unroll
  - `port_conflict` → `array_partition` + matching unroll
  - resource util >80% / >100% → pressure / overflow guidance
- `render_latency_analysis_pack(report, part, *, max_scopes=12, max_bottlenecks=6, trajectory_summary="") -> str` with Design PPA, ranked scope table, bottlenecks, **Guided actions**, budget lines (richer than `hls_feedback.render_feedback_for_prompt`).

- [ ] **Step 4: Re-run tests — expect pass**

- [ ] **Step 5: Commit** (if requested)

---

### Task 3: Prompts (plan / modify / repair) + extract helpers

**Files:**
- Modify: `post_flash_latency_opt.py`
- Modify: `tests/test_post_flash_latency_opt.py`

- [ ] **Step 1: Tests for prompt contents**

```python
def test_plan_and_modify_prompts_structure():
    docs = plo.prompt_text_for_docs()
    assert "analyst" in docs["plan_system"].lower() or "plan" in docs["plan_system"].lower()
    assert "target" in docs["plan_user"].lower() or "action" in docs["plan_user"].lower()
    assert "kernel" in docs["modify_system"].lower()
    assert "kernel" in docs["modify_user"].lower()
    assert "error" in docs["repair_user"].lower() or "fix" in docs["repair_user"].lower()
```

- [ ] **Step 2: Implement prompt templates**

Constants (mirror pragma_opt style):

- `_PLAN_SYSTEM` — HLS performance analyst; structured plan only (`targets` / `actions` / `avoid` / `risk`); cite `scope_id`s from analysis.
- `_PLAN_USER` — `{analysis_pack}`, `{kernel_code}`, `{best_latency}`, `{budget_block}`, `{trajectory}`
- `_MODIFY_SYSTEM` — apply plan; preserve `extern "C"` signature and INTERFACE pragmas; one ```kernel``` block.
- `_MODIFY_USER` — `{plan_text}`, `{kernel_code}`, `{budget_block}`, optional truncated `{pragma_guide}`
- `_REPAIR_USER` — `{stage}`, `{error}`, `{plan_text}`, `{kernel_code}`, `{analysis_pack}`

Add `prompt_text_for_docs()` for docs/smoke.  
Reuse `extract_kernel_block` from `post_flash_dataflow`.  
Optional soft check: `plan_mentions_scope(plan_text, scope_ids) -> bool` — warn if false; do not hard-fail v1.

- [ ] **Step 3: Run tests — pass**

- [ ] **Step 4: Commit** (if requested)

---

### Task 4: Artifact paths + trajectory schema helpers

**Files:**
- Modify: `post_flash_latency_opt.py`
- Modify: `tests/test_post_flash_latency_opt.py`

- [ ] **Step 1: Test artifact naming**

```python
def test_artifact_paths():
    cell = Path("/tmp/cell")
    flash = plo.artifact_paths(cell, "atax", "flash_final")
    df = plo.artifact_paths(cell, "atax", "dataflow")
    assert flash["kernel"].name == "atax_latency_opt.cpp"
    assert flash["trajectory"].name == "atax_latency_opt_trajectory.json"
    assert df["kernel"].name == "atax_dataflow_latency_opt.cpp"
```

- [ ] **Step 2: Implement helpers**

`artifact_paths(cell_dir, bench, source_role) -> dict` with keys: `kernel`, `report`, `result`, `history`, `trajectory`, `manifest`.

`new_trajectory(...)`, `append_round_event(traj, event)`, `set_best_so_far(traj, ...)`.  
Schema: `post_flash_latency_opt_trajectory_v1` per spec (seed, best_so_far, rounds with `phase` in `plan|optimize|repair`, `plan_summary`, final).

- [ ] **Step 3: Tests pass + commit if requested**

---

### Task 5: Core round loop `run_latency_opt_for_cell` (mock LLM/synth)

**Files:**
- Modify: `post_flash_latency_opt.py`
- Modify: `tests/test_post_flash_latency_opt.py`

- [ ] **Step 1: Mocked end-to-end unit test**

```python
class FakeOrch:
    part = "xcu280-fsvh2892-2L-e"
    clock_ns = 3.33
    gpt_model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0

    def _call_llm(self, messages):
        r = self.replies[self.i]
        self.i += 1
        return r
```

Monkeypatch `c2hls._run_synth_csim_cosim` (same import path pragma_opt uses) to return seed-valid then candidate with lower latency under budget.  
Set `C2HLS_LATENCY_OPT_ROUNDS=1`, `C2HLS_LATENCY_OPT_REPAIR_ROUNDS=1`.  
Assert success, trajectory has plan+optimize, final latency &lt; seed, kernel file written.

- [ ] **Step 2: Implement `run_latency_opt_for_cell`**

1. Resolve source (prefer successful pragma_opt for role, else selected/dataflow — Task 6 helpers).
2. Load bench inputs like pragma_opt (`_load_benchmark_inputs`, header, tb, part, clock).
3. Validate seed; if invalid → repair ≤R; if still invalid → write trajectory, return failure.
4. If seed under budget → `best_so_far = seed`; else `best_so_far = None`.
5. For round `1..N`: analysis pack → plan LLM → modify LLM → validate; fail → repair×R; `should_accept` → update best or reject.
6. Write best kernel/report/result/trajectory/history/manifest; cosim off.

`LatencyOptOutcome` dataclass: `bench, source_role, success, cell_dir, error, result`.

Reuse: `compile_check_cpp`, `_run_synth_csim_cosim`, `sha256_text`, `extract_kernel_block`.

- [ ] **Step 3: Run mock test — pass**

- [ ] **Step 4: Commit** (if requested)

---

### Task 6: Source resolution preference + chain hooks

**Files:**
- Modify: `post_flash_mem_parallel.py` (`resolve_selected_kernel`)
- Modify: `post_flash_latency_opt.py` (`resolve_latency_source_kernel`, `maybe_chain_latency_opt`)
- Modify: `c2hls.py` (~9005–9017)
- Modify: `post_flash_dataflow.py` (~1148–1168)
- Modify: `tests/test_post_flash_latency_opt.py`

- [ ] **Step 1: Prefer post-pass kernels for downstream**

Update `resolve_selected_kernel` order:

1. `{bench}_latency_opt.cpp` if `{bench}_latency_opt_result.json` has `success: true`
2. `{bench}_pragma_opt.cpp` if `{bench}_pragma_opt_result.json` has `success: true`
3. existing selected/final logic

In latency_opt, `resolve_latency_source_kernel(cell_dir, bench, source_role)`:

- `flash_final`: prefer successful `{bench}_pragma_opt.cpp`, else `resolve_selected_kernel` (without re-entering latency_opt preference recursively — call a private base resolver or pass `prefer_post=False` flag).
- `dataflow`: prefer successful `{bench}_dataflow_pragma_opt.cpp`, else successful `{bench}_dataflow.cpp`.

**Avoid recursion:** split `resolve_selected_kernel` into `_resolve_flash_base_kernel` + preference wrapper, or check latency_opt only when `include_latency_opt=True` (default True for dataflow consumers; False when latency_opt resolves its own seed).

- [ ] **Step 2: Implement `maybe_chain_latency_opt`** (mirror pragma_opt; swallow exceptions; log)

- [ ] **Step 3: Wire flash chain in `c2hls.py` after pragma_opt**

```python
maybe_chain_pragma_opt(...)
from post_flash_latency_opt import maybe_chain_latency_opt
maybe_chain_latency_opt(
    bench=bench_name,
    bench_dir=BENCHMARKS_DIR / bench_name,
    cell_dir=Path(output_dir),
    orchestrator=orchestrator,
    source_role="flash_final",
    skip_existing=True,
)
```

- [ ] **Step 4: Wire dataflow chain in `post_flash_dataflow.py` after pragma_opt**; attach `latency_opt_chain` on `result_payload`.

- [ ] **Step 5: Test resolve preference**

```python
def test_resolve_prefers_latency_opt(tmp_path):
    from post_flash_mem_parallel import resolve_selected_kernel
    bench = "atax"
    (tmp_path / f"{bench}_selected.cpp").write_text("selected", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt.cpp").write_text("opt", encoding="utf-8")
    (tmp_path / f"{bench}_latency_opt_result.json").write_text(
        '{"success": true}', encoding="utf-8"
    )
    path, role = resolve_selected_kernel(tmp_path, bench)
    assert path.name == f"{bench}_latency_opt.cpp"
```

- [ ] **Step 6: Commit** (if requested)

---

### Task 7: Standalone PC2 runner + starter scripts

**Files:**
- Create: `scripts/pc2/run_post_flash_latency_opt.py` (clone from `run_post_flash_pragma_opt.py`)
- Create: `scripts/pc2/start_post_flash_latency_opt.sh` (clone from `start_post_flash_pragma_opt.sh`)

- [ ] **Step 1: Adapt runner/starter; force-enable docs via env when submitting**
- [ ] **Step 2: Support `--source flash_final|dataflow`, `--show-prompts`, `--dry-run`, matrix discovery**
- [ ] **Step 3: Smoke**

Run: `python3 scripts/pc2/run_post_flash_latency_opt.py --show-prompts | head`  
Expected: plan/modify prompt text visible

- [ ] **Step 4: Commit** (if requested)

---

### Task 8: A/B enable note in spec

**Files:**
- Modify: `docs/superpowers/specs/2026-07-19-c2hls-post-flash-latency-opt-design.md`

- [ ] **Step 1: Append A/B enable snippet**

```markdown
## A/B enable (campaign)

export C2HLS_POST_FLASH_LATENCY_OPT=1
export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
export C2HLS_LATENCY_OPT_ROUNDS=3
export C2HLS_LATENCY_OPT_REPAIR_ROUNDS=3
# leave unset for control campaign
```

- [ ] **Step 2: Run full unit file**

`python3 tests/test_post_flash_latency_opt.py` → `ok`

- [ ] **Step 3: Commit** (if requested)

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Optional env master + chain flash/dataflow | 1, 6 |
| N=3 / R=3 defaults | 1 |
| ≤100% device accept | 1, 5 |
| Analysis pack + templates | 2 |
| Plan then modify LLM | 3, 5 |
| Repair with last plan | 3, 5 |
| Trajectory JSON | 4, 5 |
| Restore best_so_far / legalization | 1, 5 |
| Chain after pragma_opt | 6 |
| Dataflow consumes latency_opt | 6 |
| Standalone runner | 7 |
| Off by default A/B | 1, 8 |

## Self-review

- Spec sections map to tasks; no TBD/placeholder steps.
- Names consistent: `latency_opt_enabled`, `should_accept`, `render_latency_analysis_pack`, `run_latency_opt_for_cell`, `maybe_chain_latency_opt`.
- Device caps via `rubric._device_limits_for_part`.
- Git commits marked optional unless user explicitly requests commits during execution.
