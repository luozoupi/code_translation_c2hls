# Design: Post-flash latency-opt pass (constrained DSE + trajectory)

**Date:** 2026-07-19  
**Status:** Approved (2026-07-19); implemented  
**Scope:** c2hls only. Additive post-pass. Does not change flash/dataflow “first correctness win” stop rules when disabled.

## Goal

Today flash and dataflow stop at the **first** csim+csynth-passing kernel. Skills/RAG focus on II, pipeline, and synthesizability — not a closed loop that **improves latency under a hard device budget** while keeping history.

Add an optional **latency-opt** pass that:

1. Starts from a validated flash-final or dataflow kernel.
2. Runs up to **N** latency-improvement rounds with csynth PPA feedback and trajectory context.
3. On validation failure, runs up to **R** repairs for that round.
4. Accepts a candidate only if it is **strictly faster** and **≤100% of the target device** on LUT/DSP/FF/BRAM/URAM.
5. Always finishes with the **last best legal** kernel (never leaves a broken or over-budget design as selected).
6. Is **off by default** so A/B tests compare with vs without the pass.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Architecture | **A** — new sibling module `post_flash_latency_opt` (mirror `post_flash_pragma_opt` / dataflow) |
| Accept rule | Lower `latency_cycles` **and** all of LUT/DSP/FF/BRAM/URAM ≤ **100% device** for `C2HLS_PART` |
| Chain points | **Both** — after flash-final and after dataflow (when that stage succeeds) |
| Round budgets | **N=3** latency rounds; **R=3** repairs per failed round |
| Default | Pass **disabled** (`C2HLS_POST_FLASH_LATENCY_OPT` unset/off) |
| Cosim | Off by default for this pass (same as pragma_opt); csim+csynth required |
| Optimize LLM flow | **B** — deterministic cycle/bottleneck templates **+** short LLM **plan** step, then code **modify** (ChatHLS-like analyze→modify) |

## Non-goals (v1)

- Training / calling ChatHLS HLSTuner weights.
- Changing flash or dataflow inner loops to multi-round latency DSE.
- Soft Pareto or 80%-budget modes (v1 is hard ≤100% device only).
- Replacing `pragma_opt`; both may be enabled (see order below).
- Online model fine-tuning from trajectory logs.

## Problem this solves

Without this pass, c2hls geomean latency lags ChatHLS partly because ChatHLS runs an optimize-for-latency step with csynth feedback. c2hls needs a **toggleable** equivalent that:

- Keeps a **trajectory** (rounds, metrics, accept/reject reasons).
- Enforces **resource legality** so we do not “win” by over-device designs (ChatHLS `matmul` / `3mm` failure mode).

## Pipeline placement

```text
flash (first correctness) 
  → [optional pragma_opt]
  → [optional latency_opt]          ← chain flash
  → [optional dataflow]
  → [optional pragma_opt on dataflow]
  → [optional latency_opt]          ← chain dataflow
```

**Order vs pragma_opt:** when both are on, run **pragma_opt then latency_opt** at each chain site so latency-opt starts from the pragma-improved kernel. Either pass alone is fine.

**Hook sites:** same places as `maybe_chain_pragma_opt` (`c2hls.py` after flash final; `post_flash_dataflow.py` after dataflow success). Add `maybe_chain_latency_opt(...)`.

## Accept / reject / restore

### Seed (round 0)

1. Load flash-final or dataflow kernel.
2. If seed fails csim or csynth: up to **R** repairs aimed at restoring validity (not yet requiring latency improvement). If still invalid → **abort pass**; leave prior selected artifacts unchanged; record failure in trajectory.
3. If seed passes but is **over device budget**: do **not** set it as `best_so_far`. `best_so_far` stays empty until the first **legal** (under budget + pass) candidate. That first legal candidate may have latency ≥ seed (legalization). Later rounds require **strictly lower** latency than `best_so_far`.
4. If seed passes and is under budget: `best_so_far = seed`.

### Latency round `i = 1..N`

Each round is **two LLM calls** (plus optional repairs), not a single free-form rewrite:

1. **Build analysis pack (deterministic):** design PPA, per-scope cycle/parallelism table from `hls_feedback`, typed bottlenecks, resource % vs device, trajectory summary, and **template guided actions** (from bottleneck kinds + hot scopes).
2. **Plan step (LLM):** given analysis pack + kernel (or kernel excerpt), produce a short structured plan (which scopes to touch, which pragmas/transforms, expected latency/resource effect, what *not* to touch). Must follow / refine the templates — not invent unrelated rewrites.
3. **Modify step (LLM):** apply that plan to the full kernel; return one ```kernel``` block. Preserve top signature and INTERFACE pragmas.
4. Validate (csim + csynth).
5. If fail → up to **R** repair turns (error + last plan + constraints). After repairs, re-validate.
6. If still invalid → **discard**; keep `best_so_far`; continue to next latency round (or end).
7. If valid but over budget or latency not better than `best_so_far` → **reject**; keep `best_so_far`.
8. If valid, under budget, and `latency_cycles < best_so_far.latency` → **accept**; update `best_so_far` and working kernel.
9. Early stop: v1 default **no early stop** (always run N) for predictable A/B cost.

### Exit

Write selected kernel = `best_so_far` if any legal best exists; else leave input kernel unchanged (and mark `success=false` if seed never legalized). Never promote an invalid or over-budget kernel as the selected output.

## Resource budget check

- Part from `C2HLS_PART` (campaign default `xcu280-fsvh2892-2L-e`).
- Compare report totals (LUT, DSP, FF, BRAM, URAM) to device availability for that part (reuse or add a small helper next to existing report parsing in `hls_eval` / feedback; ChatHLS CSV already uses avail_* for U280 — mirror those numbers for U280 and keep a part→capacity map extensible).
- Threshold: **≤ 100%** of each resource (equality allowed). Missing URAM/BRAM in report → treat as 0 used.
- Env override (optional v1.1): `C2HLS_LATENCY_OPT_BUDGET_PCT=100` (default 100). v1 ships 100 only; keep knob for future.

## Env / CLI

| Env | Default | Meaning |
|-----|---------|---------|
| `C2HLS_POST_FLASH_LATENCY_OPT` | off | Master enable |
| `C2HLS_LATENCY_OPT_CHAIN_FLASH` | follows master | Chain after flash-final |
| `C2HLS_LATENCY_OPT_CHAIN_DATAFLOW` | follows master | Chain after dataflow success |
| `C2HLS_LATENCY_OPT_ROUNDS` | `3` | N |
| `C2HLS_LATENCY_OPT_REPAIR_ROUNDS` | `3` | R |
| `C2HLS_LATENCY_OPT_BUDGET_PCT` | `100` | Device % cap (v1 fixed intent) |
| `C2HLS_RUN_COSIM` / cosim required | off for this pass | Same pattern as pragma_opt |

Batch flavors (later, optional): e.g. `rag2_skills_lat` with `C2HLS_POST_FLASH_LATENCY_OPT=1` for A/B vs existing flavors.

No new required CLI flag for v1 if env-only matches pragma_opt; optional `--latency-opt` alias may be added for local runs.

## Module layout

| Path | Role |
|------|------|
| `post_flash_latency_opt.py` | Enable/chain helpers, prompts, round loop, accept logic, trajectory I/O, `maybe_chain_latency_opt` |
| `tests/test_post_flash_latency_opt.py` | Unit tests: accept/reject, over-budget, restore best, env parsing (mock synth) |
| `scripts/pc2/run_post_flash_latency_opt.py` | Optional standalone batch runner (mirror pragma_opt) |
| `scripts/pc2/start_post_flash_latency_opt.sh` | Optional PC2 starter |
| Chain call sites | `c2hls.py`, `post_flash_dataflow.py` after pragma_opt |

## Artifacts (per cell)

Stem examples:

- Flash: `{bench}_latency_opt.*`
- Dataflow: `{bench}_dataflow_latency_opt.*`

| File | Content |
|------|---------|
| `*_latency_opt.cpp` | Final selected (best legal) kernel |
| `*_latency_opt_report.json` | Csynth report for selected |
| `*_latency_opt_result.json` | success, N/R used, baseline vs final latency, budget ok |
| `*_latency_opt_trajectory.json` | Full history (schema below) |
| `*_latency_opt_history.json` | LLM message log |
| `*_latency_opt_manifest.json` | Pointer for matrix tooling |

### Trajectory schema (`post_flash_latency_opt_trajectory_v1`)

```json
{
  "schema": "post_flash_latency_opt_trajectory_v1",
  "benchmark": "...",
  "source_role": "flash_final|dataflow",
  "part": "xcu280-fsvh2892-2L-e",
  "budget_pct": 100,
  "N": 3,
  "R": 3,
  "seed": {
    "latency_cycles": 0,
    "resources": {},
    "under_budget": true,
    "validated": true
  },
  "best_so_far": {
    "round": 0,
    "latency_cycles": 0,
    "resources": {},
    "kernel_sha256": "..."
  },
  "rounds": [
    {
      "round": 1,
      "phase": "plan|optimize|repair",
      "repair_index": null,
      "plan_summary": null,
      "validated": false,
      "latency_cycles": null,
      "resources": {},
      "under_budget": null,
      "decision": "accept|reject_latency|reject_budget|reject_invalid|restore",
      "reason": "..."
    }
  ],
  "final": {
    "latency_cycles": 0,
    "speedup_vs_seed": 1.0,
    "under_budget": true,
    "success": true
  }
}
```

## Prompting (v1) — analysis-guided plan → modify

**Principle:** kernel source alone is insufficient. Every optimize round must supply **cycle-level analysis** (loops/modules: latency, trip, II, pipelined, local resources, share of parent) and **guided parallelism advice**, then a short LLM plan, then a code edit.

### Analysis pack (deterministic; required in both plan and modify prompts)

Built from `attach_feedback` / report + `_resource_utilization`:

| Block | Content |
|-------|---------|
| Design PPA | `latency_cycles`, interval, LUT/DSP/FF/BRAM/URAM used/cap/% |
| Scope table | Ranked hot scopes: `scope_id`, kind, lat, trip, II/interval, pipelined, iteration_latency, DSP/LUT if present, rough share of parent latency |
| Bottlenecks | Typed list from feedback (`ii_target_miss`, `non_pipelined_hot_loop`, `port_conflict`, …) with evidence |
| Template actions | Ordered bullets from bottleneck kind → advice (see below) |
| Budget / trajectory | ≤100% device rule; prior rounds’ accept/reject reasons |

Reuse/extend `render_feedback_for_prompt`; add a dedicated **latency-opt renderer** that always includes the scope table + template actions (richer than repair’s compact dump).

### Template action examples (deterministic)

| Signal | Guided action |
|--------|----------------|
| Hot loop not pipelined | `#pragma HLS PIPELINE II=1` on that loop; label the loop |
| `ii_target_miss` | Reduce II: fix dependences / partition locals / modest unroll; state target II |
| Port conflict / low parallelism on hot MAC loop | `array_partition` on local tiles + matching `UNROLL factor` |
| Memory-bound outer + cold inner | Prefer `m_axi` bundle/burst/widen on hot loads; do not unroll cold loops |
| Resource >80% already | Prefer schedule/II fixes; forbid large new unroll/partition factors |
| Resource >100% on candidate | Reject at gate; next plan must reduce that resource |

### Call 1 — Plan (LLM)

- **System:** HLS performance analyst. Output a concise plan only (no full kernel).
- **User:** analysis pack + current best kernel + goal (latency &lt; best, ≤100% device).
- **Expected output (structured text or JSON):**
  - `targets`: list of `scope_id`s to change (max ~3–5)
  - `actions`: pragma/transform per target, tied to template actions
  - `avoid`: scopes/resources not to worsen
  - `risk`: e.g. DSP growth estimate qualitative
- Plan must **reference** analysis scopes; reject empty/generic plans in post-check if trivial (optional soft check: require ≥1 known `scope_id` from the table).

### Call 2 — Modify (LLM)

- **System:** HLS code editor. Apply the plan exactly; preserve interfaces; one ```kernel``` block.
- **User:** plan + full kernel + short reminder of budget + optional truncated pragma guide.
- Do **not** re-diagnose from scratch; follow the plan.

### Repair (LLM; up to R)

- Failing stage + error + **last plan** + current kernel + analysis pack snapshot.
- Goal: restore csim/csynth while staying as close as possible to the plan; never accept over-budget.

### Skills / RAG

Optional injection when already enabled in the campaign; not required for v1. Prefer II/latency skills matched to top bottleneck kinds when present.

## Testing / A/B

1. **Unit:** accept only when lower latency + under budget; over-budget reject; invalid → repair → restore best; seed over-budget legalization; env off → no-op.
2. **Smoke:** one bench with `LATENCY_OPT=1`, N=1, R=1 on a known small kernel.
3. **Campaign A/B:** same DeepSeek/GLM U280 flavor with and without `C2HLS_POST_FLASH_LATENCY_OPT=1`; compare geomean latency and over-device rate.

## Success criteria

- With pass **off**, bit-identical behavior to today (no chain calls effective).
- With pass **on**, every selected `*_latency_opt.cpp` that `success=true` is under 100% device and ≤ seed latency when seed was already legal (or ≤ first legalized best thereafter).
- Trajectory JSON present for every attempted cell.
- Documented env toggles for with/without experiments.

## Open points (non-blocking for v1)

- Exact part→capacity table source (hardcode U280 avail from ChatHLS CSV vs query Vivado) — prefer shared constant map with U280 numbers matching `final_resources_csynth.csv` avail_* columns.
- **Resolved:** successful latency-opt **updates** canonical selected pointers (`{bench}_selected.*` for flash; `dataflow_result.json` `selected_stage` / `latency_cycles` for dataflow). Downstream dataflow therefore consumes flash latency-opt via the updated selected kernel. Resolvers and inventory prefer `*_latency_opt` / `*_dataflow_latency_opt` when present.
- Early-stop env if N rounds with zero accepts — defer; default run all N.

## A/B enable (campaign)

export C2HLS_POST_FLASH_LATENCY_OPT=1
export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
export C2HLS_LATENCY_OPT_ROUNDS=3
export C2HLS_LATENCY_OPT_REPAIR_ROUNDS=3
# leave unset for control campaign
